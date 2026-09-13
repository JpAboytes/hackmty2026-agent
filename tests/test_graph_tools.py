"""Focused financial graph and presentation-selection tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

import pytest
from fastmcp.client.client import CallToolResult
from langgraph.graph import END
from mcp.types import TextContent

import fluidbank_orchestrator.agent.gemini as gemini_module
import fluidbank_orchestrator.agent.nodes as nodes_module
from fluidbank_orchestrator.agent.gemini import GeminiToolAwareModel, _api_failure_reason, _Intent
from fluidbank_orchestrator.agent.nodes import route_after_agent
from fluidbank_orchestrator.agent.tool_loop import run_pending_tools
from fluidbank_orchestrator.agent.tool_visibility import model_tool_schema
from fluidbank_orchestrator.graph import ModelTurn, ToolAwareModel, build_graph
from fluidbank_orchestrator.mcp_client import (
    MCPToolDefinition,
    MCPToolExecution,
    TrustedUserScopeError,
    UserContext,
    UserContextError,
    enforce_trusted_user_scope,
)
from fluidbank_orchestrator.schemas.a2ui import A2UIBundle
from fluidbank_orchestrator.services.financial_presentation import build_financial_presentation
from fluidbank_orchestrator.state import UserProfile

USER_A = UUID("68dc4d66-07b8-5893-95f1-07f06989a552")
USER_B = UUID("c1a3797d-b335-5a9d-98a1-402311f82c7a")
PROFILE: UserProfile = {
    "literacy_level": "medium",
    "font_scale": "lg",
    "contrast": "high",
    "hit_target": "large",
    "overdraft_risk": 0.1,
    "recurring_expenses": 200.0,
    "available_balance": 150.0,
    "owned_balances": {"MXN": 150.0},
}


class FakeModel(ToolAwareModel):
    async def generate(
        self,
        *,
        query: str,
        profile: UserProfile,
        tools: Sequence[MCPToolDefinition],
        observations: Sequence[Mapping[str, Any]],
    ) -> ModelTurn:
        del query, profile, tools, observations
        return ModelTurn(message="Hola, ¿en qué te ayudo?")


async def _discovery_tools() -> list[MCPToolDefinition]:
    """What the MCP server advertises under progressive discovery.

    The full domain catalog is reachable through these two, never listed.
    """
    return [
        MCPToolDefinition(
            name="search_tools",
            description="Search for tools using natural language.",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        ),
        MCPToolDefinition(
            name="call_tool",
            description="Call a tool by name with the given arguments.",
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
            },
        ),
    ]


def _execution(rows: list[dict[str, Any]]) -> MCPToolExecution:
    return MCPToolExecution(
        result=CallToolResult(
            content=[TextContent(text="Rows loaded.")],
            structured_content={"ok": True, "rows": rows},
            meta=None,
            data=None,
        ),
        a2ui=None,
    )


async def _run_graph(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    rows: list[dict[str, Any]],
    context_rows: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_profile(_current_user_id: UUID) -> UserContext:
        return UserContext(profile=PROFILE.copy(), rows=dict(context_rows or {}))

    async def execute(
        name: str,
        arguments: Mapping[str, Any] | None,
        *,
        current_user_id: UUID | None = None,
    ) -> MCPToolExecution:
        resolved = enforce_trusted_user_scope(name, arguments, current_user_id) or {}
        calls.append((name, resolved))
        return _execution(rows)

    monkeypatch.setattr(nodes_module, "fetch_user_context", fake_profile)
    result = await build_graph(
        model=FakeModel(), tool_loader=_discovery_tools, tool_executor=execute
    ).ainvoke({"user_query": query, "current_user_id": USER_A})
    return result, calls


@pytest.mark.asyncio
async def test_balance_request_selects_financial_summary_not_chat_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "id": "checking",
            "account_type": "checking",
            "currency": "MXN",
            "available_balance": 100,
        },
        {
            "id": "savings",
            "account_type": "savings",
            "currency": "MXN",
            "available_balance": 50,
        },
        {
            "id": "credit",
            "account_type": "credit",
            "currency": "USD",
            "available_balance": 900,
        },
    ]
    result, calls = await _run_graph(
        monkeypatch,
        "¿Cuánto dinero tengo?",
        [],
        context_rows={"accounts": rows, "cards": []},
    )

    assert calls == []
    presentation = result["financial_presentation"]
    assert presentation.intent == "financial-summary"
    assert presentation.data["owned_balance"] == 150
    assert presentation.a2ui.messages[0]["createSurface"]["catalogId"].endswith("/finance/v2")
    components = presentation.a2ui.messages[1]["updateComponents"]["components"]
    assert {component["component"] for component in components} == {
        "Column",
        "BankingView",
        "Text",
        "Button",
    }
    view = presentation.a2ui.messages[-1]["updateDataModel"]["value"]["view"]
    assert view["totalOwnedBalance"] == 150
    assert all(account["accountType"] != "credit" for account in view["accounts"])


@pytest.mark.asyncio
async def test_context_rows_answer_a_balance_without_a_second_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The profile already read accounts; the turn must not read them again."""
    accounts = [
        {
            "id": "checking",
            "account_type": "checking",
            "currency": "MXN",
            "available_balance": 150,
        }
    ]
    result, calls = await _run_graph(
        monkeypatch,
        "¿Cuánto dinero tengo?",
        [],
        context_rows={"accounts": accounts},
    )

    assert calls == []
    presentation = result["financial_presentation"]
    assert presentation.intent == "financial-summary"
    assert presentation.data["owned_balance"] == 150


@pytest.mark.asyncio
async def test_a_table_the_profile_never_read_is_still_fetched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefetching accounts must not suppress an unrelated domain read."""
    transactions = [
        {
            "id": "t1",
            "amount": 20,
            "direction": "debit",
            "category": "food",
            "merchant": "Cafe",
            "occurred_at": "2026-09-01T10:00:00+00:00",
        }
    ]
    _result, calls = await _run_graph(
        monkeypatch,
        "Muéstrame mis movimientos",
        transactions,
        context_rows={"accounts": [{"id": "checking"}]},
    )

    assert [name for name, _ in calls] == ["get_transactions"]
    assert calls[0][1]["request"]["scope"] == {"user_id": str(USER_A)}


def test_spending_analysis_retains_and_combines_multiple_tool_results() -> None:
    observations = [
        {
            "name": "get_transactions",
            "arguments": {"table": "transactions"},
            "is_error": False,
            "data": {
                "transactions": [
                    {
                        "id": "food-1",
                        "amount": 75,
                        "direction": "debit",
                        "category": "dining",
                        "merchant": "Café",
                        "occurred_at": "2026-09-10T14:00:00+00:00",
                    }
                ]
            },
        },
        {
            "name": "get_transactions",
            "arguments": {"table": "transactions", "offset": 1},
            "is_error": False,
            "data": {
                "transactions": [
                    {
                        "id": "transport-1",
                        "amount": 25,
                        "direction": "debit",
                        "category": "transport",
                        "merchant": "Metro",
                        "occurred_at": "2026-09-11T15:00:00+00:00",
                    }
                ]
            },
        },
    ]

    presentation = build_financial_presentation("spending-analysis", observations, PROFILE)
    view = presentation.a2ui.messages[-1]["updateDataModel"]["value"]["view"]

    assert presentation.data["source_result_count"] == 2
    assert presentation.data["expense_total"] == 100
    assert {item["category"] for item in view["categories"]} == {"food", "transport"}
    assert view["trend"]["data"]
    assert view["activity"]["data"]


@pytest.mark.asyncio
async def test_semantic_activity_request_does_not_require_chart_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "id": "expense",
            "amount": 20,
            "direction": "debit",
            "category": "dining",
            "merchant": "Café",
            "occurred_at": "2026-09-11T15:00:00+00:00",
        }
    ]
    result, calls = await _run_graph(monkeypatch, "¿Qué días gasto más?", rows)
    assert [name for name, _ in calls] == ["analyze_spending"]
    assert calls[0][1]["request"]["period"] == "current_month"
    assert result["financial_presentation"].intent == "spending-analysis"


def test_visualization_scope_is_overwritten_without_mutating_model_arguments() -> None:
    model_arguments = {
        "request": {
            "source": {"schema": "public", "table": "transactions"},
            "scope": {"user_id": str(USER_B)},
            "filters": [
                {"column": "user_id", "operator": "eq", "value": str(USER_B)},
                {"column": "category", "operator": "eq", "value": "groceries"},
            ],
        }
    }

    scoped = enforce_trusted_user_scope("visualize_allowed_data", model_arguments, USER_A)

    assert scoped is not None
    assert scoped["request"]["scope"] == {"user_id": str(USER_A)}
    assert scoped["request"]["filters"] == [
        {"column": "category", "operator": "eq", "value": "groceries"}
    ]
    assert model_arguments["request"]["scope"] == {"user_id": str(USER_B)}


def test_model_facing_schemas_hide_trusted_scope_fields() -> None:
    tool = MCPToolDefinition(
        name="visualize_allowed_data",
        description="Visualize rows.",
        input_schema={
            "type": "object",
            "properties": {
                "request": {
                    "type": "object",
                    "properties": {
                        "scope": {
                            "type": "object",
                            "properties": {"user_id": {"type": "string"}},
                        },
                        "visualization": {"type": "string"},
                    },
                    "required": ["scope", "visualization"],
                }
            },
        },
    )

    model_schema = model_tool_schema(tool)
    request_schema = model_schema["properties"]["request"]

    assert "scope" not in request_schema["properties"]
    assert request_schema["required"] == ["visualization"]
    assert "scope" in tool.input_schema["properties"]["request"]["properties"]


@pytest.mark.asyncio
async def test_missing_graph_identity_fails_before_loading_or_executing_tools() -> None:
    loaded = False
    executed = False

    async def load_tools() -> list[MCPToolDefinition]:
        nonlocal loaded
        loaded = True
        return await _discovery_tools()

    async def execute(
        _name: str,
        _arguments: Mapping[str, Any] | None,
        *,
        current_user_id: UUID | None = None,
    ) -> MCPToolExecution:
        del current_user_id
        nonlocal executed
        executed = True
        return _execution([])

    with pytest.raises(TrustedUserScopeError):
        await build_graph(model=FakeModel(), tool_loader=load_tools, tool_executor=execute).ainvoke(
            {"user_query": "Muéstrame mis movimientos"}
        )

    assert loaded is False
    assert executed is False


@pytest.mark.asyncio
async def test_plain_conversation_can_finish_without_chat_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_profile(_current_user_id: UUID) -> UserContext:
        return UserContext(profile=PROFILE.copy(), rows={})

    monkeypatch.setattr(nodes_module, "fetch_user_context", fake_profile)
    result = await build_graph(model=FakeModel(), tool_loader=_discovery_tools).ainvoke(
        {"user_query": "Hola", "current_user_id": USER_A}
    )
    assert result["message"] == "Hola, ¿en qué te ayudo?"
    assert result["final_tool_execution"] is None


@pytest.mark.asyncio
async def test_classified_financial_request_never_invokes_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingModel(ToolAwareModel):
        async def generate(self, **_kwargs: Any) -> ModelTurn:
            raise AssertionError("classified requests must not invoke Gemini")

    async def fake_profile(_current_user_id: UUID) -> UserContext:
        return UserContext(
            profile=PROFILE.copy(),
            rows={
                "accounts": [
                    {
                        "id": "checking",
                        "account_type": "checking",
                        "currency": "MXN",
                        "available_balance": 150,
                    }
                ],
                "cards": [],
                "subscriptions": [],
            },
        )

    monkeypatch.setattr(nodes_module, "fetch_user_context", fake_profile)
    result = await build_graph(model=FailingModel(), tool_loader=_discovery_tools).ainvoke(
        {"user_query": "¿Cuánto dinero tengo?", "current_user_id": USER_A}
    )

    assert result["financial_presentation"].intent == "financial-summary"


@pytest.mark.asyncio
async def test_classified_request_calls_domain_tool_without_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingModel(ToolAwareModel):
        async def generate(self, **_kwargs: Any) -> ModelTurn:
            raise AssertionError("classified requests must not invoke Gemini")

    calls: list[str] = []

    async def fake_profile(_current_user_id: UUID) -> UserContext:
        return UserContext(profile=PROFILE.copy(), rows={"accounts": [], "cards": []})

    async def execute(
        name: str,
        _arguments: Mapping[str, Any] | None,
        *,
        current_user_id: UUID | None = None,
    ) -> MCPToolExecution:
        assert current_user_id == USER_A
        calls.append(name)
        return MCPToolExecution(
            result=CallToolResult(
                content=[TextContent(text="Rows loaded.")],
                structured_content={"ok": True, "transactions": []},
                meta=None,
                data=None,
            ),
            a2ui=None,
        )

    monkeypatch.setattr(nodes_module, "fetch_user_context", fake_profile)
    await build_graph(
        model=FailingModel(), tool_loader=_discovery_tools, tool_executor=execute
    ).ainvoke({"user_query": "Muéstrame mis movimientos", "current_user_id": USER_A})

    assert calls == ["get_transactions"]


def test_finance_route_requires_a_retained_retrieval_observation() -> None:
    state: Any = {
        "user_query": "Muéstrame mis movimientos",
        "financial_request_intent": "transactions",
        "tool_calls": [],
        "tool_observations": [],
        "context_observations": [],
    }

    assert route_after_agent(state) == END


@pytest.mark.asyncio
async def test_failed_proxied_tool_is_recorded_as_attempted_domain_tool() -> None:
    async def fail(
        _name: str,
        _arguments: Mapping[str, Any] | None,
        *,
        current_user_id: UUID | None = None,
    ) -> MCPToolExecution:
        assert current_user_id == USER_A
        raise UserContextError("offline")

    state: Any = {
        "current_user_id": USER_A,
        "available_tools": [tool.as_dict() for tool in await _discovery_tools()],
        "tool_observations": [],
        "tool_loop_count": 0,
    }
    update = await run_pending_tools(
        state,
        [
            {
                "name": "call_tool",
                "arguments": {"name": "get_transactions", "arguments": {"request": {}}},
            }
        ],
        fail,
    )

    assert update["tool_observations"] == [
        {
            "name": "get_transactions",
            "arguments": {"request": {}},
            "is_error": True,
            "data": {},
            "text": "No pude consultar el servicio de datos en este momento.",
        }
    ]
    failed_state = {
        **state,
        **update,
        "user_query": "Muéstrame mis movimientos",
        "financial_request_intent": "transactions",
    }
    assert route_after_agent(failed_state) == "select_presentation"


@pytest.mark.asyncio
async def test_mcp_owned_a2ui_bypasses_finance_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Model(ToolAwareModel):
        calls = 0

        async def generate(self, **_kwargs: Any) -> ModelTurn:
            self.calls += 1
            return ModelTurn(
                message="",
                tool_calls=(
                    {
                        "name": "call_tool",
                        "arguments": {
                            "name": "get_debt_overview",
                            "arguments": {"request": {}},
                        },
                    },
                ),
            )

    async def fake_profile(_current_user_id: UUID) -> UserContext:
        return UserContext(profile=PROFILE.copy(), rows={})

    bundle = A2UIBundle(
        resource_uri="a2ui://mcp/owned",
        messages=[{"version": "v0.9.1", "createSurface": {}}],
    )

    async def execute(
        _name: str,
        _arguments: Mapping[str, Any] | None,
        *,
        current_user_id: UUID | None = None,
    ) -> MCPToolExecution:
        assert current_user_id == USER_A
        return MCPToolExecution(
            result=CallToolResult(
                content=[TextContent(text="MCP surface")],
                structured_content={"ok": True, "debts": []},
                meta={"ui": {"resourceUri": bundle.resource_uri}},
                data=None,
            ),
            a2ui=bundle,
        )

    model = Model()
    monkeypatch.setattr(nodes_module, "fetch_user_context", fake_profile)
    result = await build_graph(
        model=model, tool_loader=_discovery_tools, tool_executor=execute
    ).ainvoke({"user_query": "Ayúdame con esto", "current_user_id": USER_A})

    assert model.calls == 1
    assert result["final_tool_execution"].a2ui == bundle
    assert result["financial_presentation"] is None
    assert result["presentation_intent"] is None


@pytest.mark.asyncio
async def test_turn_initialization_clears_stale_calls_and_presentations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_profile(_current_user_id: UUID) -> UserContext:
        return UserContext(profile=PROFILE.copy(), rows={})

    stale_bundle = A2UIBundle(
        resource_uri="a2ui://stale",
        messages=[{"version": "v0.9.1", "createSurface": {}}],
    )
    stale_execution = MCPToolExecution(result=_execution([]).result, a2ui=stale_bundle)
    stale_finance = build_financial_presentation("transactions", [], PROFILE)
    monkeypatch.setattr(nodes_module, "fetch_user_context", fake_profile)

    result = await build_graph(model=FakeModel(), tool_loader=_discovery_tools).ainvoke(
        {
            "user_query": "Hola",
            "current_user_id": USER_A,
            "requested_intent": "transactions",
            "action_requested": False,
            "tool_calls": [{"name": "stale", "arguments": {}}],
            "tool_observations": [
                {
                    "name": "get_transactions",
                    "arguments": {},
                    "is_error": False,
                    "data": {},
                    "text": "stale",
                }
            ],
            "final_tool_execution": stale_execution,
            "financial_presentation": stale_finance,
            "financial_request_intent": "transactions",
            "presentation_intent": "transactions",
        }
    )

    assert result["message"] == "Hola, ¿en qué te ayudo?"
    assert result["tool_calls"] == []
    assert result["tool_observations"] == []
    assert result["final_tool_execution"] is None
    assert result["financial_presentation"] is None
    assert result["financial_request_intent"] is None
    assert result["presentation_intent"] is None


@pytest.mark.asyncio
async def test_unresolvable_user_never_receives_placeholder_money(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unavailable(_current_user_id: UUID) -> UserContext:
        raise nodes_module.UserContextError("missing")

    monkeypatch.setattr(nodes_module, "fetch_user_context", unavailable)
    result = await build_graph(model=FakeModel(), tool_loader=_discovery_tools).ainvoke(
        {"user_query": "¿Cuánto dinero tengo?", "current_user_id": USER_A}
    )
    assert result["context_available"] is False
    assert result["user_profile"]["available_balance"] is None
    assert result["user_profile"]["owned_balances"] == {}
    assert "1,200" not in result["message"]


class _FakeResponse:
    def __init__(self, *, function_calls: list[Any] | None = None, parsed: Any = None) -> None:
        self.function_calls = function_calls or []
        self.parsed = parsed
        self.usage_metadata = None


class _FakeFunctionCall:
    def __init__(self, name: str, args: Mapping[str, Any]) -> None:
        self.name = name
        self.args = dict(args)


class _RecordingModels:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.configs: list[Any] = []

    async def generate_content(self, *, model: str, contents: str, config: Any) -> _FakeResponse:
        del model, contents
        self.configs.append(config)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _fake_genai(monkeypatch: pytest.MonkeyPatch, outcomes: list[Any]) -> _RecordingModels:
    models = _RecordingModels(outcomes)
    client = type("_Client", (), {"aio": type("_Aio", (), {"models": models})()})
    monkeypatch.setattr(gemini_module, "genai", type("_Genai", (), {"Client": lambda: client}))
    return models


def _declared_tools() -> list[MCPToolDefinition]:
    return [
        MCPToolDefinition(f"tool_{index}", "herramienta", {"type": "object", "properties": {}})
        for index in range(25)
    ]


@pytest.mark.asyncio
async def test_tool_declarations_and_the_response_schema_never_share_one_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini answers 400 when a large tool catalogue rides with a response schema.

    Declaring the financial domain tools alongside `response_schema` lost every
    unclassified turn, so the two concerns must travel in separate requests.
    """
    models = _fake_genai(
        monkeypatch,
        [_FakeResponse(), _FakeResponse(parsed=_Intent(message="listo"))],
    )

    turn = await GeminiToolAwareModel().generate(
        query="¿Qué es el CAT?", profile=PROFILE, tools=_declared_tools(), observations=[]
    )

    assert turn.message == "listo"
    tool_phase, answer_phase = models.configs
    assert tool_phase.tools is not None
    assert tool_phase.response_schema is None
    assert tool_phase.response_mime_type is None
    assert answer_phase.tools is None
    assert answer_phase.response_schema is _Intent


@pytest.mark.asyncio
async def test_a_selected_tool_skips_the_answer_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = _fake_genai(
        monkeypatch,
        [_FakeResponse(function_calls=[_FakeFunctionCall("tool_3", {"limit": 5})])],
    )

    turn = await GeminiToolAwareModel().generate(
        query="¿Cuánto debo?", profile=PROFILE, tools=_declared_tools(), observations=[]
    )

    assert turn.tool_calls == ({"name": "tool_3", "arguments": {"limit": 5}},)
    assert len(models.configs) == 1


@pytest.mark.asyncio
async def test_a_failed_tool_phase_still_answers_the_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken tool catalogue must not cost the user their answer."""
    models = _fake_genai(
        monkeypatch,
        [RuntimeError("tool declarations rejected"), _FakeResponse(parsed=_Intent(message="hola"))],
    )

    turn = await GeminiToolAwareModel().generate(
        query="¿Qué es el CAT?", profile=PROFILE, tools=_declared_tools(), observations=[]
    )

    assert turn.message == "hola"
    assert len(models.configs) == 2


def test_a_failed_call_is_logged_by_bounded_code_and_status() -> None:
    class _ClientError(Exception):
        code = 400
        status = "INVALID_ARGUMENT"

    assert _api_failure_reason(_ClientError()) == "_ClientError/400/INVALID_ARGUMENT"
    assert _api_failure_reason(RuntimeError("secreto")) == "RuntimeError"
