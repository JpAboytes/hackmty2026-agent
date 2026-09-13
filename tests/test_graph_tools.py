"""Focused financial graph and presentation-selection tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

import pytest
from fastmcp.client.client import CallToolResult
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
    resolve_tool_call,
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
    "color_vision_mode": "none",
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


class ScriptedModel(ToolAwareModel):
    """A model that plays a fixed sequence of turns and records what it saw.

    Which capability answers a question is the model's decision now, so the
    tests drive that decision explicitly instead of relying on a phrase table.
    What is still asserted is everything downstream: scoping, provenance, the
    presentation and the surface.
    """

    def __init__(self, *turns: ModelTurn) -> None:
        self._turns = list(turns)
        self.seen: list[dict[str, Any]] = []

    async def generate(
        self,
        *,
        query: str,
        profile: UserProfile,
        tools: Sequence[MCPToolDefinition],
        observations: Sequence[Mapping[str, Any]],
    ) -> ModelTurn:
        self.seen.append(
            {
                "query": query,
                "profile": dict(profile),
                "tools": sorted(tool.name for tool in tools),
                "observations": [dict(observation) for observation in observations],
            }
        )
        if self._turns:
            return self._turns.pop(0)
        return ModelTurn(message="No tengo nada más que consultar.")


def _searches(query: str) -> ModelTurn:
    return ModelTurn(
        message="", tool_calls=({"name": "search_tools", "arguments": {"query": query}},)
    )


def _calls(*targets: tuple[str, dict[str, Any]]) -> ModelTurn:
    """One turn that calls 1..N discovered tools through the `call_tool` proxy."""
    return ModelTurn(
        message="",
        tool_calls=tuple(
            {"name": "call_tool", "arguments": {"name": name, "arguments": arguments}}
            for name, arguments in targets
        ),
    )


def _presents(intent: str, message: str = "Listo.") -> ModelTurn:
    return ModelTurn(message=message, presentation_intent=intent)  # type: ignore[arg-type]


def _candidates(*names: str) -> MCPToolExecution:
    """A ranked `search_tools` result, as the server would return it."""
    return MCPToolExecution(
        result=CallToolResult(
            content=[TextContent(text="Tools found.")],
            structured_content={
                "result": [
                    {
                        "name": name,
                        "description": f"{name} description",
                        "inputSchema": {"type": "object", "properties": {"request": {}}},
                    }
                    for name in names
                ]
            },
            meta=None,
            data=None,
        ),
        a2ui=None,
    )


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


def _execution(rows: list[dict[str, Any]], payload_key: str = "rows") -> MCPToolExecution:
    return MCPToolExecution(
        result=CallToolResult(
            content=[TextContent(text="Rows loaded.")],
            structured_content={"ok": True, payload_key: rows},
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
    model: ToolAwareModel | None = None,
    candidates: tuple[str, ...] = (),
    payload_key: str = "rows",
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    """Run the graph, recording every call by the *domain* tool it resolved to.

    The model addresses a discovered tool through the `call_tool` envelope, so
    recording the envelope would say nothing about which capability ran.
    """
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
        target, target_arguments = resolve_tool_call(name, resolved)
        calls.append((target, dict(target_arguments or {})))
        if target == "search_tools":
            return _candidates(*candidates)
        return _execution(rows, payload_key)

    monkeypatch.setattr(nodes_module, "fetch_user_context", fake_profile)
    result = await build_graph(
        model=model or FakeModel(), tool_loader=_discovery_tools, tool_executor=execute
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
    model = ScriptedModel(
        _searches("saldo disponible"),
        _calls(("get_accounts", {"request": {}})),
        _presents("financial-summary"),
    )
    result, calls = await _run_graph(
        monkeypatch,
        "¿Cuánto dinero tengo?",
        rows,
        model=model,
        payload_key="accounts",
        candidates=("get_accounts", "get_financial_overview", "analyze_spending"),
    )

    assert [name for name, _ in calls] == ["search_tools", "get_accounts"]
    assert calls[1][1]["request"]["scope"] == {"user_id": str(USER_A)}
    # The candidate list held three tools and the model picked one of them:
    # the ranking is an input to its decision, not the decision itself.
    assert len(model.seen[1]["observations"][0]["data"]["result"]) == 3
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
    assert "chat_message" not in {name for name, _ in calls}


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
    # The model is handed the prefetched context and decides no read is needed.
    # The old deterministic planner could not reach this outcome: it read the
    # overview again because its plan came from the intent, not from what the
    # turn already held.
    result, calls = await _run_graph(
        monkeypatch,
        "¿Cuánto dinero tengo?",
        [],
        context_rows={"accounts": accounts, "cards": []},
        model=ScriptedModel(_presents("financial-summary")),
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
        model=ScriptedModel(
            _calls(("get_transactions", {"request": {"period": "current_month"}})),
            _presents("transactions"),
        ),
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
async def test_the_tool_and_intent_the_model_chose_flow_through_untouched(
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
    result, calls = await _run_graph(
        monkeypatch,
        "¿Qué días gasto más?",
        rows,
        model=ScriptedModel(
            _calls(("analyze_spending", {"request": {"period": "current_month"}})),
            _presents("spending-analysis"),
        ),
    )
    assert [name for name, _ in calls] == ["analyze_spending"]
    # The arguments the model chose reach MCP unaltered except for scope.
    assert calls[0][1]["request"]["period"] == "current_month"
    assert calls[0][1]["request"]["scope"] == {"user_id": str(USER_A)}
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
async def test_a_financial_request_is_answered_from_prefetched_context_when_it_suffices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No classifier decides this; the model does, and it may need no tool at all.

    `get_user_context` prefetches the account and card rows, so a balance
    question can be answered without a domain read. The old deterministic
    planner could not reach this outcome - it planned from the intent rather
    than from what the turn already held.
    """

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
    model = ScriptedModel(_presents("financial-summary"))
    result = await build_graph(model=model, tool_loader=_discovery_tools).ainvoke(
        {"user_query": "¿Cuánto dinero tengo?", "current_user_id": USER_A}
    )

    assert result["financial_presentation"].intent == "financial-summary"
    assert result["financial_presentation"].data["owned_balance"] == 150
    # The model was consulted exactly once and chose to read nothing further.
    assert len(model.seen) == 1


@pytest.mark.asyncio
async def test_the_domain_tool_the_model_chose_runs_scoped_without_a_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery is available, not mandatory: the model may address a tool directly."""
    calls: list[str] = []

    async def fake_profile(_current_user_id: UUID) -> UserContext:
        return UserContext(profile=PROFILE.copy(), rows={"accounts": [], "cards": []})

    async def execute(
        name: str,
        arguments: Mapping[str, Any] | None,
        *,
        current_user_id: UUID | None = None,
    ) -> MCPToolExecution:
        assert current_user_id == USER_A
        target, _ = resolve_tool_call(name, enforce_trusted_user_scope(name, arguments, current_user_id) or {})
        calls.append(target)
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
    model = ScriptedModel(
        _calls(("get_transactions", {"request": {"period": "current_month"}})),
        _presents("transactions"),
    )
    await build_graph(
        model=model, tool_loader=_discovery_tools, tool_executor=execute
    ).ainvoke({"user_query": "Muéstrame mis movimientos", "current_user_id": USER_A})

    assert calls == ["get_transactions"]
    assert "search_tools" not in calls


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
        "presentation_intent": "transactions",
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
            "presentation_intent": "transactions",
        }
    )

    assert result["message"] == "Hola, ¿en qué te ayudo?"
    assert result["tool_calls"] == []
    assert result["tool_observations"] == []
    assert result["final_tool_execution"] is None
    assert result["financial_presentation"] is None
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
        self.prompts: list[str] = []

    async def generate_content(self, *, model: str, contents: str, config: Any) -> _FakeResponse:
        del model
        self.prompts.append(contents)
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


@pytest.mark.asyncio
async def test_the_prompt_never_carries_the_users_balances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model renders no figures, so it is shown none.

    Every figure the user sees comes from the trusted builder reading MCP
    observations. Putting balances in the prompt bought no behaviour and put
    the user's money into every request of every turn.
    """
    models = _fake_genai(
        monkeypatch,
        [_FakeResponse(), _FakeResponse(parsed=_Intent(message="listo"))],
    )
    profile: UserProfile = {
        **PROFILE,
        "available_balance": 98765.43,
        "owned_balances": {"MXN": 98765.43},
        "overdraft_risk": 0.87,
        "recurring_expenses": 4321.0,
    }

    await GeminiToolAwareModel().generate(
        query="¿Cuánto dinero tengo?", profile=profile, tools=_declared_tools(), observations=[]
    )

    for prompt in models.prompts:
        for secret in ("98765", "4321", "0.87", "owned_balances", "available_balance"):
            assert secret not in prompt
        # The one thing a model can act on is how plainly to speak.
        assert "literacy_level" in prompt


def test_the_model_profile_projection_keeps_only_literacy_level() -> None:
    from fluidbank_orchestrator.agent.gemini import model_profile

    assert model_profile(PROFILE) == {"literacy_level": "medium"}


@pytest.mark.asyncio
async def test_the_model_may_select_more_than_two_tools_in_one_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-tool selection was silently truncated to two calls."""
    from fluidbank_orchestrator.agent.gemini import MAX_CALLS_PER_TURN

    requested = [_FakeFunctionCall(f"tool_{index}", {"limit": index}) for index in range(5)]
    models = _fake_genai(monkeypatch, [_FakeResponse(function_calls=requested)])

    turn = await GeminiToolAwareModel().generate(
        query="Necesito varias cosas",
        profile=PROFILE,
        tools=_declared_tools(),
        observations=[],
    )

    assert MAX_CALLS_PER_TURN >= 5
    assert [call["name"] for call in turn.tool_calls] == [
        "tool_0",
        "tool_1",
        "tool_2",
        "tool_3",
        "tool_4",
    ]
    assert len(models.configs) == 1


@pytest.mark.asyncio
async def test_a_runaway_fan_out_is_still_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    from fluidbank_orchestrator.agent.gemini import MAX_CALLS_PER_TURN

    requested = [
        _FakeFunctionCall(f"tool_{index}", {}) for index in range(MAX_CALLS_PER_TURN + 6)
    ]
    _fake_genai(monkeypatch, [_FakeResponse(function_calls=requested)])

    turn = await GeminiToolAwareModel().generate(
        query="Llama todo", profile=PROFILE, tools=_declared_tools(), observations=[]
    )

    assert len(turn.tool_calls) == MAX_CALLS_PER_TURN


def test_the_model_may_select_a_form_only_from_the_declared_vocabulary() -> None:
    """`_Intent.action_form` is an enum, so an invented form cannot be parsed."""
    import pydantic

    from fluidbank_orchestrator.a2ui_actions.forms import ACTION_FORM_NAMES

    for name in ACTION_FORM_NAMES:
        assert _Intent(message="", action_form=name).action_form == name
    with pytest.raises(pydantic.ValidationError):
        _Intent(message="", action_form="budget.delete")
