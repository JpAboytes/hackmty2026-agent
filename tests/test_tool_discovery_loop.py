"""The model-driven half of progressive discovery: search, then call."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

import pytest
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent

import fluidbank_orchestrator.graph as graph_module
from fluidbank_orchestrator.graph import ModelTurn, ToolAwareModel, build_graph
from fluidbank_orchestrator.mcp_client import (
    CALL_TOOL_NAME,
    SEARCH_TOOL_NAME,
    MCPToolDefinition,
    MCPToolExecution,
    enforce_trusted_user_scope,
    resolve_tool_call,
)
from fluidbank_orchestrator.state import UserProfile

USER_A = UUID("68dc4d66-07b8-5893-95f1-07f06989a552")

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

DISCOVERED_DEBT_TOOL = {
    "name": "get_debt_overview",
    "description": "Deudas y tarjetas de credito / Debts and credit-card balances owed.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "request": {
                "type": "object",
                "properties": {
                    "scope": {"type": "object"},
                    "status": {"type": "string"},
                },
                "required": ["scope"],
            }
        },
    },
}


async def _discovery_tools() -> list[MCPToolDefinition]:
    return [
        MCPToolDefinition(
            name=SEARCH_TOOL_NAME,
            description="Search for tools using natural language.",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        ),
        MCPToolDefinition(
            name=CALL_TOOL_NAME,
            description="Call a tool by name with the given arguments.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {
                        "anyOf": [
                            {"type": "object", "additionalProperties": True},
                            {"type": "null"},
                        ]
                    },
                },
            },
        ),
    ]


class ScriptedModel(ToolAwareModel):
    """Replays a fixed sequence of turns and records what it was offered."""

    def __init__(self, turns: list[ModelTurn]) -> None:
        self._turns = list(turns)
        self.offered: list[tuple[str, ...]] = []
        self.observed: list[tuple[str, ...]] = []

    async def generate(
        self,
        *,
        query: str,
        profile: UserProfile,
        tools: Sequence[MCPToolDefinition],
        observations: Sequence[Mapping[str, Any]],
    ) -> ModelTurn:
        del query, profile
        self.offered.append(tuple(tool.name for tool in tools))
        self.observed.append(tuple(str(item.get("name")) for item in observations))
        return self._turns.pop(0) if self._turns else ModelTurn(message="Listo.")


def _result(structured: dict[str, Any], text: str = "ok") -> MCPToolExecution:
    return MCPToolExecution(
        result=CallToolResult(
            content=[TextContent(text=text)],
            structured_content=structured,
            meta=None,
            data=None,
        ),
        a2ui=None,
    )


async def _run(
    monkeypatch: pytest.MonkeyPatch, query: str, model: ScriptedModel
) -> tuple[dict[str, Any], list[tuple[str, Mapping[str, Any] | None]]]:
    calls: list[tuple[str, Mapping[str, Any] | None]] = []

    async def fake_profile(current_user_id: UUID) -> Any:
        del current_user_id

        class _Context:
            profile = PROFILE
            rows: dict[str, list[dict[str, object]]] = {}

        return _Context()

    async def execute(
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        current_user_id: UUID | None = None,
    ) -> MCPToolExecution:
        # Mirror the real executor: scope is enforced before anything is sent.
        calls.append((name, enforce_trusted_user_scope(name, arguments, current_user_id)))
        if name == SEARCH_TOOL_NAME:
            return _result({"result": [DISCOVERED_DEBT_TOOL]}, "1 tool found.")
        return _result({"ok": True, "debts": [{"id": "d1", "balance": 100}]}, "Datos.")

    monkeypatch.setattr(graph_module, "fetch_user_context", fake_profile)
    result = await build_graph(
        model=model, tool_loader=_discovery_tools, tool_executor=execute
    ).ainvoke({"user_query": query, "current_user_id": USER_A})
    return result, calls


async def test_the_model_is_offered_only_the_discovery_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 15 financial schemas must not reach the prompt."""
    model = ScriptedModel([ModelTurn(message="Hola.")])

    await _run(monkeypatch, "cuéntame un chiste", model)

    assert model.offered == [(SEARCH_TOOL_NAME, CALL_TOOL_NAME)]


async def test_search_then_call_reaches_the_domain_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ScriptedModel(
        [
            ModelTurn(
                message="",
                tool_calls=({"name": SEARCH_TOOL_NAME, "arguments": {"query": "deudas"}},),
            ),
            ModelTurn(
                message="",
                tool_calls=(
                    {
                        "name": CALL_TOOL_NAME,
                        "arguments": {
                            "name": "get_debt_overview",
                            "arguments": {"request": {"status": "active"}},
                        },
                    },
                ),
            ),
            ModelTurn(message="Tienes una deuda activa."),
        ]
    )

    result, calls = await _run(monkeypatch, "cuéntame un chiste", model)

    assert [name for name, _ in calls] == [SEARCH_TOOL_NAME, CALL_TOOL_NAME]
    # The proxy envelope is what travels; the trusted scope is injected into the
    # tool inside it, never into the envelope.
    envelope = calls[1][1]
    assert envelope is not None
    assert envelope["name"] == "get_debt_overview"
    assert envelope["arguments"]["request"]["scope"] == {"user_id": str(USER_A)}
    assert envelope["arguments"]["request"]["status"] == "active"
    # Observations are labelled by the capability, not by the envelope.
    assert model.observed[-1] == (SEARCH_TOOL_NAME, "get_debt_overview")
    assert result["message"] == "Tienes una deuda activa."


async def test_discovered_schemas_reach_the_model_without_identity_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ScriptedModel(
        [
            ModelTurn(
                message="",
                tool_calls=({"name": SEARCH_TOOL_NAME, "arguments": {"query": "deudas"}},),
            ),
            ModelTurn(message="Listo."),
        ]
    )

    result, _calls = await _run(monkeypatch, "cuéntame un chiste", model)

    search_observation = next(
        item for item in result["tool_observations"] if item["name"] == SEARCH_TOOL_NAME
    )
    schema = search_observation["data"]["result"][0]["inputSchema"]
    assert "scope" not in schema["properties"]["request"]["properties"]
    assert schema["properties"]["request"]["required"] == []


async def test_the_same_intent_is_not_searched_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated search burns a turn and returns the definitions already held."""
    repeat = ModelTurn(
        message="", tool_calls=({"name": SEARCH_TOOL_NAME, "arguments": {"query": "Deudas"}},)
    )
    model = ScriptedModel(
        [
            ModelTurn(
                message="",
                tool_calls=({"name": SEARCH_TOOL_NAME, "arguments": {"query": "deudas"}},),
            ),
            repeat,
            ModelTurn(message="Listo."),
        ]
    )

    _result, calls = await _run(monkeypatch, "cuéntame un chiste", model)

    assert [name for name, _ in calls] == [SEARCH_TOOL_NAME]


# Envelope shapes gemini-3.6-flash actually produced for one discovered tool,
# captured live. Roughly three calls in ten are malformed, in these ways; each
# has exactly one valid reading, so the client normalizes rather than paying a
# server round trip and a retry turn for it.
MALFORMED_ENVELOPES = {
    "well_formed": {
        "name": "get_debt_overview",
        "arguments": {"request": {"status": "active"}},
    },
    "double_wrapped": {
        "name": "call_tool",
        "arguments": {"name": "get_debt_overview", "arguments": {"request": {"status": "active"}}},
    },
    "wrapped_with_sibling_request": {
        "name": "call_tool",
        "arguments": {"name": "get_debt_overview", "request": {"status": "active"}},
    },
    "request_wrapper_dropped": {
        "name": "get_debt_overview",
        "arguments": {"status": "active"},
    },
    "name_one_level_down": {
        "arguments": {"name": "get_debt_overview", "request": {"status": "active"}},
    },
    "request_wrapper_applied_twice": {
        "name": "call_tool",
        "arguments": {"name": "get_debt_overview", "request": {"request": {"status": "active"}}},
    },
}


@pytest.mark.parametrize("shape", sorted(MALFORMED_ENVELOPES))
def test_every_observed_envelope_shape_normalizes_to_one_scoped_call(shape: str) -> None:
    envelope = MALFORMED_ENVELOPES[shape]

    target, _ = resolve_tool_call(CALL_TOOL_NAME, envelope)
    scoped = enforce_trusted_user_scope(CALL_TOOL_NAME, envelope, USER_A)

    assert target == "get_debt_overview"
    assert scoped is not None
    assert scoped["name"] == "get_debt_overview"
    assert scoped["arguments"]["request"] == {
        "status": "active",
        "scope": {"user_id": str(USER_A)},
    }


def test_normalizing_the_envelope_does_not_open_a_path_to_search_tools() -> None:
    """Unwrapping must not turn the proxy into a way to re-enter discovery."""
    envelope = {"name": SEARCH_TOOL_NAME, "arguments": {"query": "deudas"}}

    target, _ = resolve_tool_call(CALL_TOOL_NAME, envelope)

    assert target == CALL_TOOL_NAME


def test_an_envelope_nested_beyond_reason_is_not_followed_forever() -> None:
    envelope: dict[str, Any] = {"name": "get_debt_overview", "arguments": {"request": {}}}
    for _ in range(12):
        envelope = {"name": CALL_TOOL_NAME, "arguments": envelope}

    target, _ = resolve_tool_call(CALL_TOOL_NAME, envelope)

    assert target == CALL_TOOL_NAME
