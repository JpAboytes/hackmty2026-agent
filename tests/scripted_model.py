"""A scripted model and graph runners shared by the agent-behaviour tests.

Which capability answers a question is the model's decision, so these tests
state that decision explicitly and assert everything downstream of it:
discovery, selection, scoping, provenance, the presentation and the reported
lifecycle phases. There is no phrase table left to assert against.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

import pytest
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent

import fluidbank_orchestrator.agent.nodes as nodes_module
from fluidbank_orchestrator.agent.model import ModelTurn, ToolAwareModel
from fluidbank_orchestrator.graph import build_graph
from fluidbank_orchestrator.mcp_client import (
    MCPToolDefinition,
    MCPToolExecution,
    UserContext,
    enforce_trusted_user_scope,
    resolve_tool_call,
)
from fluidbank_orchestrator.state import UserProfile

USER_A = UUID("68dc4d66-07b8-5893-95f1-07f06989a552")

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


class ScriptedModel(ToolAwareModel):
    """Plays a fixed sequence of turns and records what it was shown."""

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


def searches(query: str) -> ModelTurn:
    """One turn that asks for candidate tools."""
    return ModelTurn(
        message="", tool_calls=({"name": "search_tools", "arguments": {"query": query}},)
    )


def calls(*targets: tuple[str, dict[str, Any]]) -> ModelTurn:
    """One turn that calls 1..N discovered tools through the `call_tool` proxy."""
    return ModelTurn(
        message="",
        tool_calls=tuple(
            {"name": "call_tool", "arguments": {"name": name, "arguments": arguments}}
            for name, arguments in targets
        ),
    )


def presents(intent: str, message: str = "Listo.") -> ModelTurn:
    return ModelTurn(message=message, presentation_intent=intent)  # type: ignore[arg-type]


def prepares(form: str, message: str = "Revisa los datos.") -> ModelTurn:
    return ModelTurn(message=message, action_form=form)


def answers(message: str) -> ModelTurn:
    return ModelTurn(message=message)


async def discovery_tools() -> list[MCPToolDefinition]:
    """What the server advertises under progressive discovery: the search pair."""
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


def rows_result(rows: list[dict[str, Any]], payload_key: str = "rows") -> MCPToolExecution:
    return MCPToolExecution(
        result=CallToolResult(
            content=[TextContent(text="Rows loaded.")],
            structured_content={"ok": True, payload_key: rows},
            meta=None,
            data=None,
        ),
        a2ui=None,
    )


def candidates_result(*names: str) -> MCPToolExecution:
    """A ranked `search_tools` result, in the shape the server returns."""
    return MCPToolExecution(
        result=CallToolResult(
            content=[TextContent(text="Tools found.")],
            structured_content={
                "result": [
                    {
                        "name": name,
                        "description": f"{name} description",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"request": {"type": "object"}},
                        },
                    }
                    for name in names
                ]
            },
            meta=None,
            data=None,
        ),
        a2ui=None,
    )


def error_result(message: str) -> MCPToolExecution:
    """A tool failure the model must be able to read and react to."""
    return MCPToolExecution(
        result=CallToolResult(
            content=[TextContent(text=message)],
            structured_content={
                "ok": False,
                "error": {"code": "database_timeout", "message": message, "retryable": True},
            },
            meta=None,
            data=None,
            is_error=True,
        ),
        a2ui=None,
    )


class Recorder:
    """A fake MCP executor that records the domain tool each call resolved to."""

    def __init__(
        self,
        *,
        results: Mapping[str, MCPToolExecution] | None = None,
        default: MCPToolExecution | None = None,
    ) -> None:
        self.results = dict(results or {})
        self.default = default or rows_result([])
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        current_user_id: UUID | None = None,
    ) -> MCPToolExecution:
        scoped = enforce_trusted_user_scope(name, arguments, current_user_id) or {}
        target, target_arguments = resolve_tool_call(name, scoped)
        self.calls.append((target, dict(target_arguments or {})))
        return self.results.get(target, self.default)

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


def graph_for(
    monkeypatch: pytest.MonkeyPatch,
    model: ToolAwareModel,
    executor: Recorder,
    context_rows: dict[str, list[dict[str, Any]]] | None = None,
) -> Any:
    async def fake_profile(_current_user_id: UUID) -> UserContext:
        return UserContext(profile=PROFILE.copy(), rows=dict(context_rows or {}))

    monkeypatch.setattr(nodes_module, "fetch_user_context", fake_profile)
    return build_graph(model=model, tool_loader=discovery_tools, tool_executor=executor)


async def run_turn(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    model: ToolAwareModel,
    executor: Recorder,
    context_rows: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    graph = graph_for(monkeypatch, model, executor, context_rows)
    result: dict[str, Any] = await graph.ainvoke(
        {"user_query": query, "current_user_id": USER_A}
    )
    return result


async def statuses_for(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    model: ToolAwareModel,
    executor: Recorder,
    context_rows: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every custom stream payload the turn emitted, plus its final state."""
    graph = graph_for(monkeypatch, model, executor, context_rows)
    payloads: list[dict[str, Any]] = []
    final: dict[str, Any] = {}
    async for mode, payload in graph.astream(
        {"user_query": query, "current_user_id": USER_A},
        stream_mode=["custom", "values"],
    ):
        if mode == "custom":
            payloads.append(dict(payload))
        elif mode == "values" and isinstance(payload, dict):
            final = payload
    return payloads, final
