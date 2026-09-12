"""Offline routing tests for overview requests, actions, and ordinary chat."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent

from fluidbank_orchestrator import api
from fluidbank_orchestrator.api import ChatRequest
from fluidbank_orchestrator.mcp_client import MCPToolExecution

ACTION = {
    "name": "refresh_database_overview",
    "surfaceId": "database-overview",
    "sourceComponentId": "refresh_button",
    "timestamp": "2026-09-12T12:00:00.000Z",
    "context": {"limit": 50},
}


def _legacy_action(action: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Actualiza esta consulta financiera de solo lectura o simulación usando la acción "
            "de interfaz adjunta.",
            f"Acción A2UI: {json.dumps(action, ensure_ascii=False, separators=(',', ':'))}",
        ]
    )


def _execution(message: str = "Updated.") -> MCPToolExecution:
    return MCPToolExecution(
        result=CallToolResult(
            content=[TextContent(text=message)],
            structured_content={"ok": True},
            meta=None,
            data=None,
        ),
        a2ui=None,
    )


@pytest.mark.asyncio
async def test_valid_legacy_action_calls_a2ui_action_once_with_all_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any] | None]] = []

    async def fake_execute(name: str, arguments: dict[str, Any] | None = None) -> MCPToolExecution:
        calls.append((name, arguments))
        return _execution()

    monkeypatch.setattr(api, "execute_remote_tool", fake_execute)
    response = await api.chat(ChatRequest(query=_legacy_action(ACTION), persona="ana"))

    assert calls == [("a2ui_action", ACTION)]
    assert response.message == "Updated."
    assert response.data == {"ok": True}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        _legacy_action({**ACTION, "extra": True}),
        _legacy_action({key: value for key, value in ACTION.items() if key != "timestamp"}),
        _legacy_action({**ACTION, "timestamp": "not-a-time"}),
        _legacy_action({**ACTION, "context": {"value": "x" * 17_000}}),
        "Actualiza esta consulta financiera de solo lectura o simulación usando la acción de "
        "interfaz adjunta.\nAcción A2UI: not-json",
    ],
)
async def test_invalid_or_oversized_actions_never_call_mcp(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    calls = 0

    async def fake_execute(name: str, arguments: dict[str, Any] | None = None) -> MCPToolExecution:
        nonlocal calls
        calls += 1
        return _execution()

    monkeypatch.setattr(api, "execute_remote_tool", fake_execute)
    response = await api.chat(ChatRequest(query=query, persona="ana"))
    assert calls == 0
    assert response.a2ui is None
    assert response.message == "La acción de interfaz no es válida."


@pytest.mark.asyncio
async def test_database_overview_intent_calls_existing_domain_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any] | None]] = []

    async def fake_execute(name: str, arguments: dict[str, Any] | None = None) -> MCPToolExecution:
        calls.append((name, arguments))
        return _execution("Database overview loaded.")

    monkeypatch.setattr(api, "execute_remote_tool", fake_execute)
    response = await api.chat(
        ChatRequest(query="Muéstrame los objetos disponibles de la base de datos", persona="ana")
    )
    assert calls == [("database_overview", {"limit": 50})]
    assert response.message == "Database overview loaded."


@pytest.mark.asyncio
async def test_ordinary_non_a2ui_chat_behavior_remains_graph_backed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGraph:
        async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
            assert state["user_query"] == "¿Tengo dinero para el fin de semana?"
            return {
                "message": "Respuesta habitual.",
                "user_profile": {"available_balance": 1200.0},
                "months": 6,
            }

    async def forbidden_execute(
        name: str, arguments: dict[str, Any] | None = None
    ) -> MCPToolExecution:
        raise AssertionError("ordinary chat must not call a presentation tool")

    monkeypatch.setattr(api, "graph", FakeGraph())
    monkeypatch.setattr(api, "execute_remote_tool", forbidden_execute)
    response = await api.chat(
        ChatRequest(query="¿Tengo dinero para el fin de semana?", persona="ana")
    )
    assert response.model_dump() == {
        "message": "Respuesta habitual.",
        "data": {"available_balance": 1200.0, "months": 6},
        "a2ui": None,
    }
