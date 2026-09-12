"""Offline API authentication and action-routing tests."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import HTTPException
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent
from pydantic import ValidationError

from fluidbank_orchestrator import api
from fluidbank_orchestrator.api import ChatRequest
from fluidbank_orchestrator.mcp_client import MCPToolExecution

USER_A = "68dc4d66-07b8-5893-95f1-07f06989a552"
USER_B = "c1a3797d-b335-5a9d-98a1-402311f82c7a"
AUTH = "Bearer test-token"
ACTION = {
    "name": "refresh_database_overview",
    "surfaceId": "database-overview",
    "sourceComponentId": "refresh_button",
    "timestamp": "2026-09-12T12:00:00.000Z",
    "context": {"limit": 50},
}


def test_chat_request_requires_one_input_and_a_user_id() -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"query": "Hola"})
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"query": "Hola", "action": ACTION, "user_id": USER_A})
    assert ChatRequest(query="Hola", user_id=USER_A).query == "Hola"
    assert ChatRequest(action=ACTION, user_id=USER_A).action is not None


async def _authenticate_as_a(_authorization: str | None) -> str:
    return USER_A


def _execution(
    message: str = "Updated.", structured_content: dict[str, Any] | None = None
) -> MCPToolExecution:
    return MCPToolExecution(
        result=CallToolResult(
            content=[TextContent(text=message)],
            structured_content=structured_content or {"ok": True},
            meta=None,
            data=None,
        ),
        a2ui=None,
    )


@pytest.mark.asyncio
async def test_token_subject_body_uuid_mismatch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def authenticate_as_b(_authorization: str | None) -> str:
        return USER_B

    monkeypatch.setattr(api, "verify_supabase_access_token", authenticate_as_b)
    with pytest.raises(HTTPException) as caught:
        await api.chat(ChatRequest(query="Hola", user_id=USER_A), AUTH)
    assert caught.value.status_code == 403


@pytest.mark.asyncio
async def test_structured_financial_action_preserves_authenticated_user_in_graph_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invocations: list[tuple[dict[str, Any], dict[str, Any]]] = []
    tool_calls: list[tuple[str, dict[str, Any] | None]] = []

    class FakeGraph:
        async def ainvoke(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            invocations.append((state, config))
            return {"message": "Movimientos listos.", "user_profile": {}}

    action = {
        "name": "request_financial_view",
        "surfaceId": "financial-view",
        "sourceComponentId": "request_financial_view_button",
        "timestamp": "2026-09-12T12:00:00.000Z",
        "context": {"intent": "transactions"},
    }

    async def execute(name: str, arguments: dict[str, Any] | None = None) -> MCPToolExecution:
        tool_calls.append((name, arguments))
        return _execution(
            structured_content={
                "ok": True,
                "action": action,
                "request": {"intent": "transactions"},
                "trustedScope": {"user_id": USER_A},
            }
        )

    monkeypatch.setattr(api, "verify_supabase_access_token", _authenticate_as_a)
    monkeypatch.setattr(api, "execute_remote_tool", execute)
    monkeypatch.setattr(api, "graph", FakeGraph())
    response = await api.chat(ChatRequest(action=action, user_id=USER_A), AUTH)

    state, config = invocations[0]
    assert state["current_user_id"] == USER_A
    assert state["requested_intent"] == "transactions"
    assert state["action_requested"] is True
    assert config == {"configurable": {"thread_id": f"user:{USER_A}"}}
    assert response.message == "Movimientos listos."
    assert tool_calls == [
        ("a2ui_action", {**action, "trustedScope": {"user_id": USER_A}})
    ]


@pytest.mark.asyncio
async def test_rejected_financial_action_never_reaches_the_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoked = False

    class FakeGraph:
        async def ainvoke(self, _state: dict[str, Any], _config: dict[str, Any]) -> dict[str, Any]:
            nonlocal invoked
            invoked = True
            return {"message": "Unexpected", "user_profile": {}}

    async def reject(_name: str, _arguments: dict[str, Any] | None = None) -> MCPToolExecution:
        return _execution(
            structured_content={
                "ok": False,
                "error": {"code": "component_mismatch", "message": "Invalid action."},
            }
        )

    action = {
        "name": "request_financial_view",
        "surfaceId": "financial-view",
        "sourceComponentId": "request_financial_view_button",
        "timestamp": "2026-09-12T12:00:00.000Z",
        "context": {"intent": "transactions"},
    }
    monkeypatch.setattr(api, "verify_supabase_access_token", _authenticate_as_a)
    monkeypatch.setattr(api, "execute_remote_tool", reject)
    monkeypatch.setattr(api, "graph", FakeGraph())

    response = await api.chat(ChatRequest(action=action, user_id=USER_A), AUTH)

    assert invoked is False
    assert response.message == "La acción de interfaz no es válida."


def _legacy_action(action: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Actualiza esta consulta financiera de solo lectura o simulación usando la acción "
            "de interfaz adjunta.",
            f"Acción A2UI: {json.dumps(action, ensure_ascii=False, separators=(',', ':'))}",
        ]
    )


@pytest.mark.asyncio
async def test_legacy_nonfinancial_action_remains_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any] | None]] = []

    async def execute(name: str, arguments: dict[str, Any] | None = None) -> MCPToolExecution:
        calls.append((name, arguments))
        return _execution()

    monkeypatch.setattr(api, "verify_supabase_access_token", _authenticate_as_a)
    monkeypatch.setattr(api, "execute_remote_tool", execute)
    response = await api.chat(ChatRequest(query=_legacy_action(ACTION), user_id=USER_A), AUTH)
    assert calls == [
        ("a2ui_action", {**ACTION, "trustedScope": {"user_id": USER_A}})
    ]
    assert response.data == {"ok": True}


@pytest.mark.asyncio
async def test_database_overview_is_authenticated_before_domain_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any] | None]] = []

    async def execute(name: str, arguments: dict[str, Any] | None = None) -> MCPToolExecution:
        calls.append((name, arguments))
        return _execution("Database overview loaded.")

    monkeypatch.setattr(api, "verify_supabase_access_token", _authenticate_as_a)
    monkeypatch.setattr(api, "execute_remote_tool", execute)
    response = await api.chat(
        ChatRequest(query="Muéstrame los objetos disponibles de la base de datos", user_id=USER_A),
        AUTH,
    )
    assert calls == [("database_overview", {"limit": 50})]
    assert response.message == "Database overview loaded."
