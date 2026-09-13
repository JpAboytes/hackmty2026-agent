"""Offline API authentication and action-routing tests."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import pytest
from fastapi import HTTPException
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent
from pydantic import ValidationError

from fluidbank_orchestrator import api
from fluidbank_orchestrator.api import ChatRequest
from fluidbank_orchestrator.mcp_client import MCPToolExecution

USER_A = UUID("68dc4d66-07b8-5893-95f1-07f06989a552")
USER_B = UUID("c1a3797d-b335-5a9d-98a1-402311f82c7a")
AUTH = "Bearer test-token"
ACTION = {
    "name": "refresh_database_overview",
    "surfaceId": "database-overview",
    "sourceComponentId": "refresh_button",
    "timestamp": "2026-09-12T12:00:00.000Z",
    "context": {"limit": 50},
}


def test_chat_request_requires_exactly_one_input() -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"query": "Hola", "action": ACTION})
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"query": "Hola", "unknown": "x"})
    assert ChatRequest(query="Hola").query == "Hola"
    assert ChatRequest(action=ACTION).action is not None


def test_chat_request_tolerates_but_never_trusts_a_client_user_id() -> None:
    """The deployed client still sends it; forbidding it 422s every request."""
    request = ChatRequest.model_validate({"query": "Hola", "user_id": str(USER_A)})
    assert request.user_id == USER_A


@pytest.mark.asyncio
async def test_a_client_user_id_that_contradicts_the_token_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "verify_supabase_access_token", _authenticate_as_a)
    with pytest.raises(HTTPException) as refused:
        await api.chat(ChatRequest(query="Hola", user_id=USER_B), AUTH)
    assert refused.value.status_code == 403


async def _authenticate_as_a(_authorization: str | None) -> UUID:
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
@pytest.mark.parametrize("authorization", [None, "Basic token", "Bearer "])
async def test_missing_or_malformed_authorization_is_rejected(
    authorization: str | None,
) -> None:
    with pytest.raises(HTTPException) as caught:
        await api.chat(ChatRequest(query="Hola"), authorization)
    assert caught.value.status_code == 401
    assert caught.value.detail == "Authentication required"


@pytest.mark.asyncio
async def test_invalid_supabase_token_is_rejected_without_exposing_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def reject(_authorization: str | None) -> UUID:
        raise api.AuthenticationError("token detail that must not escape")

    monkeypatch.setattr(api, "verify_supabase_access_token", reject)
    with pytest.raises(HTTPException) as caught:
        await api.chat(ChatRequest(query="Hola"), AUTH)
    assert caught.value.status_code == 401
    assert caught.value.detail == "Authentication required"


@pytest.mark.asyncio
async def test_valid_token_subject_reaches_graph_as_canonical_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invocations: list[tuple[dict[str, Any], dict[str, Any]]] = []

    class FakeGraph:
        async def ainvoke(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            invocations.append((state, config))
            return {"message": "Hola.", "user_profile": {}}

    monkeypatch.setattr(api, "verify_supabase_access_token", _authenticate_as_a)
    monkeypatch.setattr(api, "graph", FakeGraph())

    response = await api.chat(ChatRequest(query="Hola"), AUTH)

    state, config = invocations[0]
    assert state == {"user_query": "Hola", "current_user_id": USER_A}
    assert isinstance(state["current_user_id"], UUID)
    assert config == {"configurable": {"thread_id": f"user:{USER_A}"}}
    assert response.message == "Hola."


@pytest.mark.asyncio
async def test_structured_financial_action_preserves_authenticated_user_in_graph_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invocations: list[tuple[dict[str, Any], dict[str, Any]]] = []
    tool_calls: list[tuple[str, dict[str, Any] | None, UUID | None]] = []

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

    async def execute(
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        current_user_id: UUID | None = None,
    ) -> MCPToolExecution:
        tool_calls.append((name, arguments, current_user_id))
        return _execution(
            structured_content={
                "ok": True,
                "action": action,
                "request": {"intent": "transactions"},
                "trustedScope": {"user_id": str(USER_A)},
            }
        )

    monkeypatch.setattr(api, "verify_supabase_access_token", _authenticate_as_a)
    monkeypatch.setattr(api, "execute_remote_tool", execute)
    monkeypatch.setattr(api, "graph", FakeGraph())
    response = await api.chat(ChatRequest(action=action), AUTH)

    state, config = invocations[0]
    assert state["current_user_id"] == USER_A
    assert state["requested_intent"] == "transactions"
    assert state["action_requested"] is True
    assert config == {"configurable": {"thread_id": f"user:{USER_A}"}}
    assert response.message == "Movimientos listos."
    assert tool_calls == [("a2ui_action", action, USER_A)]


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

    async def reject(
        _name: str,
        _arguments: dict[str, Any] | None = None,
        *,
        current_user_id: UUID | None = None,
    ) -> MCPToolExecution:
        del current_user_id
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

    response = await api.chat(ChatRequest(action=action), AUTH)

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
    calls: list[tuple[str, dict[str, Any] | None, UUID | None]] = []

    async def execute(
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        current_user_id: UUID | None = None,
    ) -> MCPToolExecution:
        calls.append((name, arguments, current_user_id))
        return _execution()

    monkeypatch.setattr(api, "verify_supabase_access_token", _authenticate_as_a)
    monkeypatch.setattr(api, "execute_remote_tool", execute)
    response = await api.chat(ChatRequest(query=_legacy_action(ACTION)), AUTH)
    assert calls == [("a2ui_action", ACTION, USER_A)]
    assert response.data == {"ok": True}


@pytest.mark.asyncio
async def test_a_plain_query_reaches_the_graph_and_never_a_phrase_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No query bypasses the graph.

    `database_overview` used to be reached by matching phrases before the graph
    ran. It is a model-visible tool, so the model now discovers and calls it
    like any other capability, and the boundary's only job is to authenticate
    and hand the turn to the graph.
    """
    invoked: list[dict[str, Any]] = []

    async def execute(
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        current_user_id: UUID | None = None,
    ) -> MCPToolExecution:
        raise AssertionError(f"the boundary must not call {name} itself")

    class _Graph:
        async def ainvoke(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            invoked.append(state)
            return {"message": "Database overview loaded.", "tool_loop_count": 2}

    monkeypatch.setattr(api, "verify_supabase_access_token", _authenticate_as_a)
    monkeypatch.setattr(api, "execute_remote_tool", execute)
    monkeypatch.setattr(api, "graph", _Graph())
    response = await api.chat(
        ChatRequest(query="Muéstrame los objetos disponibles de la base de datos"),
        AUTH,
    )
    assert [state["user_query"] for state in invoked] == [
        "Muéstrame los objetos disponibles de la base de datos"
    ]
    assert invoked[0]["current_user_id"] == USER_A
    assert response.message == "Database overview loaded."


def _preflight(origin: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api, "_allowed_origins", lambda environment=None: [])
    application = FastAPI()
    application.post("/api/v1/agent/chat")(lambda: {"ok": True})
    api._install_cors(application)
    with TestClient(application) as client:
        return client.options(
            "/api/v1/agent/chat",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )


def test_the_expo_web_dev_server_passes_preflight_on_any_localhost_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without this the browser blocks every call before it is ever sent."""
    for origin in ("http://localhost:8082", "http://localhost:19006", "http://127.0.0.1:8081"):
        response = _preflight(origin, monkeypatch)
        assert response.headers.get("access-control-allow-origin") == origin, origin
        assert "authorization" in response.headers.get("access-control-allow-headers", "").lower()


def test_an_unrelated_origin_is_not_granted_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _preflight("https://evil.example.com", monkeypatch)
    assert "access-control-allow-origin" not in response.headers


def test_configured_origins_replace_the_localhost_pattern() -> None:
    assert api._allowed_origins(
        {"AGENT_ALLOWED_ORIGINS": "https://app.example.com/, , https://b.io"}
    ) == [
        "https://app.example.com",
        "https://b.io",
    ]
    assert api._allowed_origins({}) == []


# --------------------------------------------------------------------------
# The streaming route
# --------------------------------------------------------------------------


class _StatusGraph:
    """A graph that reports two phases and then finishes."""

    def __init__(self, *payloads: dict[str, Any]) -> None:
        self._payloads = payloads
        self.states: list[dict[str, Any]] = []

    async def astream(
        self, state: dict[str, Any], *, config: dict[str, Any], stream_mode: list[str]
    ) -> Any:
        self.states.append(state)
        assert stream_mode == ["custom", "values"]
        for payload in self._payloads:
            yield "custom", payload
        yield "values", {"message": "Listo.", "user_profile": {}, "tool_loop_count": 1}


async def _lines(request: ChatRequest, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    monkeypatch.setattr(api, "verify_supabase_access_token", _authenticate_as_a)
    body = await api.chat_stream(request, AUTH)
    chunks = [chunk async for chunk in body.body_iterator]
    text = b"".join(
        chunk if isinstance(chunk, bytes) else chunk.encode() for chunk in chunks
    ).decode()
    return [json.loads(line) for line in text.splitlines() if line]


@pytest.mark.asyncio
async def test_the_stream_reports_phases_then_one_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _StatusGraph(
        {"type": "agent_status", "status": "interpreting"},
        {"type": "agent_status", "status": "building_ui"},
    )
    monkeypatch.setattr(api, "graph", graph)

    lines = await _lines(ChatRequest(query="¿Cuánto tengo?"), monkeypatch)

    assert [line["type"] for line in lines] == ["agent_status", "agent_status", "result"]
    assert [line["status"] for line in lines[:2]] == ["interpreting", "building_ui"]
    assert lines[-1]["result"]["message"] == "Listo."
    assert graph.states[0]["current_user_id"] == USER_A


@pytest.mark.asyncio
async def test_the_stream_forwards_no_payload_that_is_not_a_known_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A node cannot turn the progress channel into a reasoning channel."""
    graph = _StatusGraph(
        {"type": "reasoning", "text": "the user looks overdrawn"},
        {"type": "agent_status", "status": "pondering"},
        {"type": "agent_status", "status": "validating_ui", "note": "internal"},
    )
    monkeypatch.setattr(api, "graph", graph)

    lines = await _lines(ChatRequest(query="¿Cuánto tengo?"), monkeypatch)

    assert [line["type"] for line in lines] == ["agent_status", "result"]
    assert lines[0] == {"type": "agent_status", "status": "validating_ui"}
    assert "overdrawn" not in json.dumps(lines)


@pytest.mark.asyncio
async def test_the_stream_refuses_a_body_user_id_that_disagrees_with_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "verify_supabase_access_token", _authenticate_as_a)
    with pytest.raises(HTTPException) as raised:
        body = await api.chat_stream(ChatRequest(query="Hola", user_id=USER_B), AUTH)
        [chunk async for chunk in body.body_iterator]
    assert raised.value.status_code == 403
