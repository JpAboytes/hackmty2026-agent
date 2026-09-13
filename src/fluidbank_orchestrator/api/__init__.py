"""The HTTP boundary: authenticate, dispatch one turn, return one envelope.

This module owns the ASGI app, the browser boundary, and the order in which an
authenticated turn is dispatched: a structured action re-enters through MCP's
allowlist, and everything else goes to the graph. No phrase-to-tool or
phrase-to-view routing happens here - the graph's deterministic policy node
only gates scope and safety, then the model decides what an accepted banking
query needs. There is exactly one query route. It deliberately keeps
``verify_supabase_access_token``,
``execute_remote_tool`` and ``graph`` as module-level names, so the whole
boundary can be exercised with those three dependencies substituted.

Two shapes of the same turn are exposed. ``/chat`` returns one JSON envelope.
``/chat/stream`` returns the same envelope preceded by coarse
``agent_status`` lines, as newline-delimited JSON, so the client can show which
phase the turn is in. The status lines carry an identifier and nothing else.

Supporting modules:

* ``schemas.chat``  - the request/response wire contract.
* ``actions``       - the A2UI action transport and its trust checks.
* ``responses``     - building and logging the client envelope.

Identity comes from the bearer token and from nowhere else: not from the body,
not from action context, not from the model.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Mapping
from typing import Annotated, Any
from uuid import UUID

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ..agent.status import AGENT_STATUS_EVENT, AGENT_STATUSES
from ..auth import AuthenticationError, verify_supabase_access_token
from ..graph import graph
from ..mcp_client import (
    MCPConfigurationError,
    UserContextError,
    execute_remote_tool,
    mcp_session,
    require_current_user_id,
)
from ..observability import configure_logging, end_turn, event, stage, start_turn
from ..schemas.chat import ChatRequest, ChatResponse
from .actions import (
    ACTION_REFRESH_INTENTS,
    FINANCIAL_VIEW_ACTION,
    action_payload,
    successful_action_result,
    trusted_financial_intent,
)
from .responses import (
    invalid_action_response,
    log_client_response,
    response_from_graph,
    response_from_tool,
    unavailable_response,
)

load_dotenv()
configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="FluidBank Orchestrator", version="0.1.0")

# Expo's web build is a browser origin, so it is subject to CORS while the
# native builds are not. The dev server picks whatever port is free, so
# localhost is matched by pattern rather than enumerated; a deployed web origin
# must be listed explicitly in AGENT_ALLOWED_ORIGINS.
_LOCALHOST_ORIGIN_PATTERN = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"

#: Media type of the streaming route: one JSON object per line.
NDJSON_MEDIA_TYPE = "application/x-ndjson"


def _allowed_origins(environment: Mapping[str, str] | None = None) -> list[str]:
    """Parse the explicitly configured browser origins, if any."""
    env = os.environ if environment is None else environment
    raw = env.get("AGENT_ALLOWED_ORIGINS", "")
    return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


def _install_cors(application: FastAPI) -> None:
    """Allow the configured browser origins, or any localhost port in development.

    Credentials stay off: this API authenticates with an explicit Authorization
    header, never a cookie, so no origin needs permission to send one.
    """
    configured = _allowed_origins()
    application.add_middleware(
        CORSMiddleware,
        allow_origins=configured,
        allow_origin_regex=None if configured else _LOCALHOST_ORIGIN_PATTERN,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
        max_age=600,
    )
    logger.info(
        "CORS configured origins=%s",
        ",".join(configured) if configured else "localhost-pattern",
    )


_install_cors(app)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness/readiness check for the Cloud Run container.

    Deliberately not named /healthz: that exact path is intercepted at
    Google's infrastructure level on some Cloud Run configurations and never
    reaches the container.
    """
    return {"status": "ok"}


@app.post("/api/v1/agent/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> ChatResponse:
    """Run one chat turn under a single correlated, timed log stream."""
    start_turn(
        input="action" if request.action is not None else "query",
        query_chars=len(request.query) if request.query is not None else None,
    )
    status = "ok"
    try:
        return await _handle_chat(request, authorization)
    except HTTPException as exc:
        status = f"http_{exc.status_code}"
        raise
    except Exception as exc:
        status = f"error:{type(exc).__name__}"
        raise
    finally:
        end_turn(status=status)


async def _handle_chat(
    request: ChatRequest,
    authorization: str | None,
) -> ChatResponse:
    """Authenticate, parse the action form, then dispatch inside one MCP session."""
    async with stage("http.auth") as step:
        try:
            current_user_id = require_current_user_id(
                await verify_supabase_access_token(authorization)
            )
        except (AuthenticationError, UserContextError) as exc:
            raise HTTPException(status_code=401, detail="Authentication required") from exc
        # Only the first octet: enough to correlate a turn, not to identify anyone.
        step.set(user=str(current_user_id)[:8], claimed_user=request.user_id is not None)
    if request.user_id is not None and request.user_id != current_user_id:
        event("http.identity_mismatch")
        raise HTTPException(status_code=403, detail="Authenticated user does not match user_id")

    action, invalid_reason = action_payload(request)
    if invalid_reason is not None:
        return invalid_action_response(invalid_reason)

    # One MCP session for the whole turn: the routes below make between one
    # and six calls, and each used to pay its own handshake.
    async with mcp_session():
        if request.account_id is not None:
            await _require_owned_account(request.account_id, current_user_id)
        return await _route_request(action, request.query, current_user_id)


async def _require_owned_account(account_id: UUID, current_user_id: UUID) -> None:
    """Verify Expo's account context inside the authenticated MCP scope."""
    try:
        execution = await execute_remote_tool(
            "get_accounts",
            {"request": {"account_ids": [str(account_id)]}},
            current_user_id=current_user_id,
        )
    except (MCPConfigurationError, UserContextError) as exc:
        raise HTTPException(status_code=503, detail="Account verification unavailable") from exc
    structured = execution.result.structured_content
    accounts = structured.get("accounts") if isinstance(structured, dict) else None
    if not isinstance(accounts, list) or not any(
        isinstance(account, dict) and account.get("id") == str(account_id)
        for account in accounts
    ):
        event("http.account_mismatch")
        raise HTTPException(status_code=403, detail="Account does not belong to user")


async def _route_request(
    action: dict[str, Any] | None,
    query: str | None,
    current_user_id: UUID,
) -> ChatResponse:
    """Dispatch one authenticated turn over the session already opened for it."""
    if action is not None:
        return await _run_action(action, current_user_id)

    assert query is not None
    event("route.selected", route="graph")
    async with stage("graph.invoke", entry="query"):
        try:
            result = await graph.ainvoke(
                _query_turn(query, current_user_id),
                config=_graph_config(current_user_id),
            )
        except (MCPConfigurationError, UserContextError):
            return unavailable_response()
    return log_client_response("graph", response_from_graph(result))


def _query_turn(query: str, current_user_id: UUID) -> dict[str, Any]:
    # The two action channels are sent explicitly so a plain query can never
    # inherit an approved view from a checkpointed earlier turn.
    return {
        "user_query": query,
        "current_user_id": current_user_id,
        "requested_intent": None,
        "action_requested": False,
    }


def _action_turn(action_name: str, intent: str, current_user_id: UUID) -> dict[str, Any]:
    """The graph input for a view the user already approved through A2UI.

    `requested_intent` is trusted because MCP produced it, never the model, and
    it pins the presentation so an approved action cannot be upgraded into a
    different view.
    """
    return {
        "user_query": f"{action_name}:{intent}",
        "requested_intent": intent,
        "action_requested": True,
        "current_user_id": current_user_id,
    }


def _graph_config(current_user_id: UUID) -> dict[str, Any]:
    return {"configurable": {"thread_id": f"user:{current_user_id}"}}


async def _run_action(action: dict[str, Any], current_user_id: UUID) -> ChatResponse:
    """Validate the action with MCP, then re-enter the graph only if it says so."""
    event("route.selected", route="action", action=action["name"])
    try:
        async with stage("route.action", action=action["name"]):
            action_execution = await execute_remote_tool(
                "a2ui_action",
                action,
                current_user_id=current_user_id,
            )
    except (MCPConfigurationError, UserContextError):
        return unavailable_response()
    if action["name"] != FINANCIAL_VIEW_ACTION:
        refresh_intent = ACTION_REFRESH_INTENTS.get(action["name"])
        action_result = successful_action_result(action_execution.result.structured_content)
        if refresh_intent is None or action_result is None:
            return log_client_response("action", response_from_tool(action_execution))
        event("route.selected", route="action_refresh", intent=refresh_intent)
        try:
            async with stage("graph.invoke", entry="action_refresh", intent=refresh_intent):
                result = await graph.ainvoke(
                    _action_turn(action["name"], refresh_intent, current_user_id),
                    config=_graph_config(current_user_id),
                )
        except Exception as exc:
            # The write was already committed and explicitly confirmed by MCP.
            # A failed refresh must not turn that success into a misleading
            # "save failed" response that encourages a duplicate retry.
            logger.warning("Post-action view refresh failed: %s", type(exc).__name__)
            return log_client_response(
                "action_refresh_failed", response_from_tool(action_execution)
            )
        response = response_from_graph(result)
        response.data["actionResult"] = action_result
        return log_client_response("action_refresh", response)
    intent = trusted_financial_intent(action_execution.result.structured_content, current_user_id)
    if intent is None:
        return invalid_action_response("untrusted_action_result")
    event("route.selected", route="action_graph", intent=intent)
    async with stage("graph.invoke", entry="action", intent=intent):
        result = await graph.ainvoke(
            _action_turn(FINANCIAL_VIEW_ACTION, intent, current_user_id),
            config=_graph_config(current_user_id),
        )
    return log_client_response("action_graph", response_from_graph(result))


def _status_line(payload: Mapping[str, Any]) -> str | None:
    """Serialize one custom stream payload, or drop it.

    The allowlist is the security boundary of this route: only the discriminator
    and a known status identifier are forwarded, so no node can widen the stream
    into a reasoning channel by writing a richer payload.
    """
    if payload.get("type") != AGENT_STATUS_EVENT:
        return None
    status = payload.get("status")
    if status not in AGENT_STATUSES:
        return None
    return json.dumps({"type": AGENT_STATUS_EVENT, "status": status}, separators=(",", ":"))


async def _stream_turn(
    request: ChatRequest,
    authorization: str | None,
) -> AsyncIterator[str]:
    """Yield coarse status lines while the turn runs, then the one envelope.

    An action turn is not streamed phase by phase: it is dispatched through the
    same code path as `/chat` and reported as a single `result` line, so the two
    routes cannot disagree about what an action does.
    """
    async with stage("http.auth") as step:
        try:
            current_user_id = require_current_user_id(
                await verify_supabase_access_token(authorization)
            )
        except (AuthenticationError, UserContextError) as exc:
            raise HTTPException(status_code=401, detail="Authentication required") from exc
        step.set(user=str(current_user_id)[:8], claimed_user=request.user_id is not None)
    if request.user_id is not None and request.user_id != current_user_id:
        event("http.identity_mismatch")
        raise HTTPException(status_code=403, detail="Authenticated user does not match user_id")

    action, invalid_reason = action_payload(request)
    if invalid_reason is not None:
        yield _result_line(invalid_action_response(invalid_reason))
        return

    async with mcp_session():
        # The same ownership check `/chat` runs: the streaming route must not
        # be a way to skip it.
        if request.account_id is not None:
            await _require_owned_account(request.account_id, current_user_id)
        if action is not None or request.query is None:
            yield _result_line(await _route_request(action, request.query, current_user_id))
            return
        event("route.selected", route="graph_stream")
        # `values` yields after every superstep; only the terminal state is the
        # answer, so the envelope is built once rather than once per step.
        final_state: dict[str, Any] | None = None
        failed = False
        try:
            async with stage("graph.stream", entry="query"):
                async for mode, payload in graph.astream(
                    _query_turn(request.query, current_user_id),
                    config=_graph_config(current_user_id),
                    stream_mode=["custom", "values"],
                ):
                    if mode == "custom" and isinstance(payload, Mapping):
                        line = _status_line(payload)
                        if line is not None:
                            yield line
                    elif mode == "values" and isinstance(payload, dict):
                        final_state = payload
        except (MCPConfigurationError, UserContextError):
            failed = True
        response = (
            unavailable_response()
            if failed or final_state is None
            else response_from_graph(final_state)
        )
        yield _result_line(log_client_response("graph_stream", response))


def _result_line(response: ChatResponse) -> str:
    return json.dumps(
        {"type": "result", "result": response.model_dump(mode="json")},
        separators=(",", ":"),
    )


@app.post("/api/v1/agent/chat/stream")
async def chat_stream(
    request: ChatRequest,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> StreamingResponse:
    """Run one chat turn, reporting coarse lifecycle phases as they happen."""
    start_turn(
        input="action" if request.action is not None else "query",
        query_chars=len(request.query) if request.query is not None else None,
        transport="stream",
    )

    async def body() -> AsyncIterator[bytes]:
        status = "ok"
        try:
            async for line in _stream_turn(request, authorization):
                yield f"{line}\n".encode()
        except HTTPException as exc:
            status = f"http_{exc.status_code}"
            raise
        except Exception as exc:
            status = f"error:{type(exc).__name__}"
            raise
        finally:
            end_turn(status=status)

    return StreamingResponse(body(), media_type=NDJSON_MEDIA_TYPE)


__all__ = [
    "NDJSON_MEDIA_TYPE",
    "ChatRequest",
    "ChatResponse",
    "app",
    "chat",
    "chat_stream",
    "health",
]
