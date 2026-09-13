"""The HTTP boundary: authenticate, dispatch one turn, return one envelope.

This module owns the ASGI app, the browser boundary, and the order in which an
authenticated turn is dispatched: structured actions first, then the two
deterministic query routes, then the graph. It deliberately keeps
``verify_supabase_access_token``, ``execute_remote_tool`` and ``graph`` as
module-level names, so the whole boundary can be exercised with those three
dependencies substituted.

Supporting modules:

* ``schemas.chat``  - the request/response wire contract.
* ``actions``       - the A2UI action transport and its trust checks.
* ``query_routing`` - deterministic pre-graph query classification.
* ``responses``     - building and logging the client envelope.

Identity comes from the bearer token and from nowhere else: not from the body,
not from action context, not from the model.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Annotated, Any
from uuid import UUID

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
from .actions import FINANCIAL_VIEW_ACTION, action_payload, trusted_financial_intent
from .query_routing import requested_form, requests_database_overview
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

_DATABASE_OVERVIEW_LIMIT = 50


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
        return await _route_request(action, request.query, current_user_id)


async def _route_request(
    action: dict[str, Any] | None,
    query: str | None,
    current_user_id: UUID,
) -> ChatResponse:
    """Dispatch one authenticated turn over the session already opened for it."""
    if action is not None:
        return await _run_action(action, current_user_id)

    assert query is not None
    form_name = requested_form(query)
    if form_name is not None:
        try:
            execution = await execute_remote_tool(
                "a2ui_form", {"name": form_name}, current_user_id=current_user_id
            )
            return log_client_response("action_form", response_from_tool(execution))
        except (MCPConfigurationError, UserContextError):
            return unavailable_response()
    if requests_database_overview(query):
        event("route.selected", route="database_overview")
        try:
            async with stage("route.database_overview"):
                overview = await execute_remote_tool(
                    "database_overview",
                    {"limit": _DATABASE_OVERVIEW_LIMIT},
                    current_user_id=current_user_id,
                )
            return log_client_response("database_overview", response_from_tool(overview))
        except (MCPConfigurationError, UserContextError):
            return unavailable_response()

    event("route.selected", route="graph")
    async with stage("graph.invoke", entry="query"):
        result = await graph.ainvoke(
            {"user_query": query, "current_user_id": current_user_id},
            config={"configurable": {"thread_id": f"user:{current_user_id}"}},
        )
    return log_client_response("graph", response_from_graph(result))


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
        return log_client_response("action", response_from_tool(action_execution))
    intent = trusted_financial_intent(action_execution.result.structured_content, current_user_id)
    if intent is None:
        return invalid_action_response("untrusted_action_result")
    event("route.selected", route="action_graph", intent=intent)
    async with stage("graph.invoke", entry="action", intent=intent):
        result = await graph.ainvoke(
            {
                "user_query": f"{FINANCIAL_VIEW_ACTION}:{intent}",
                "requested_intent": intent,
                "action_requested": True,
                "current_user_id": current_user_id,
            },
            config={"configurable": {"thread_id": f"user:{current_user_id}"}},
        )
    return log_client_response("action_graph", response_from_graph(result))


__all__ = ["ChatRequest", "ChatResponse", "app", "chat", "health"]
