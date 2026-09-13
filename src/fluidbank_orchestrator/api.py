"""FastAPI entrypoint for chat, domain-tool presentation, and A2UI actions."""

from __future__ import annotations

import logging
import os
import unicodedata
from collections.abc import Mapping
from copy import deepcopy
from typing import Annotated, Any
from uuid import UUID

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mcp.types import TextContent
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .auth import AuthenticationError, verify_supabase_access_token
from .graph import graph
from .mcp_client import (
    MCPConfigurationError,
    MCPToolExecution,
    UserContextError,
    execute_remote_tool,
    mcp_session,
    require_current_user_id,
)
from .observability import configure_logging, end_turn, event, preview, stage, start_turn
from .schemas.a2ui import A2UIBundle
from .schemas.a2ui_action import (
    A2UIActionPayloadError,
    A2UIActionRequest,
    parse_legacy_a2ui_action,
)
from .services.financial_presentation import (
    FinancialPresentation,
    normalize_action_intent,
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


class ChatRequest(BaseModel):
    """A chat turn whose security identity comes only from the bearer token."""

    model_config = ConfigDict(extra="forbid")

    query: Annotated[str, Field(min_length=1, max_length=20_000)] | None = None
    action: A2UIActionRequest | None = None
    # Compatibility only. The deployed mobile client sends the caller's id, and
    # forbidding it outright made every request 422. It is never trusted:
    # identity comes from the bearer token, and a value that disagrees with the
    # token is refused rather than honoured.
    user_id: UUID | None = None

    @model_validator(mode="after")
    def exactly_one_input(self) -> ChatRequest:
        if (self.query is None) == (self.action is None):
            raise ValueError("exactly one of query or action is required")
        return self


class ChatResponse(BaseModel):
    message: str
    data: dict[str, Any]
    a2ui: A2UIBundle | None = None


def _requests_database_overview(query: str) -> bool:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", query.casefold())
        if not unicodedata.combining(character)
    )
    if "database" in normalized:
        return any(term in normalized for term in ("overview", "objects", "tables", "views"))
    if "base de datos" in normalized:
        return any(
            term in normalized
            for term in ("resumen", "vista general", "objetos", "tablas", "vistas", "disponibles")
        )
    return "tablas permitidas" in normalized or "tablas disponibles" in normalized


def _response_from_tool(execution: MCPToolExecution) -> ChatResponse:
    message = next(
        (
            content.text.strip()
            for content in execution.result.content
            if isinstance(content, TextContent) and content.text.strip()
        ),
        "La operación se completó.",
    )
    structured = execution.result.structured_content
    data = deepcopy(structured) if isinstance(structured, dict) else {}
    return ChatResponse(message=message, data=data, a2ui=execution.a2ui)


def _log_client_response(route: str, response: ChatResponse) -> None:
    """Describe the exact envelope leaving for Expo, without its financial values."""
    bundle = response.a2ui
    event(
        "client.response",
        route=route,
        message_chars=len(response.message),
        data_keys=",".join(sorted(response.data)) or "none",
        a2ui=bundle is not None,
        resource_uri=bundle.resource_uri if bundle is not None else None,
        a2ui_messages=len(bundle.messages) if bundle is not None else None,
        a2ui_bytes=len(bundle.model_dump_json()) if bundle is not None else None,
    )
    preview("client.response", response.model_dump(mode="json"))


def _invalid_action_response(reason: str) -> ChatResponse:
    event("client.response", route="invalid_action", reason=reason, a2ui=False)
    return ChatResponse(message="La acción de interfaz no es válida.", data={}, a2ui=None)


def _unavailable_response() -> ChatResponse:
    event("client.response", route="unavailable", a2ui=False)
    return ChatResponse(
        message="No pude consultar el servicio de datos en este momento.",
        data={},
        a2ui=None,
    )


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
    """Route structured actions directly, then domain intents, then ordinary chat."""
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

    action: dict[str, Any] | None = None
    query = request.query
    if request.action is not None:
        try:
            action = request.action.as_payload()
        except A2UIActionPayloadError:
            return _invalid_action_response("structured_payload")
    try:
        if query is not None:
            action = parse_legacy_a2ui_action(query)
    except A2UIActionPayloadError:
        return _invalid_action_response("legacy_payload")

    # One MCP session for the whole turn: the routes below make between one
    # and six calls, and each used to pay its own handshake.
    async with mcp_session():
        return await _route_request(action, query, current_user_id)


async def _route_request(
    action: dict[str, Any] | None,
    query: str | None,
    current_user_id: UUID,
) -> ChatResponse:
    """Dispatch one authenticated turn over the session already opened for it."""
    if action is not None:
        event("route.selected", route="action", action=action["name"])
        try:
            async with stage("route.action", action=action["name"]):
                action_execution = await execute_remote_tool(
                    "a2ui_action",
                    action,
                    current_user_id=current_user_id,
                )
        except (MCPConfigurationError, UserContextError):
            return _unavailable_response()
        if action["name"] == "request_financial_view":
            structured = action_execution.result.structured_content
            request_data = structured.get("request") if isinstance(structured, dict) else None
            trusted_scope = structured.get("trustedScope") if isinstance(structured, dict) else None
            intent = (
                normalize_action_intent(request_data.get("intent"))
                if isinstance(request_data, dict)
                else None
            )
            if (
                not isinstance(structured, dict)
                or structured.get("ok") is not True
                or not isinstance(trusted_scope, dict)
                or trusted_scope.get("user_id") != str(current_user_id)
                or intent is None
            ):
                return _invalid_action_response("untrusted_action_result")
            event("route.selected", route="action_graph", intent=intent)
            async with stage("graph.invoke", entry="action", intent=intent):
                result = await graph.ainvoke(
                    {
                        "user_query": f"request_financial_view:{intent}",
                        "requested_intent": intent,
                        "action_requested": True,
                        "current_user_id": current_user_id,
                    },
                    config={"configurable": {"thread_id": f"user:{current_user_id}"}},
                )
            return _logged(_response_from_graph(result), route="action_graph")
        return _logged(_response_from_tool(action_execution), route="action")

    assert query is not None
    from .a2ui_actions.routing import requested_form

    form_name = requested_form(query)
    if form_name is not None:
        try:
            execution = await execute_remote_tool(
                "a2ui_form", {"name": form_name}, current_user_id=current_user_id
            )
            return _logged(_response_from_tool(execution), route="action_form")
        except (MCPConfigurationError, UserContextError):
            return _unavailable_response()
    if _requests_database_overview(query):
        event("route.selected", route="database_overview")
        try:
            async with stage("route.database_overview"):
                overview = await execute_remote_tool(
                    "database_overview",
                    {"limit": 50},
                    current_user_id=current_user_id,
                )
            return _logged(_response_from_tool(overview), route="database_overview")
        except (MCPConfigurationError, UserContextError):
            return _unavailable_response()

    event("route.selected", route="graph")
    async with stage("graph.invoke", entry="query"):
        result = await graph.ainvoke(
            {"user_query": query, "current_user_id": current_user_id},
            config={"configurable": {"thread_id": f"user:{current_user_id}"}},
        )
    return _logged(_response_from_graph(result), route="graph")


def _logged(response: ChatResponse, *, route: str) -> ChatResponse:
    _log_client_response(route, response)
    return response


def _response_from_graph(result: dict[str, Any]) -> ChatResponse:
    presentation = result.get("financial_presentation")
    if isinstance(presentation, FinancialPresentation):
        event("graph.output", source="financial_presentation", intent=presentation.intent)
        return ChatResponse(
            message=presentation.message,
            data=deepcopy(presentation.data),
            a2ui=presentation.a2ui,
        )
    final_execution = result.get("final_tool_execution")
    if isinstance(final_execution, MCPToolExecution):
        event("graph.output", source="final_tool_execution")
        return _response_from_tool(final_execution)
    event("graph.output", source="model_message", turns=result.get("tool_loop_count", 0))
    data: dict[str, Any] = dict(result.get("user_profile", {}))
    if "months" in result:
        data["months"] = result["months"]
    return ChatResponse(message=result["message"], data=data, a2ui=None)
