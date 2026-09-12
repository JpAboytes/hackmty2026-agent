"""FastAPI entrypoint for chat, domain-tool presentation, and A2UI actions."""

from __future__ import annotations

import logging
import unicodedata
from copy import deepcopy
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from mcp.types import TextContent
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .auth import AuthenticationError, verify_supabase_access_token
from .graph import graph
from .mcp_client import (
    MCPConfigurationError,
    MCPToolExecution,
    UserContextError,
    execute_remote_tool,
    require_current_user_id,
)
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
logger = logging.getLogger(__name__)

app = FastAPI(title="FluidBank Orchestrator", version="0.1.0")


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
    response = ChatResponse(message=message, data=data, a2ui=execution.a2ui)
    if response.a2ui is not None:
        logger.info("A2UI emitted to client resource_uri=%s", response.a2ui.resource_uri)
    return response


def _unavailable_response() -> ChatResponse:
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
    """Route structured actions directly, then domain intents, then ordinary chat."""
    try:
        current_user_id = require_current_user_id(await verify_supabase_access_token(authorization))
    except (AuthenticationError, UserContextError) as exc:
        raise HTTPException(status_code=401, detail="Authentication required") from exc

    action: dict[str, Any] | None = None
    query = request.query
    if request.action is not None:
        try:
            action = request.action.as_payload()
        except A2UIActionPayloadError:
            return ChatResponse(message="La acción de interfaz no es válida.", data={}, a2ui=None)
    try:
        if query is not None:
            action = parse_legacy_a2ui_action(query)
    except A2UIActionPayloadError:
        return ChatResponse(
            message="La acción de interfaz no es válida.",
            data={},
            a2ui=None,
        )

    if action is not None:
        try:
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
                return ChatResponse(
                    message="La acción de interfaz no es válida.", data={}, a2ui=None
                )
            result = await graph.ainvoke(
                {
                    "user_query": f"request_financial_view:{intent}",
                    "requested_intent": intent,
                    "action_requested": True,
                    "current_user_id": current_user_id,
                },
                config={"configurable": {"thread_id": f"user:{current_user_id}"}},
            )
            return _response_from_graph(result)
        return _response_from_tool(action_execution)

    assert query is not None
    if _requests_database_overview(query):
        try:
            return _response_from_tool(
                await execute_remote_tool(
                    "database_overview",
                    {"limit": 50},
                    current_user_id=current_user_id,
                )
            )
        except (MCPConfigurationError, UserContextError):
            return _unavailable_response()

    result = await graph.ainvoke(
        {"user_query": query, "current_user_id": current_user_id},
        config={"configurable": {"thread_id": f"user:{current_user_id}"}},
    )
    return _response_from_graph(result)


def _response_from_graph(result: dict[str, Any]) -> ChatResponse:
    presentation = result.get("financial_presentation")
    if isinstance(presentation, FinancialPresentation):
        return ChatResponse(
            message=presentation.message,
            data=deepcopy(presentation.data),
            a2ui=presentation.a2ui,
        )
    final_execution = result.get("final_tool_execution")
    if isinstance(final_execution, MCPToolExecution):
        return _response_from_tool(final_execution)
    data: dict[str, Any] = dict(result.get("user_profile", {}))
    if "months" in result:
        data["months"] = result["months"]
    return ChatResponse(message=result["message"], data=data, a2ui=None)
