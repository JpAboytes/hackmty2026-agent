"""FastAPI entrypoint for chat, domain-tool presentation, and A2UI actions."""

from __future__ import annotations

import logging
import unicodedata
from copy import deepcopy
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from mcp.types import TextContent
from pydantic import BaseModel, Field

from .graph import graph
from .mcp_client import (
    MCPConfigurationError,
    MCPToolExecution,
    UserContextError,
    execute_remote_tool,
)
from .schemas.a2ui import A2UIBundle
from .schemas.a2ui_action import A2UIActionPayloadError, parse_legacy_a2ui_action

# Structural check only: this is a public.users.email lookup key, not an
# address used for delivery. EmailStr rejects the seeded fluidbank.test users.
_EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

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
    query: str = Field(min_length=1, max_length=20_000)
    email: str = Field(min_length=3, max_length=254, pattern=_EMAIL_PATTERN)


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
async def chat(request: ChatRequest) -> ChatResponse:
    """Route structured actions directly, then domain intents, then ordinary chat."""
    try:
        action = parse_legacy_a2ui_action(request.query)
    except A2UIActionPayloadError:
        return ChatResponse(
            message="La acción de interfaz no es válida.",
            data={},
            a2ui=None,
        )

    if action is not None:
        try:
            return _response_from_tool(await execute_remote_tool("a2ui_action", action))
        except (MCPConfigurationError, UserContextError):
            return _unavailable_response()

    if _requests_database_overview(request.query):
        try:
            return _response_from_tool(
                await execute_remote_tool("database_overview", {"limit": 50})
            )
        except (MCPConfigurationError, UserContextError):
            return _unavailable_response()

    result = await graph.ainvoke({"user_query": request.query, "user_email": request.email})
    final_execution = result.get("final_tool_execution")
    if isinstance(final_execution, MCPToolExecution):
        return _response_from_tool(final_execution)
    data: dict[str, Any] = dict(result["user_profile"])
    if "months" in result:
        data["months"] = result["months"]
    return ChatResponse(message=result["message"], data=data, a2ui=None)
