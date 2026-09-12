"""FastAPI entrypoint for the FluidBank orchestrator.

The agent never builds or modifies A2UI. Once hackmty2026-mcp exposes its
A2UI-producing tools, this endpoint will call them and relay the returned
payload under `a2ui` untouched. Until then, `a2ui` is always null.
"""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .graph import graph
from .personas import PERSONA_USER_IDS, Persona

load_dotenv()

app = FastAPI(title="FluidBank Orchestrator", version="0.1.0")


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    persona: Persona = "ana"


class A2UIEnvelope(BaseModel):
    """Minimal transport envelope for the A2UI payload the MCP server
    returns. The agent must never construct or edit its contents - only
    relay it."""

    resource_uri: str
    messages: list[dict[str, Any]]


class ChatResponse(BaseModel):
    message: str
    data: dict[str, Any]
    a2ui: A2UIEnvelope | None = None


@app.post("/api/v1/agent/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Run the graph against real Supabase-backed context and return a
    conversational reply. `a2ui` stays null until the MCP server's A2UI
    tools exist - this endpoint will relay them, never generate them."""
    result = await graph.ainvoke(
        {"user_query": request.query, "user_id": PERSONA_USER_IDS[request.persona]}
    )
    data: dict[str, Any] = dict(result["user_profile"])
    if "months" in result:
        data["months"] = result["months"]
    return ChatResponse(message=result["message"], data=data, a2ui=None)
