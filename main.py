"""FastAPI entrypoint for the FluidBank orchestrator."""

from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from graph import graph
from personas import PERSONA_USER_IDS, Persona
from schemas.a2ui import A2UIPayload

load_dotenv()

app = FastAPI(title="FluidBank Orchestrator", version="0.1.0")


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    persona: Persona = "ana"


@app.post("/api/v1/agent/chat", response_model=A2UIPayload)
async def chat(request: ChatRequest) -> A2UIPayload:
    """Run the graph against real Supabase-backed context and return the
    validated A2UI payload."""
    result = await graph.ainvoke(
        {"user_query": request.query, "user_id": PERSONA_USER_IDS[request.persona]}
    )
    return result["a2ui_response"]
