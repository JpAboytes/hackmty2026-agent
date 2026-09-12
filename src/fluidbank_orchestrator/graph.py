"""LangGraph workflow for ordinary conversational requests.

The API routes deterministic A2UI domain tools and actions outside this graph.
This graph preserves the existing context-fetching and Gemini reasoning path;
it never receives or generates A2UI messages.
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .mcp_client import UserContextError, fetch_user_context
from .state import GraphState, UserProfile

_FALLBACK_PROFILE: UserProfile = {
    "literacy_level": "medium",
    "font_scale": "lg",
    "contrast": "high",
    "hit_target": "large",
    "overdraft_risk": 0.82,
    "recurring_expenses": 3200.0,
    "available_balance": 1200.0,
}


class _Intent(BaseModel):
    message: str
    months: int | None = None


async def fetch_context_node(state: GraphState) -> dict[str, UserProfile]:
    """Fetch accessibility/financial context through the MCP server.

    Falls back to a static demo profile if Supabase or the MCP server is
    unreachable, per the local demo mode required by PROJECT_SPEC.MD.
    """
    try:
        profile = await fetch_user_context(state["user_email"])
    except UserContextError:
        profile = _FALLBACK_PROFILE.copy()
    return {"user_profile": profile}


def _fallback_intent() -> _Intent:
    """Deterministic reply used when Gemini is unavailable."""
    return _Intent(message="No pude generar una respuesta personalizada en este momento.")


_INTENT_PROMPT = """Eres el asistente conversacional de un agente bancario accesible.
Responde brevemente (1-2 oraciones, en español) a la consulta del usuario usando
únicamente los datos de su contexto financiero. Nunca inventes cifras que no
estén en el contexto, y nunca describas botones, pantallas ni componentes
visuales: eso lo decide otro sistema.

Si la consulta pide simular una compra a plazos, incluye "months" (entero,
3 a 18) con el número de meses solicitado; si no aplica, deja "months" como
null.

Ajusta el vocabulario al nivel de alfabetización financiera del usuario:
"low" = lenguaje muy simple y directo, "medium" = lenguaje claro, "high" = puede
incluir más detalle.

Contexto del usuario:
- Saldo disponible: {available_balance}
- Gastos recurrentes mensuales: {recurring_expenses}
- Riesgo de sobregiro (0 a 1): {overdraft_risk}
- Nivel de alfabetización financiera: {literacy_level}

Consulta del usuario: "{query}"
"""


async def intent_node(state: GraphState) -> dict[str, object]:
    """Draft a grounded conversational reply and extract structured
    parameters (e.g. months) with Gemini. Falls back to a plain message if
    the LLM call fails, per the local demo mode required by PROJECT_SPEC.MD.
    """
    profile = state["user_profile"]
    query = state["user_query"]
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"),
            contents=_INTENT_PROMPT.format(query=query, **profile),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_Intent,
            ),
        )
        parsed = response.parsed
        intent = parsed if isinstance(parsed, _Intent) else _Intent.model_validate(parsed)
    except Exception:  # noqa: BLE001 - any LLM failure falls back to local demo mode
        intent = _fallback_intent()

    result: dict[str, object] = {"message": intent.message}
    if intent.months is not None:
        result["months"] = intent.months
    return result


workflow = StateGraph(GraphState)
workflow.add_node("fetch_context", fetch_context_node)
workflow.add_node("intent", intent_node)
workflow.add_edge(START, "fetch_context")
workflow.add_edge("fetch_context", "intent")
workflow.add_edge("intent", END)
graph = workflow.compile()
