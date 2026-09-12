"""LangGraph workflow that fetches real Supabase-backed context through the
read-only MCP server, interprets the natural-language query with Gemini, and
produces a schema-validated A2UI response.

Both the MCP fetch and the Gemini call fall back to deterministic local logic
on failure, preserving the local demo mode required by PROJECT_SPEC.MD.
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from mcp_client import UserContextError, fetch_user_context
from schemas.a2ui import (
    A2UIAccessibility,
    A2UIAction,
    A2UIComponent,
    A2UIMeta,
    A2UIPayload,
    A2UISurface,
    ActionIntent,
    ComponentType,
    TemplateId,
)
from state import GraphState, UserProfile

_FALLBACK_PROFILE: UserProfile = {
    "literacy_level": "medium",
    "font_scale": "lg",
    "contrast": "high",
    "hit_target": "large",
    "overdraft_risk": 0.82,
    "recurring_expenses": 3200.0,
    "available_balance": 1200.0,
}

_SLIDER_STEPS = (3, 6, 9, 12, 15, 18)


class _Intent(BaseModel):
    template_id: TemplateId
    narrative: str
    months: int | None = None


async def fetch_context_node(state: GraphState) -> dict[str, UserProfile]:
    """Fetch accessibility/financial context through the MCP server.

    Falls back to a static demo profile if Supabase or the MCP server is
    unreachable, per the local demo mode required by PROJECT_SPEC.MD.
    """
    try:
        profile = await fetch_user_context(state["user_id"])
    except UserContextError:
        profile = dict(_FALLBACK_PROFILE)
    return {"user_profile": profile}  # type: ignore[typeddict-item]


def _clamp_months(months: int | None) -> int:
    if months is None:
        return 6
    return min(_SLIDER_STEPS, key=lambda step: abs(step - months))


def _fallback_intent(query: str, profile: UserProfile) -> _Intent:
    """Deterministic keyword routing used when Gemini is unavailable."""
    lowered = query.lower()
    if any(term in lowered for term in ("suscripción", "suscripciones", "gasto recurrente")):
        return _Intent(
            template_id="Template_Subscriptions",
            narrative="Revisemos tus gastos recurrentes para encontrar oportunidades de ahorro.",
        )
    if any(term in lowered for term in ("simular", "compra", "meses", "proyección")):
        return _Intent(
            template_id="Template_Projection",
            narrative="Puedes explorar el impacto mensual de una compra antes de confirmarla.",
            months=6,
        )
    if profile["overdraft_risk"] >= 0.75 or any(
        term in lowered for term in ("sobregiro", "riesgo", "urgente", "dinero")
    ):
        return _Intent(
            template_id="Template_Crisis_Flujo",
            narrative="Atención: tienes un riesgo alto de sobregiro antes del viernes.",
        )
    return _Intent(
        template_id="Template_Projection",
        narrative="Puedes explorar el impacto mensual de una compra antes de confirmarla.",
        months=6,
    )


_INTENT_PROMPT = """Eres el clasificador de intención de un agente bancario accesible.
Dada la consulta del usuario y su contexto financiero, elige la plantilla de interfaz
más adecuada y escribe una narrativa breve (1-2 oraciones, en español) que la explique.

Plantillas disponibles:
- Template_Crisis_Flujo: advertencia financiera urgente (bajo saldo frente a gastos fijos).
- Template_Subscriptions: revisión de suscripciones y gastos recurrentes.
- Template_Projection: simulación de una compra a meses.

Si la plantilla es Template_Projection, incluye "months" (entero, 3 a 18) con el número
de meses que el usuario pidió; si no especificó ninguno, usa 6.

Ajusta el vocabulario de la narrativa al nivel de alfabetización financiera del usuario:
"low" = lenguaje muy simple y directo, "medium" = lenguaje claro, "high" = puede incluir
más detalle. Nunca inventes cifras que no estén en el contexto.

Contexto del usuario:
- Saldo disponible: {available_balance}
- Gastos recurrentes mensuales: {recurring_expenses}
- Riesgo de sobregiro (0 a 1): {overdraft_risk}
- Nivel de alfabetización financiera: {literacy_level}

Consulta del usuario: "{query}"
"""


async def intent_node(state: GraphState) -> dict[str, object]:
    """Classify the query into a template using Gemini, adapting the
    narrative to the user's literacy level. Falls back to deterministic
    keyword routing if the LLM call fails."""
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
        intent = response.parsed
        if intent is None:
            raise ValueError("Gemini returned no parsed intent")
    except Exception:  # noqa: BLE001 - any LLM failure falls back to local demo mode
        intent = _fallback_intent(query, profile)

    result: dict[str, object] = {
        "selected_template": intent.template_id,
        "narrative": intent.narrative,
    }
    if intent.template_id == "Template_Projection":
        result["months"] = _clamp_months(intent.months)
    return result


def _action(intent: ActionIntent, payload: dict[str, object]) -> dict[str, object]:
    return A2UIAction(type="A2UI_DISPATCH", intent=intent, payload=payload).model_dump()


def _component(
    component_id: str,
    component_type: ComponentType,
    tags: list[str],
    props: dict[str, object],
) -> A2UIComponent:
    return A2UIComponent(id=component_id, type=component_type, tags=tags, props=props)


def a2ui_generator_node(state: GraphState) -> dict[str, A2UIPayload]:
    """Build and validate the selected A2UI template from real context."""
    profile = state["user_profile"]
    template = state["selected_template"]
    narrative = state["narrative"]
    accessibility = A2UIAccessibility(
        font_scale=profile["font_scale"],
        contrast=profile["contrast"],
        hit_target=profile["hit_target"],
    )
    if template == "Template_Crisis_Flujo":
        tags = ["#alerta-roja", "#high-contrast", "#big-targets"]
        components = [
            _component(
                "alert_card_01",
                "Banner",
                ["#alerta-roja"],
                {"variant": "warning", "title": "Riesgo de Liquidez"},
            ),
            _component(
                "action_btn_01",
                "Button",
                ["#big-targets"],
                {
                    "label": "Solicitar Préstamo Express",
                    "action": _action("REQUEST_CREDIT", {"amount": 5000}),
                },
            ),
        ]
    elif template == "Template_Subscriptions":
        tags = ["#high-contrast", "#action-first"]
        components = [
            _component(
                "subscriptions_card_01",
                "MetricCard",
                ["#high-contrast"],
                {"title": "Gastos recurrentes", "value": profile["recurring_expenses"]},
            ),
            _component(
                "subscriptions_btn_01",
                "Button",
                ["#action-first"],
                {
                    "label": "Administrar suscripciones",
                    "action": _action("MANAGE_SUBSCRIPTIONS", {}),
                },
            ),
        ]
    else:
        months = state.get("months", 6)
        tags = ["#high-contrast", "#big-targets"]
        components = [
            _component(
                "projection_header",
                "Banner",
                ["#high-contrast"],
                {"variant": "info", "title": "Simulación de compra"},
            ),
            _component(
                "projection_term",
                "InteractiveSlider",
                ["#big-targets"],
                {
                    "label": "Número de meses",
                    "min": 3,
                    "max": 18,
                    "step": 3,
                    "default_value": months,
                },
            ),
            _component(
                "projection_confirm",
                "Button",
                ["#big-targets"],
                {
                    "label": "Confirmar simulación",
                    "action": _action("CONFIRM_SIMULATION", {"months": months}),
                },
            ),
        ]
    payload = A2UIPayload(
        template_id=template,
        applied_tags=tags,
        meta=A2UIMeta(narrative=narrative, accessibility=accessibility),
        surface=A2UISurface(layout="vertical_stack", components=components),
    )
    return {"a2ui_response": payload}


workflow = StateGraph(GraphState)
workflow.add_node("fetch_context", fetch_context_node)
workflow.add_node("intent", intent_node)
workflow.add_node("a2ui_generator", a2ui_generator_node)
workflow.add_edge(START, "fetch_context")
workflow.add_edge("fetch_context", "intent")
workflow.add_edge("intent", "a2ui_generator")
workflow.add_edge("a2ui_generator", END)
graph = workflow.compile()
