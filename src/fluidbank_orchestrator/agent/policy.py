"""Deterministic scope and prompt-injection guards for conversational turns.

The model is useful for choosing among bounded banking capabilities, but it is
not the authority on whether a request may reach those capabilities.  This
module is deliberately model-free: a denied request never enters a prompt,
never loads user context, and never reaches MCP.

This is a safety/scope gate, not a financial intent router.  It never chooses a
tool, presentation, or action.  Approved A2UI action re-entry bypasses the text
gate because its name and intent were already validated by MCP.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

PolicyReason = Literal["prompt_injection", "executable_code", "history", "out_of_scope"]

_REFUSALS: dict[PolicyReason, str] = {
    "prompt_injection": (
        "No puedo cambiar, revelar ni evadir mis políticas o instrucciones. "
        "Solo puedo ayudar con consultas y operaciones bancarias permitidas."
    ),
    "executable_code": (
        "No genero scripts ni código ejecutable. "
        "Solo puedo ayudar con consultas y operaciones bancarias permitidas."
    ),
    "history": (
        "No proporciono relatos ni explicaciones históricas. "
        "Solo puedo ayudar con consultas y operaciones bancarias permitidas."
    ),
    "out_of_scope": "Solo puedo ayudar con consultas y operaciones bancarias permitidas.",
}


@dataclass(frozen=True, slots=True)
class QueryPolicyDecision:
    """Result of the deterministic query gate."""

    allowed: bool
    reason: PolicyReason | None = None
    message: str = ""


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(without_marks.split())


_PROMPT_INJECTION_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:ignora|omite|olvida|desobedece|ignore|forget|bypass|override)\b.{0,60}"
        r"\b(?:instrucciones?|reglas?|politicas?|prompt|system|developer|mensajes?)\b",
        r"\b(?:system prompt|prompt (?:del )?sistema|developer message|"
        r"mensaje (?:del )?desarrollador)\b",
        r"\b(?:prompt|jailbreak|system|developer)\b",
        r"\b(?:instrucciones?|reglas?|politicas?)\b.{0,20}\b(?:internas?|ocultas?|"
        r"del sistema)\b",
        r"\b(?:revela|muestra|imprime|repite|reveal|show|print|repeat)\b.{0,50}"
        r"\b(?:prompt|instrucciones? internas?|system|developer)\b",
        r"\b(?:nuevas?|new|previous|prior|anteriores?|previas?)\b.{0,30}"
        r"\b(?:instrucciones?|instructions?|directions?|reglas?|indicaciones?)\b",
        r"\b(?:haz|hace|do)\b.{0,20}\b(?:lo contrario|the opposite)\b",
        r"\b(?:jailbreak|dan mode|modo desarrollador|prompt injection)\b",
        r"\b(?:actua|act|finge|pretende|roleplay)\b.{0,35}\b(?:as|como|ser)\b",
        r"\b(?:ahora eres|you are now)\b",
        r"\b(?:base64|rot13|decodifica|decode)\b.{0,40}\b(?:instrucciones?|instructions?)\b",
    )
)

_CODE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"```",
        r"\b(?:python|javascript|typescript|node\.js|powershell|bash|java|"
        r"golang|rust|ruby|php|kotlin|sql)\b|c\+\+|c#",
        r"\b(?:script|codigo fuente|source code)\b",
        r"\b(?:escribe|genera|crea|haz|dame|produce|write|generate|create)\b.{0,60}"
        r"\b(?:codigo|code|programa|funcion|clase)\b",
        r"(?:^|\s)(?:import|from|def|class)\s+[a-z_][a-z0-9_.]*",
        r"\b(?:eval|exec|subprocess|os\.system)\s*\(",
    )
)

# ``historial`` is intentionally absent: a transaction or account history is a
# normal banking request.  The guard rejects requests for historical narratives.
_HISTORY_PATTERNS = (
    re.compile(r"\b(?:historia|history)\b"),
    re.compile(r"\b(?:cuentame|narra|explica)\b.{0,40}\b(?:origen|evolucion historica)\b"),
)

_OFF_TOPIC_PATTERNS = (
    re.compile(r"\b(?:dame|haz|escribe|comparte|tienes?)\b.{0,30}\brecetas?\b"),
    re.compile(r"\b(?:cuentame|dime|escribe|haz)\b.{0,30}\b(?:chistes?|poemas?|cuentos?)\b"),
    re.compile(r"\b(?:clima de|pronostico del tiempo|va a llover|hara calor)\b"),
    re.compile(r"\b(?:quien gano|resultado)\b.{0,30}\b(?:mundial|partido|eleccion)\b"),
    re.compile(r"\b(?:diagnostica|receta un medicamento|consejo medico|consejo legal)\b"),
    re.compile(r"\b(?:resuelve|haz)\b.{0,30}\b(?:tarea|examen)\b"),
    re.compile(r"\b(?:traduce|traduccion de)\b"),
)

_BANKING_TERMS = re.compile(
    r"\b(?:"
    r"banc(?:a|ario|aria|arios|arias|o)|finanz(?:a|as|a personal|as personales|iero|iera)|"
    r"cuentas?|saldos?|dinero|tarjetas?|creditos?|debitos?|cat|interes(?:es)?|tasas?|"
    r"comisiones?|transacciones?|movimientos?|transferencias?|depositos?|retiros?|"
    r"pagos?|mensualidades?|compras?|gastos?|ingresos?|presupuestos?|ahorros?|"
    r"metas? de ahorro|deudas?|prestamos?|hipotecas?|suscripciones?|cargos?|abonos?|"
    r"sobregiros?|estados? de cuenta|beneficiarios?|destinatarios?|plazos?|"
    r"amortizaciones?|cuotas?|rendimientos?|inversiones?|cripto(?:moneda)?s?|"
    r"monedas?|mxn|usd|cuanto (?:tengo|debo)|debo|gaste|ahorrar|"
    r"transferir|transfiere|transfiero|pagar|"
    r"historial (?:de )?(?:transacciones|movimientos|pagos|cuenta|crediticio)"
    r")\b"
)

_COURTESY_OR_CAPABILITY = re.compile(
    r"^(?:hola|hey|buenos dias|buenas tardes|buenas noches|gracias|muchas gracias|"
    r"adios|hasta luego|ayuda|ayudame con esto|menu|que puedes hacer|"
    r"como puedes ayudarme)[!?. ]*$"
)


def _matches(patterns: tuple[re.Pattern[str], ...], value: str) -> bool:
    return any(pattern.search(value) is not None for pattern in patterns)


def evaluate_query_policy(query: str) -> QueryPolicyDecision:
    """Fail closed unless a plain-text request is safe and banking-scoped."""
    normalized = _normalize(query)
    reason: PolicyReason | None = None
    if _matches(_PROMPT_INJECTION_PATTERNS, normalized):
        reason = "prompt_injection"
    elif _matches(_CODE_PATTERNS, normalized):
        reason = "executable_code"
    elif _matches(_HISTORY_PATTERNS, normalized):
        reason = "history"
    elif _matches(_OFF_TOPIC_PATTERNS, normalized):
        reason = "out_of_scope"
    elif not (
        _BANKING_TERMS.search(normalized) is not None
        or _COURTESY_OR_CAPABILITY.fullmatch(normalized) is not None
    ):
        reason = "out_of_scope"
    if reason is None:
        return QueryPolicyDecision(allowed=True)
    return QueryPolicyDecision(allowed=False, reason=reason, message=_REFUSALS[reason])


def safe_model_message(message: str) -> str | None:
    """Return a fixed refusal when model prose violates a non-negotiable rule.

    Empty messages are legitimate while the model is selecting tools.  Scope is
    enforced on input; this second boundary catches code, prompt disclosure, or
    historical narration introduced by the model or by hostile tool text.
    """
    if not message.strip():
        return None
    normalized = _normalize(message)
    if _matches(_PROMPT_INJECTION_PATTERNS, normalized):
        return _REFUSALS["prompt_injection"]
    if _matches(_CODE_PATTERNS, normalized):
        return _REFUSALS["executable_code"]
    if _matches(_HISTORY_PATTERNS, normalized):
        return _REFUSALS["history"]
    if _matches(_OFF_TOPIC_PATTERNS, normalized):
        return _REFUSALS["out_of_scope"]
    return None
