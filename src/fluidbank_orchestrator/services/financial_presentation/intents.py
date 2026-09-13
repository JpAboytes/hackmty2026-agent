"""The finite financial vocabulary, and how a request is bound to it.

Three separate questions, deliberately kept apart:

* ``normalize_action_intent`` - is this arbitrary value one of the 13 intents?
  Everything crossing a trust boundary (an action, a model answer, graph state)
  passes through it, so no caller can introduce a protocol value.
* ``classify_financial_request`` - does this natural-language query name a
  financial intent? Deterministic phrase matching, no model involved.
* ``select_presentation_intent`` - now that retrieval has happened, which
  presentation do the observations actually support?
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from ...schemas.banking_view import FINANCIAL_INTENTS, FinancialIntent
from .verified_rows import rows_for_table

TITLES: dict[FinancialIntent, str] = {
    "financial-summary": "Tu panorama financiero",
    "transactions": "Tus movimientos",
    "spending-analysis": "Así estás gastando",
    "cash-flow": "Tu flujo de efectivo",
    "budgets": "Tus presupuestos",
    "recurring-payments": "Tus próximos cobros",
    "credit-card": "Tu tarjeta de crédito",
    "debts": "Tus deudas",
    "transfers": "Tus transferencias",
    "card-security": "Seguridad de tu tarjeta",
    "savings-goals": "Tus metas de ahorro",
    "banking-information": "Tu información bancaria",
    "financial-education": "Guía financiera",
}

#: Bilingual phrases that bind natural language to one intent. Order matters:
#: the first matching rule wins, so the more specific intents come first.
_CLASSIFICATION_RULES: tuple[tuple[FinancialIntent, tuple[str, ...]], ...] = (
    (
        "financial-summary",
        (
            "como van mis finanzas",
            "como estan mis finanzas",
            "resumen financiero",
            "cuanto dinero tengo",
            "cual es mi saldo",
            "mi saldo disponible",
            "how much money do i have",
            "account balance",
            "how are my finances",
            "financial overview",
        ),
    ),
    (
        "spending-analysis",
        (
            "mis gastos",
            "en que se me fue",
            "como han cambiado mis gastos",
            "que dias gasto mas",
            "spending analysis",
            "my spending",
        ),
    ),
    (
        "transactions",
        ("movimientos", "transacciones", "transactions", "gaste ayer", "compras de"),
    ),
    (
        "cash-flow",
        (
            "flujo de efectivo",
            "me alcanzara",
            "cash flow",
            "ingresos y gastos",
            "income and expenses",
        ),
    ),
    ("budgets", ("presupuesto", "limite semanal", "budget")),
    (
        "recurring-payments",
        (
            "pagos recurrentes",
            "suscripciones",
            "proximos cobros",
            "proximos pagos",
            "upcoming payments",
            "recurring payments",
        ),
    ),
    ("credit-card", ("tarjeta de credito", "pago minimo", "credit card")),
    ("debts", ("deuda", "liquidar", "debts")),
    ("transfers", ("transferir", "transferencia", "transfer")),
    (
        "card-security",
        ("no reconozco", "aclaracion", "disputa", "bloquear tarjeta", "card security"),
    ),
    (
        "savings-goals",
        ("meta de ahorro", "meta para", "quiero ahorrar", "savings goal"),
    ),
    (
        "banking-information",
        (
            "estado de cuenta",
            "clabe",
            "informacion bancaria",
            "bank statement",
            "beneficiario",
            "beneficiary",
        ),
    ),
    (
        "financial-education",
        ("que pasa si pago", "explicame", "educacion financiera", "financial education"),
    ),
)

#: A transactions request phrased as a spending question is upgraded once the
#: rows to analyse are actually present.
_SPENDING_TERMS = ("gasto", "spending", "categoria", "dias gasto")


def normalized(value: str) -> str:
    """Fold case and strip accents so bilingual matching is stable."""
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )


def normalize_action_intent(value: object) -> FinancialIntent | None:
    """Return the value only if it is one of the shared contract's intents."""
    for intent in FINANCIAL_INTENTS:
        if value == intent:
            return intent
    return None


def classify_financial_request(query: str) -> FinancialIntent | None:
    """Bound natural language to the finite presentation vocabulary."""
    text = normalized(query)
    for intent, phrases in _CLASSIFICATION_RULES:
        if any(phrase in text for phrase in phrases):
            return intent
    return None


def select_presentation_intent(
    requested: FinancialIntent,
    observations: Sequence[Mapping[str, Any]],
    *,
    query: str,
    action_requested: bool = False,
) -> FinancialIntent:
    """Choose presentation semantics only after retrieval observations exist."""
    if requested != "transactions" or action_requested:
        return requested
    text = normalized(query)
    if any(term in text for term in _SPENDING_TERMS):
        if any(rows_for_table(observations, "transactions")):
            return "spending-analysis"
    return requested
