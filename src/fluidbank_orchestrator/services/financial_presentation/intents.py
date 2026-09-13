"""The finite financial vocabulary, and how a request is bound to it.

Two separate questions, deliberately kept apart:

* ``normalize_action_intent`` - is this arbitrary value one of the 13 intents?
  Everything crossing a trust boundary (an action, a model answer, graph state)
  passes through it, so no caller can introduce a protocol value.
* ``select_presentation_intent`` - now that the model has read its tool
  results, which presentation do the observations actually support?

There is deliberately no phrase-based classifier here. Deciding *what the user
is asking for* is the model's job; this module only decides whether a value is
a legal protocol value and whether the data supports the view.
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
