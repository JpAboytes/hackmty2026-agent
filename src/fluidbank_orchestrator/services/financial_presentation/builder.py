"""Interpreting the turn's observations into one validated presentation.

This is the provenance gate for domain answers: an intent is only rendered from
data that arrived through a successful MCP call. A domain read that errored
produces an explicit empty view rather than an estimate, and an intent with no
builder yet says so instead of inventing a surface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from ...mcp_client import FINANCIAL_DOMAIN_TOOL_NAMES
from ...schemas.a2ui import A2UIBundle
from ...schemas.banking_view import FinancialIntent
from ...state import UserProfile
from . import views
from .surface import build_bundle
from .verified_rows import profile_currency


@dataclass(frozen=True, slots=True)
class FinancialPresentation:
    """One validated Finance v2 answer: message, structured data, and surface."""

    intent: FinancialIntent
    message: str
    data: dict[str, Any]
    a2ui: A2UIBundle


#: Intents with a trusted view builder. Anything else is answered as an
#: explicit "supported, but not enough verified data" empty view.
_VIEW_BUILDERS = {
    "financial-summary": views.summary_view,
    "transactions": views.transactions_view,
    "spending-analysis": views.spending_view,
    "recurring-payments": views.recurring_view,
}


def _domain_observations(
    observations: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        observation
        for observation in observations
        if observation.get("name") in FINANCIAL_DOMAIN_TOOL_NAMES
    ]


def build_financial_presentation(
    intent: FinancialIntent,
    observations: Sequence[Mapping[str, Any]],
    profile: UserProfile,
) -> FinancialPresentation:
    """Interpret retained data and construct one validated Finance v2 response."""
    domain_observations = _domain_observations(observations)
    builder = _VIEW_BUILDERS.get(intent)
    if domain_observations and domain_observations[-1].get("is_error") is True:
        view = views.empty_view(
            intent,
            "No pude verificar los datos financieros solicitados; no mostraré cifras estimadas.",
            profile_currency(profile) or "MXN",
        )
        data: dict[str, Any] = {
            "presentation_intent": intent,
            "tool_error": deepcopy(domain_observations[-1].get("data", {})),
        }
        message = view["description"]
    elif builder is not None:
        view, data, message = builder(observations, profile)
    else:
        view = views.empty_view(
            intent,
            "La consulta está soportada, pero faltan datos verificables para construir esta vista.",
            profile_currency(profile) or "MXN",
        )
        data = {"presentation_intent": intent}
        message = view["description"]
    if domain_observations and domain_observations[-1].get("is_error") is not True:
        domain_data = domain_observations[-1].get("data")
        if isinstance(domain_data, Mapping):
            data.setdefault("domain_result", deepcopy(dict(domain_data)))
    return FinancialPresentation(
        intent=intent,
        message=message,
        data=data,
        a2ui=build_bundle(intent, view),
    )
