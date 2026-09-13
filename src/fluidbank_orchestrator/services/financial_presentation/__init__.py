"""Deterministic selection and trusted construction of Finance v2 presentations.

* ``intents``       - the finite intent vocabulary: normalize, classify, select.
* ``verified_rows`` - reading only what MCP actually returned.
* ``views``         - one view payload builder per supported intent.
* ``surface``       - the trusted A2UI Finance v2 message sequence.
* ``builder``       - interpreting observations into one validated presentation.

The LLM never reaches any of this: it may select an intent, which is normalized
again here, and everything else is Python the orchestrator owns.
"""

from __future__ import annotations

from .builder import FinancialPresentation, build_financial_presentation
from .intents import (
    normalize_action_intent,
    select_presentation_intent,
)
from .surface import FINANCIAL_VIEW_RESOURCE_URI, FINANCIAL_VIEW_SURFACE_ID

__all__ = [
    "FINANCIAL_VIEW_RESOURCE_URI",
    "FINANCIAL_VIEW_SURFACE_ID",
    "FinancialPresentation",
    "build_financial_presentation",
    "normalize_action_intent",
    "select_presentation_intent",
]
