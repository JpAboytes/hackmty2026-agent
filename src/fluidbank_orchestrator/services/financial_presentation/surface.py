"""The trusted builder: the only place Finance v2 A2UI is constructed.

Gemini never emits, edits or sees any of this. The component tree is fixed
Python, the surface and catalog ids are constants, and the view payload is
validated against the shared Finance v2 contract before it is bound into
``updateDataModel``. The finished sequence is validated once more as a complete
v0.9.1 message sequence, so an invalid surface fails here rather than on the
client.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...schemas.a2ui import A2UI_FINANCE_V2_CATALOG, A2UIBundle, validate_complete_sequence
from ...schemas.banking_view import FinancialIntent, validate_banking_view

FINANCIAL_VIEW_SURFACE_ID = "financial-view"
FINANCIAL_VIEW_RESOURCE_URI = "a2ui://finance/view"
FINANCIAL_ACTION_NAME = "request_financial_view"
FINANCIAL_ACTION_COMPONENT_ID = "request_financial_view_button"


def _follow_up(intent: FinancialIntent) -> tuple[str, FinancialIntent]:
    """The one bounded next step offered with every surface."""
    if intent == "financial-summary":
        return "Ver gastos del último mes", "transactions"
    return "Volver al panorama financiero", "financial-summary"


def _components() -> list[dict[str, Any]]:
    return [
        {
            "id": "root",
            "component": "Column",
            "children": ["banking_view", FINANCIAL_ACTION_COMPONENT_ID],
        },
        {"id": "banking_view", "component": "BankingView", "view": {"path": "/view"}},
        {
            "id": "request_financial_view_label",
            "component": "Text",
            "text": {"path": "/actionLabel"},
        },
        {
            "id": FINANCIAL_ACTION_COMPONENT_ID,
            "component": "Button",
            "child": "request_financial_view_label",
            "variant": "primary",
            "action": {
                "event": {
                    "name": FINANCIAL_ACTION_NAME,
                    "context": {"intent": {"path": "/requestIntent"}},
                }
            },
        },
    ]


def build_bundle(intent: FinancialIntent, view: Mapping[str, Any]) -> A2UIBundle:
    """Validate the view and wrap it in a complete v0.9.1 message sequence."""
    validated_view = validate_banking_view(view)
    action_label, request_intent = _follow_up(intent)
    messages = [
        {
            "version": "v0.9.1",
            "createSurface": {
                "surfaceId": FINANCIAL_VIEW_SURFACE_ID,
                "catalogId": A2UI_FINANCE_V2_CATALOG,
            },
        },
        {
            "version": "v0.9.1",
            "updateComponents": {
                "surfaceId": FINANCIAL_VIEW_SURFACE_ID,
                "components": _components(),
            },
        },
        {
            "version": "v0.9.1",
            "updateDataModel": {
                "surfaceId": FINANCIAL_VIEW_SURFACE_ID,
                "path": "/",
                "value": {
                    "view": validated_view,
                    "actionLabel": action_label,
                    "requestIntent": request_intent,
                },
            },
        },
    ]
    validate_complete_sequence(messages, FINANCIAL_VIEW_SURFACE_ID)
    return A2UIBundle(resource_uri=FINANCIAL_VIEW_RESOURCE_URI, messages=messages)
