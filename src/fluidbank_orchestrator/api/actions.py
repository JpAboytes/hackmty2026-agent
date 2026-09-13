"""The A2UI action transport at the HTTP boundary.

Actions never enter the model. They arrive as a structured envelope (or the
temporarily preserved legacy chat encoding), are forwarded to MCP's allowlist
with the verified UUID carried separately as ``trustedScope``, and only a
result MCP itself normalized - and that still names this authenticated user -
is allowed to re-enter the graph.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from ..schemas.a2ui_action import A2UIActionPayloadError, parse_legacy_a2ui_action
from ..schemas.banking_view import FinancialIntent
from ..schemas.chat import ChatRequest
from ..services.financial_presentation import normalize_action_intent

#: The one action name that re-enters the graph. Everything else is a bounded
#: backend operation whose MCP result is relayed straight back.
FINANCIAL_VIEW_ACTION = "request_financial_view"


def action_payload(request: ChatRequest) -> tuple[dict[str, Any] | None, str | None]:
    """Return ``(payload, invalid_reason)`` for whichever action form was sent."""
    if request.action is not None:
        try:
            return request.action.as_payload(), None
        except A2UIActionPayloadError:
            return None, "structured_payload"
    if request.query is not None:
        try:
            return parse_legacy_a2ui_action(request.query), None
        except A2UIActionPayloadError:
            return None, "legacy_payload"
    return None, None


def trusted_financial_intent(structured: object, current_user_id: UUID) -> FinancialIntent | None:
    """The intent MCP normalized, or nothing.

    Every condition has to hold: MCP reported success, it echoed the trusted
    scope this API sent, that scope names this authenticated user, and the
    intent is one of the shared contract's. A model-supplied or client-supplied
    intent never reaches this function.
    """
    if not isinstance(structured, dict) or structured.get("ok") is not True:
        return None
    trusted_scope = structured.get("trustedScope")
    if not isinstance(trusted_scope, dict) or trusted_scope.get("user_id") != str(current_user_id):
        return None
    request_data = structured.get("request")
    if not isinstance(request_data, dict):
        return None
    return normalize_action_intent(request_data.get("intent"))
