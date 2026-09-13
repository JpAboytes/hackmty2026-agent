"""Deriving one signed-in user's profile from scoped MCP reads.

This is the only banking-domain interpretation inside the MCP boundary: it
turns rows into the accessibility and financial context the graph carries for
the rest of the turn. Every read is scoped by the authenticated UUID, and a row
that cannot be validated fails the whole context rather than becoming an
invented number.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import cast
from uuid import UUID

from ..observability import stage
from ..state import UserProfile
from .errors import MCPConfigurationError, UserContextError
from .execution import call_mcp_tool
from .models import UserContext
from .session import session
from .tool_names import USER_CONTEXT_TOOL_NAME
from .trusted_scope import require_current_user_id

# Accessible defaults for an account that has not chosen presentation settings
# yet. A new user simply has no row, which is not an error.
_DEFAULT_PREFERENCES: dict[str, str] = {
    "literacy_level": "medium",
    "font_scale": "lg",
    "contrast": "high",
    "color_vision_mode": "none",
    "hit_target": "large",
}


def _context_rows(structured: Mapping[str, object], key: str) -> list[dict[str, object]]:
    rows = structured.get(key)
    if not isinstance(rows, list):
        raise UserContextError("the MCP user context result was invalid")
    validated: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(key, str) for key in row):
            raise UserContextError("the MCP user context result was invalid")
        validated.append(cast("dict[str, object]", dict(row)))
    return validated


def _overdraft_risk(available_balance: float, recurring_expenses: float) -> float:
    """Return the uncovered share of recurring expenses, clamped to [0, 1]."""
    if recurring_expenses <= 0:
        return 0.0
    shortfall = 1 - (available_balance / recurring_expenses)
    return max(0.0, min(1.0, shortfall))


def _owned_balances(account_rows: list[dict[str, object]]) -> dict[str, float]:
    """Aggregate owned cash by currency without treating credit as money."""
    balances: dict[str, float] = {}
    for row in account_rows:
        account_type = _string_value(row, "account_type")
        if account_type == "credit":
            continue
        if account_type not in {"checking", "savings"}:
            raise UserContextError("the MCP user context was invalid")
        currency = _string_value(row, "currency")
        if currency not in {"MXN", "USD"}:
            raise UserContextError("the MCP user context was invalid")
        balances[currency] = balances.get(currency, 0.0) + _float_value(row, "available_balance")
    return balances


def _preference(prefs: Mapping[str, object], key: str) -> str:
    """One accessibility preference, or its default.

    A preference column is optional on purpose: a row written before the column
    existed, or a deployment that has not applied the migration yet, must fall
    back to the default rather than invalidate the whole user context. Financial
    columns keep using `_string_value`, which still fails closed.
    """
    value = prefs.get(key)
    return value if isinstance(value, str) and value else _DEFAULT_PREFERENCES[key]


def _string_value(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise UserContextError("the MCP user context was invalid")
    return value


def _float_value(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise UserContextError("the MCP user context was invalid")
    try:
        parsed = float(value)
    except (OverflowError, ValueError):
        raise UserContextError("the MCP user context was invalid") from None
    if not isfinite(parsed):
        raise UserContextError("the MCP user context was invalid")
    return parsed


def _profile(
    prefs: Mapping[str, object],
    account_rows: list[dict[str, object]],
    subscription_rows: list[dict[str, object]],
) -> UserProfile:
    owned_balances = _owned_balances(account_rows)
    available_balance = next(iter(owned_balances.values())) if len(owned_balances) == 1 else None
    recurring_expenses = sum(
        _float_value(row, "amount")
        for row in subscription_rows
        if _string_value(row, "status") == "active"
    )
    return {
        "literacy_level": _preference(prefs, "literacy_level"),
        "font_scale": _preference(prefs, "font_scale"),
        "contrast": _preference(prefs, "contrast"),
        # Stored and seeded in `accessibility_preferences`, so it belongs in the
        # profile; dropping it here silently discarded the whole preference.
        "color_vision_mode": _preference(prefs, "color_vision_mode"),
        "hit_target": _preference(prefs, "hit_target"),
        "available_balance": available_balance,
        "recurring_expenses": recurring_expenses,
        "overdraft_risk": (
            _overdraft_risk(available_balance, recurring_expenses)
            if available_balance is not None
            else None
        ),
        "owned_balances": owned_balances,
    }


async def fetch_user_context(current_user_id: UUID) -> UserContext:
    """Fetch one signed-in user's context through mandatory MCP scope."""
    current_user_id = require_current_user_id(current_user_id)
    try:
        async with session("user_context") as (client, identity):
            # MCP owns the fixed table set and performs its independent reads
            # concurrently. The agent receives no caller-selected query surface.
            async with stage("mcp.user_context", calls=1):
                execution = await call_mcp_tool(
                    client,
                    identity,
                    USER_CONTEXT_TOOL_NAME,
                    {},
                    current_user_id=current_user_id,
                )
            structured = execution.result.structured_content
            if not isinstance(structured, Mapping) or structured.get("ok") is not True:
                raise UserContextError("the MCP user context result was invalid")
            if structured.get("user_found") is not True:
                raise UserContextError("no user found for the supplied user id")
            raw_preferences = structured.get("preferences")
            if raw_preferences is not None and not isinstance(raw_preferences, Mapping):
                raise UserContextError("the MCP user context result was invalid")
            prefs = dict(raw_preferences) if isinstance(raw_preferences, Mapping) else None
            account_rows = _context_rows(structured, "accounts")
            subscription_rows = _context_rows(structured, "subscriptions")
            card_rows = _context_rows(structured, "cards")
    except MCPConfigurationError:
        raise
    except UserContextError:
        raise
    except Exception:  # noqa: BLE001 - expose a stable error without transport secrets
        raise UserContextError("could not reach the remote MCP server") from None

    # An account with no stored preferences reads with the accessible defaults
    # rather than losing its real balances to the generic fallback profile.
    prefs = prefs or _DEFAULT_PREFERENCES
    # Only the domain tables a later financial read would ask for again. The
    # user and preference rows stay out: nothing re-reads them, and they carry
    # identity fields that have no business travelling through graph state.
    return UserContext(
        profile=_profile(prefs, account_rows, subscription_rows),
        rows={
            "accounts": account_rows,
            "subscriptions": subscription_rows,
            "cards": card_rows,
        },
    )
