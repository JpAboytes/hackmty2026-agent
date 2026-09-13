"""Deriving one signed-in user's profile from scoped MCP reads.

This is the only banking-domain interpretation inside the MCP boundary: it
turns rows into the accessibility and financial context the graph carries for
the rest of the turn. Every read is scoped by the authenticated UUID, and a row
that cannot be validated fails the whole context rather than becoming an
invented number.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from math import isfinite
from typing import Any, cast
from uuid import UUID

from fastmcp import Client

from ..observability import stage
from ..state import UserProfile
from .errors import MCPConfigurationError, UserContextError
from .execution import call_mcp_tool
from .models import UserContext
from .session import session
from .trusted_scope import require_current_user_id

# Accessible defaults for an account that has not chosen presentation settings
# yet. A new user simply has no row, which is not an error.
_DEFAULT_PREFERENCES: dict[str, str] = {
    "literacy_level": "medium",
    "font_scale": "lg",
    "contrast": "high",
    "hit_target": "large",
}


async def _select(
    client: Client[Any],
    server_identity: str,
    table: str,
    current_user_id: UUID,
) -> list[dict[str, object]]:
    execution = await call_mcp_tool(
        client,
        server_identity,
        "select_rows",
        {
            "schema": "public",
            "table": table,
        },
        current_user_id=current_user_id,
    )
    # Read the wire payload rather than `.data`: `select_rows` is not advertised
    # in `tools/list` under progressive discovery, so the client has no output
    # schema to deserialize it into a typed object.
    structured = execution.result.structured_content
    rows = structured.get("rows") if isinstance(structured, Mapping) else None
    if not isinstance(rows, list):
        raise UserContextError("the MCP selection result was invalid")
    validated: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(key, str) for key in row):
            raise UserContextError("the MCP selection result was invalid")
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
        "literacy_level": _string_value(prefs, "literacy_level")
        or _DEFAULT_PREFERENCES["literacy_level"],
        "font_scale": _string_value(prefs, "font_scale") or _DEFAULT_PREFERENCES["font_scale"],
        "contrast": _string_value(prefs, "contrast") or _DEFAULT_PREFERENCES["contrast"],
        "hit_target": _string_value(prefs, "hit_target") or _DEFAULT_PREFERENCES["hit_target"],
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
            # The four reads are independent, so the profile costs one round
            # trip instead of four. Membership is still enforced: an id that
            # belongs to nobody returns no user row and fails below, and MCP
            # scopes every one of these selects server-side regardless.
            async with stage("mcp.user_context", selects=5, concurrent=True):
                (
                    user_rows,
                    prefs_rows,
                    account_rows,
                    subscription_rows,
                    card_rows,
                ) = await asyncio.gather(
                    _select(client, identity, "users", current_user_id),
                    _select(client, identity, "accessibility_preferences", current_user_id),
                    _select(client, identity, "accounts", current_user_id),
                    _select(client, identity, "subscriptions", current_user_id),
                    # A balance answer shows the plastic beside the totals. The read
                    # joins the concurrent context gather rather than costing the
                    # summary a second turn, and `select_rows` is pinned precisely
                    # because the orchestrator builds this context by name.
                    _select(client, identity, "cards", current_user_id),
                )
            if not user_rows:
                raise UserContextError("no user found for the supplied user id")
    except MCPConfigurationError:
        raise
    except UserContextError:
        raise
    except Exception:  # noqa: BLE001 - expose a stable error without transport secrets
        raise UserContextError("could not reach the remote MCP server") from None

    # An account with no stored preferences reads with the accessible defaults
    # rather than losing its real balances to the generic fallback profile.
    prefs = prefs_rows[0] if prefs_rows else _DEFAULT_PREFERENCES
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
