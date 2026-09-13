"""Reading only what MCP actually returned, out of the turn's observations.

Every number in a Finance v2 surface comes through here. An observation that
errored is skipped, a row that cannot be parsed is dropped, and nothing is
inferred: a view with no verified rows renders as empty rather than as an
estimate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import isfinite
from typing import Any

from ...mcp_client.tool_names import USER_CONTEXT_TOOL_NAME
from ...state import UserProfile

_SUPPORTED_CURRENCIES = frozenset({"MXN", "USD"})


#: Which domain tool returns the same entity the fixed user-context read would,
#: and under which payload key. The model chooses the capability, so the reader has
#: to recognise every capability that returns a given entity - not just one.
#: `get_financial_overview` is absent on purpose: it returns per-currency
#: aggregates, not account rows.
_DOMAIN_ROW_SOURCES: dict[tuple[object, str], str] = {
    ("get_transactions", "transactions"): "transactions",
    ("get_accounts", "accounts"): "accounts",
    ("get_bank_statements", "bank_statements"): "statements",
    ("get_beneficiaries", "beneficiaries"): "beneficiaries",
    ("get_transaction_disputes", "transaction_disputes"): "disputes",
    ("get_budget_progress", "budgets"): "budgets",
    ("get_savings_progress", "savings_goals"): "goals",
    ("get_debt_overview", "debts"): "debts",
    ("get_upcoming_payments", "upcoming_payments"): "items",
    ("get_financial_alerts", "financial_alerts"): "alerts",
}


def rows_for_table(observations: Sequence[Mapping[str, Any]], table: str) -> list[dict[str, Any]]:
    """Every verified row for one table, deduplicated by identifier.

    Both shapes count as the same table: the fixed application context read,
    and any domain tool that returns the same entity (see
    ``_DOMAIN_ROW_SOURCES``).
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for observation in observations:
        if observation.get("is_error") is True:
            continue
        arguments = observation.get("arguments")
        data = observation.get("data")
        name = observation.get("name")
        raw_rows: object = None
        if name == USER_CONTEXT_TOOL_NAME:
            if not isinstance(arguments, Mapping) or arguments.get("table") != table:
                continue
            raw_rows = data.get("rows") if isinstance(data, Mapping) else None
        elif _DOMAIN_ROW_SOURCES.get((name, table)) is not None:
            key = _DOMAIN_ROW_SOURCES[(name, table)]
            raw_rows = data.get(key) if isinstance(data, Mapping) else None
        if not isinstance(raw_rows, list):
            continue
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                continue
            row = deepcopy(dict(raw))
            key = str(row.get("id", repr(sorted(row.items()))))
            if key not in seen:
                seen.add(key)
                rows.append(row)
    return rows


def number(value: object) -> float | None:
    """A finite float, or nothing. Never a guess."""
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return None
    try:
        parsed = float(value)
    except (ValueError, OverflowError):
        return None
    return parsed if isfinite(parsed) else None


def profile_currency(profile: UserProfile) -> str | None:
    """The single currency this user's owned balances are in, if there is one."""
    balances = profile.get("owned_balances", {})
    if len(balances) == 1:
        currency = next(iter(balances))
        return currency if currency in _SUPPORTED_CURRENCIES else None
    return None
