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

from ...state import UserProfile

_SUPPORTED_CURRENCIES = frozenset({"MXN", "USD"})


def rows_for_table(observations: Sequence[Mapping[str, Any]], table: str) -> list[dict[str, Any]]:
    """Every verified row for one table, deduplicated by identifier.

    Both shapes count as the same table: the fixed application context and the
    domain tool that returns the same entity (``get_transactions``).
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for observation in observations:
        if observation.get("is_error") is True:
            continue
        arguments = observation.get("arguments")
        data = observation.get("data")
        raw_rows: object = None
        if observation.get("name") == "get_user_context":
            if not isinstance(arguments, Mapping) or arguments.get("table") != table:
                continue
            raw_rows = data.get("rows") if isinstance(data, Mapping) else None
        elif observation.get("name") == "get_transactions" and table == "transactions":
            raw_rows = data.get("transactions") if isinstance(data, Mapping) else None
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
