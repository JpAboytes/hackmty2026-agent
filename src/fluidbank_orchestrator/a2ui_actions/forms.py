"""The finite vocabulary of A2UI forms the agent may ask MCP to prepare.

The model *selects* a name from this set; it can never introduce one, because
``normalize_form_name`` is the only way a value reaches ``a2ui_form``. Which
form the user needs is a semantic decision and belongs to the model; *which
names exist* is a protocol fact and belongs here.

``.update`` is deliberately absent: MCP derives an update form from the
corresponding ``.load``, so those are the names its ``a2ui_form`` tool accepts.

Prefilled values follow the same split. The model reads the query and may
suggest an amount or a recipient, but ``normalize_form_arguments`` is the only
way one reaches MCP: out-of-range, malformed, or wrongly-targeted suggestions
are dropped rather than corrected. A suggestion is a default in a form the user
still has to confirm, never an instruction to move money.
"""

from __future__ import annotations

import json
from importlib.resources import files
from math import isfinite
from typing import Any

#: Names `a2ui_form` accepts, in the order the model sees them.
ACTION_FORM_NAMES: tuple[str, ...] = (
    "budget.create",
    "budget.load",
    "savings_goal.create",
    "savings_goal.load",
    "transfer.execute",
    "credit_card.pay",
)

#: The only form whose prefill may name a recipient, mirroring the MCP tool.
_RECIPIENT_FORM_NAME = "transfer.execute"

#: Bounds copied from the `a2ui_form` signature, so an out-of-range suggestion
#: is dropped here instead of being refused a round trip later.
_MAX_PREFILL_AMOUNT = 100_000.0
_MAX_RECIPIENT_CHARS = 120


def declared_action_names() -> frozenset[str]:
    """Every action name in the packaged contract, ``.update`` included."""
    contract = json.loads(files(__package__).joinpath("actions.json").read_text())
    return frozenset(action["name"] for action in contract["actions"])


def normalize_form_name(value: object) -> str | None:
    """Return the value only if it is one of the preparable form names."""
    for name in ACTION_FORM_NAMES:
        if value == name:
            return name
    return None


def normalize_form_arguments(
    form_name: str,
    *,
    amount: object = None,
    recipient: object = None,
) -> dict[str, Any]:
    """Bounded prefill for one form, built only from values that validate.

    Anything unusable - a non-finite number, an amount outside the range MCP
    accepts, an empty or overlong name, or a recipient offered for a form that
    has no recipient - is silently omitted, so the worst a bad suggestion can
    do is produce the same unprefilled form the user would have seen anyway.
    """
    arguments: dict[str, Any] = {}
    if isinstance(amount, int | float) and not isinstance(amount, bool):
        parsed = float(amount)
        if isfinite(parsed) and 0 < parsed <= _MAX_PREFILL_AMOUNT:
            arguments["initial_amount"] = parsed
    if form_name == _RECIPIENT_FORM_NAME and isinstance(recipient, str):
        trimmed = " ".join(recipient.split())
        if 0 < len(trimmed) <= _MAX_RECIPIENT_CHARS:
            arguments["initial_recipient"] = trimmed
    return arguments


__all__ = [
    "ACTION_FORM_NAMES",
    "declared_action_names",
    "normalize_form_arguments",
    "normalize_form_name",
]
