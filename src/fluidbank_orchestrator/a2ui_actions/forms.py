"""The finite vocabulary of A2UI forms the agent may ask MCP to prepare.

The model *selects* a name from this set; it can never introduce one, because
``normalize_form_name`` is the only way a value reaches ``a2ui_form``. Which
form the user needs is a semantic decision and belongs to the model; *which
names exist* is a protocol fact and belongs here.

``.update`` is deliberately absent: MCP derives an update form from the
corresponding ``.load``, so those are the four names its ``a2ui_form`` tool
accepts.
"""

from __future__ import annotations

import json
from importlib.resources import files

#: Names `a2ui_form` accepts, in the order the model sees them.
ACTION_FORM_NAMES: tuple[str, ...] = (
    "budget.create",
    "budget.load",
    "savings_goal.create",
    "savings_goal.load",
)


def declared_action_names() -> frozenset[str]:
    """Every action name in the packaged contract, ``.update`` included."""
    contract = json.loads(files(__package__).joinpath("actions.json").read_text())
    return frozenset(action["name"] for action in contract["actions"])


def normalize_form_name(value: object) -> str | None:
    """Return the value only if it is one of the four preparable form names."""
    for name in ACTION_FORM_NAMES:
        if value == name:
            return name
    return None


__all__ = ["ACTION_FORM_NAMES", "declared_action_names", "normalize_form_name"]
