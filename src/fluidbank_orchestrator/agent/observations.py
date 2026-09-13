"""The turn's ledger of verified MCP results.

An *observation* is one recorded tool result: the capability that produced it,
the arguments it actually ran with, whether it errored, its structured data and
its text. Provenance is what this ledger exists for - the presentation layer
only ever reads data that arrived through it, so a domain answer cannot be
published without a successful MCP call behind it.

Everything here is a pure read over ``GraphState``.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ..mcp_client import SEARCH_TOOL_NAME, MCPToolExecution
from ..state import GraphState, ToolObservation


def normalized_query(value: str) -> str:
    """Fold case and strip accents so bilingual matching is stable."""
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )


def text_from_execution(execution: MCPToolExecution) -> str:
    for content in execution.result.content:
        text = getattr(content, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return "La herramienta terminó sin una respuesta de texto."


def retained_observations(state: GraphState) -> list[ToolObservation]:
    """Every verified row set this turn holds, prefetched context included.

    Context rows come first so a later, narrower domain read of the same table
    wins on the deduplicated identifiers.
    """
    return [*state.get("context_observations", []), *state.get("tool_observations", [])]


def context_observations(rows: Mapping[str, list[dict[str, object]]]) -> list[ToolObservation]:
    """Record the profile's scoped reads in the shape a domain read produces."""
    return [
        {
            "name": "select_rows",
            "arguments": {"schema": "public", "table": table},
            "is_error": False,
            "data": {"ok": True, "rows": deepcopy(table_rows)},
            "text": f"Filas de {table} leídas con el contexto del usuario.",
        }
        for table, table_rows in rows.items()
    ]


def has_tool_observation(state: GraphState, tool_name: str) -> bool:
    """Whether the capability was attempted, successfully or not."""
    return any(
        observation.get("name") == tool_name for observation in state.get("tool_observations", [])
    )


def has_table_observation(state: GraphState, table: str) -> bool:
    """Whether a successful retained context read verified this table."""
    return any(
        observation.get("name") == "select_rows"
        and observation.get("is_error") is not True
        and observation.get("arguments", {}).get("table") == table
        for observation in retained_observations(state)
    )


def already_searched(state: GraphState, arguments: Mapping[str, Any]) -> bool:
    """Whether this exact discovery query already produced tool definitions.

    Re-searching the same intent burns a model turn and returns the same
    definitions. A search that found nothing is allowed to run again with a
    different phrasing, which is the only case worth retrying.
    """
    query = arguments.get("query")
    if not isinstance(query, str):
        return False
    wanted = normalized_query(query)
    return any(
        observation.get("name") == SEARCH_TOOL_NAME
        and observation.get("is_error") is not True
        and observation.get("data", {}).get("result")
        and isinstance(observation.get("arguments"), Mapping)
        and normalized_query(str(observation["arguments"].get("query", ""))) == wanted
        for observation in state.get("tool_observations", [])
    )
