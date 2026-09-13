"""Deterministic pre-graph classification of a plain-text query.

Two queries never reach the model or the graph: a request to prepare an input
form, and a request for database metadata. Both are recognised by phrase, both
call one bounded MCP tool, and both are relayed as-is. Anything else enters the
graph.
"""

from __future__ import annotations

import unicodedata

from ..a2ui_actions.routing import requested_form, requested_form_arguments

__all__ = ["requested_form", "requested_form_arguments", "requests_database_overview"]

_DATABASE_TERMS_EN = ("overview", "objects", "tables", "views")
_DATABASE_TERMS_ES = ("resumen", "vista general", "objetos", "tablas", "vistas", "disponibles")


def _normalized(query: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", query.casefold())
        if not unicodedata.combining(character)
    )


def requests_database_overview(query: str) -> bool:
    """Whether this query asks for the `database_overview` integration proof."""
    normalized = _normalized(query)
    if "database" in normalized:
        return any(term in normalized for term in _DATABASE_TERMS_EN)
    if "base de datos" in normalized:
        return any(term in normalized for term in _DATABASE_TERMS_ES)
    return "tablas permitidas" in normalized or "tablas disponibles" in normalized
