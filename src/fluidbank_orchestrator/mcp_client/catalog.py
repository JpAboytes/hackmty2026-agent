"""Loading and caching the endpoint's advertised tool collection.

No local filtering happens here: the server already decided what a model may
see. Under progressive discovery that is the search pair rather than every
financial schema, and re-deriving the catalog client-side would put the whole
thing back into the prompt.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from time import monotonic
from typing import Any

from ..observability import event, stage
from .config import load_mcp_config
from .errors import MCPConfigurationError, UserContextError
from .models import MCPToolDefinition
from .session import session
from .tool_names import DISCOVERY_TOOL_NAMES

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _CachedTools:
    tools: tuple[MCPToolDefinition, ...]
    stored_at: float


# Tool schemas change only when the MCP server is redeployed, but loading them
# cost between 0.9s and 4.3s of every single turn in production. They are cached
# per endpoint for a bounded time so a redeploy is still picked up without
# restarting this service.
_TOOLS_CACHE: dict[str, _CachedTools] = {}
_TOOLS_CACHE_LOCK = asyncio.Lock()
_DEFAULT_TOOLS_CACHE_SECONDS = 300.0


def _tools_cache_seconds() -> float:
    """Seconds a loaded tool collection stays usable. Zero disables the cache."""
    raw = os.environ.get("MCP_TOOLS_CACHE_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_TOOLS_CACHE_SECONDS
    try:
        seconds = float(raw)
    except ValueError:
        return _DEFAULT_TOOLS_CACHE_SECONDS
    return seconds if 0 <= seconds <= 86_400 else _DEFAULT_TOOLS_CACHE_SECONDS


def _detached(tools: Sequence[MCPToolDefinition]) -> list[MCPToolDefinition]:
    """Copy out of the cache so a caller can never mutate the retained schemas."""
    return [
        MCPToolDefinition(
            name=tool.name,
            description=tool.description,
            input_schema=deepcopy(tool.input_schema),
            model_visible=tool.model_visible,
        )
        for tool in tools
    ]


def _declared_model_visible(meta: Mapping[str, Any] | None) -> bool:
    """Read `_meta.ui.visibility` exactly as the MCP Apps spec defines it."""
    ui = (meta or {}).get("ui")
    if not isinstance(ui, Mapping):
        return True
    visibility = ui.get("visibility")
    if not isinstance(visibility, list):
        return True
    return "model" in visibility


def _fresh_tools(identity: str, ttl: float) -> list[MCPToolDefinition] | None:
    cached = _TOOLS_CACHE.get(identity)
    if cached is None or ttl <= 0 or (monotonic() - cached.stored_at) >= ttl:
        return None
    return _detached(cached.tools)


def clear_tools_cache() -> None:
    """Drop every retained tool collection. Used by tests and by redeploy hooks."""
    _TOOLS_CACHE.clear()


async def list_remote_tools() -> list[MCPToolDefinition]:
    """Return the endpoint's tool collection, loading it at most once per TTL."""
    identity = load_mcp_config().url
    ttl = _tools_cache_seconds()
    cached = _fresh_tools(identity, ttl)
    if cached is not None:
        event("mcp.tools_cache", outcome="hit", tools=len(cached))
        return cached
    # One loader at a time: concurrent turns that miss together would otherwise
    # each pay the full load. The second check covers the turn that just waited.
    async with _TOOLS_CACHE_LOCK:
        cached = _fresh_tools(identity, ttl)
        if cached is not None:
            event("mcp.tools_cache", outcome="hit_after_wait", tools=len(cached))
            return cached
        event("mcp.tools_cache", outcome="miss")
        definitions = await _load_remote_tools()
        if ttl > 0:
            # The cache keeps its own copies: the collection handed back is the
            # caller's to mutate, and must not be the one the next turn reads.
            _TOOLS_CACHE[identity] = _CachedTools(tuple(_detached(definitions)), monotonic())
    return definitions


async def _load_remote_tools() -> list[MCPToolDefinition]:
    """Load the real read-only tool collection from the configured endpoint."""
    try:
        async with session("list_tools") as (client, _identity):
            async with stage("mcp.list_tools"):
                listed = await client.list_tools()
    except MCPConfigurationError:
        raise
    except Exception:
        raise UserContextError("could not load tools from the remote MCP server") from None

    # No local filtering: the server already decided what a model may see. With
    # progressive discovery active this is the search pair rather than every
    # financial schema, and re-adding a local allowlist here would put the whole
    # catalog back into the prompt.
    definitions: list[MCPToolDefinition] = []
    for tool in listed:
        schema = dict(tool.input_schema)
        try:
            json.dumps(schema, allow_nan=False)
        except (TypeError, ValueError):
            raise UserContextError("the MCP tool schema was invalid") from None
        definitions.append(
            MCPToolDefinition(
                name=tool.name,
                description=tool.description or "",
                input_schema=schema,
                model_visible=_declared_model_visible(tool.meta),
            )
        )
    logger.info(
        "Loaded MCP tools advertised=%d model_visible=%d discovery=%s",
        len(definitions),
        sum(tool.model_visible for tool in definitions),
        DISCOVERY_TOOL_NAMES <= {tool.name for tool in definitions},
    )
    return definitions
