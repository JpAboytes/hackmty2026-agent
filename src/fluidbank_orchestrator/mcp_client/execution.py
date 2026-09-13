"""Executing one MCP tool: address it, scope it, run it, bridge its A2UI.

Two things happen here that callers must not have to think about. Arguments
pass through the trusted-scope boundary before anything is sent, and a result
carrying `_meta.ui` is handed to the A2UI bridge so a validated presentation
travels back alongside the raw result.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from uuid import UUID

from fastmcp import Client

from ..observability import preview, stage
from ..services.a2ui_bridge import A2UIBridge, A2UIBridgeError
from .catalog import list_remote_tools
from .errors import MCPConfigurationError, UserContextError
from .models import MCPToolExecution
from .session import session
from .tool_names import CALL_TOOL_NAME, DISCOVERY_TOOL_NAMES, SCOPED_TOOL_NAMES
from .trusted_scope import enforce_trusted_user_scope, require_current_user_id, resolve_tool_call

logger = logging.getLogger(__name__)

DEFAULT_A2UI_BRIDGE = A2UIBridge()


async def call_mcp_tool(
    client: Client[Any],
    server_identity: str,
    name: str,
    arguments: Mapping[str, Any] | None = None,
    *,
    current_user_id: UUID | None = None,
    bridge: A2UIBridge = DEFAULT_A2UI_BRIDGE,
) -> MCPToolExecution:
    """Call any MCP tool and process optional A2UI metadata through one bridge."""
    trusted_arguments = enforce_trusted_user_scope(name, arguments, current_user_id)
    async with stage("mcp.call", name=name) as step:
        result = await client.call_tool(
            name,
            trusted_arguments,
            raise_on_error=False,
        )
        step.set(is_error=bool(result.is_error), contents=len(result.content))
    mcp_ui_owned = isinstance(result.meta, Mapping) and "ui" in result.meta
    preview(f"mcp.call.{name}.result", result.structured_content)
    async with stage("mcp.a2ui_bridge", name=name) as step:
        try:
            a2ui = await bridge.build_bundle(client, result, server_identity=server_identity)
        except A2UIBridgeError as exc:
            step.set(outcome="rejected", code=exc.code)
            logger.warning("MCP A2UI presentation rejected code=%s", exc.code)
            return MCPToolExecution(
                result=result,
                a2ui=None,
                mcp_ui_owned=mcp_ui_owned,
                presentation_error=True,
            )
        except Exception as exc:  # noqa: BLE001 - retain the safe MCP fallback on bridge defects
            step.set(outcome="failed", reason=type(exc).__name__)
            logger.warning("MCP A2UI presentation failed (%s)", type(exc).__name__)
            return MCPToolExecution(
                result=result,
                a2ui=None,
                mcp_ui_owned=mcp_ui_owned,
                presentation_error=True,
            )
        step.set(
            outcome="bundled" if a2ui is not None else "no_presentation",
            messages=len(a2ui.messages) if a2ui is not None else None,
        )
    return MCPToolExecution(result=result, a2ui=a2ui, mcp_ui_owned=mcp_ui_owned)


async def _wire_call(
    name: str, arguments: Mapping[str, Any] | None
) -> tuple[str, Mapping[str, Any] | None]:
    """Address a tool the way the active endpoint can actually reach it.

    A hosted deployment fronts the server with a proxy that resolves
    `tools/call` against the advertised catalog, so a tool hidden by
    progressive discovery answers `Unknown tool` when addressed by name and has
    to be reached through the `call_tool` proxy instead. A tool the server does
    advertise is called directly, which is how the pinned orchestrator-driven
    tools keep working.

    The decision follows the live catalog rather than a hardcoded list, so
    pinning or unpinning a tool server-side needs no change here. If the
    catalog cannot be read, the direct call is kept - the pre-discovery
    behaviour - rather than inventing an envelope.
    """
    if name in DISCOVERY_TOOL_NAMES:
        return name, arguments
    try:
        advertised = {tool.name for tool in await list_remote_tools()}
    except (MCPConfigurationError, UserContextError):
        return name, arguments
    if name in advertised:
        return name, arguments
    return CALL_TOOL_NAME, {"name": name, "arguments": dict(arguments or {})}


async def execute_remote_tool(
    name: str,
    arguments: Mapping[str, Any] | None = None,
    *,
    current_user_id: UUID | None = None,
    bridge: A2UIBridge = DEFAULT_A2UI_BRIDGE,
) -> MCPToolExecution:
    """Execute a tool over the configured remote/local MCP connection."""
    effective_name, _ = resolve_tool_call(name, arguments)
    if effective_name in SCOPED_TOOL_NAMES:
        require_current_user_id(current_user_id)
    wire_name, wire_arguments = await _wire_call(name, arguments)
    try:
        async with session(name) as (client, identity):
            return await call_mcp_tool(
                client,
                identity,
                wire_name,
                wire_arguments,
                current_user_id=current_user_id,
                bridge=bridge,
            )
    except MCPConfigurationError:
        raise
    except Exception:  # noqa: BLE001 - expose no transport or credential details
        raise UserContextError("could not reach the remote MCP server") from None
    raise UserContextError("the remote MCP session closed without a result")
