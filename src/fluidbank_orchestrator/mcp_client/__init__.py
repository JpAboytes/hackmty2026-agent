"""The MCP boundary: everything this orchestrator sends to or reads from MCP.

Modules, in dependency order:

* ``errors``        - the failure vocabulary every other module raises from.
* ``models``        - detached values that cross the boundary.
* ``config``        - endpoint validation and FastMCP client construction.
* ``session``       - one shared connection per turn.
* ``tool_names``    - the financial, discovery and scoped tool-name sets.
* ``trusted_scope`` - the identity boundary; pure, and the security-critical file.
* ``catalog``       - loading and caching what the endpoint advertises.
* ``execution``     - addressing, scoping and running one tool, plus its A2UI.
* ``user_context``  - deriving the signed-in user's profile from scoped reads.

This module re-exports the whole public surface so callers keep importing
``fluidbank_orchestrator.mcp_client``. Patch or extend behaviour in the owning
module rather than here.
"""

from __future__ import annotations

from .catalog import clear_tools_cache, list_remote_tools
from .config import MCPAuthMode, MCPConfig, create_mcp_client, load_mcp_config
from .errors import MCPConfigurationError, TrustedUserScopeError, UserContextError
from .execution import DEFAULT_A2UI_BRIDGE, call_mcp_tool, execute_remote_tool
from .models import MCPToolDefinition, MCPToolExecution, UserContext
from .session import mcp_session
from .tool_names import (
    CALL_TOOL_NAME,
    DISCOVERY_TOOL_NAMES,
    FINANCIAL_DOMAIN_TOOL_NAMES,
    SCOPED_TOOL_NAMES,
    SEARCH_TOOL_NAME,
)
from .trusted_scope import (
    enforce_trusted_user_scope,
    require_current_user_id,
    resolve_tool_call,
)
from .user_context import fetch_user_context

__all__ = [
    "CALL_TOOL_NAME",
    "DEFAULT_A2UI_BRIDGE",
    "DISCOVERY_TOOL_NAMES",
    "FINANCIAL_DOMAIN_TOOL_NAMES",
    "SCOPED_TOOL_NAMES",
    "SEARCH_TOOL_NAME",
    "MCPAuthMode",
    "MCPConfig",
    "MCPConfigurationError",
    "MCPToolDefinition",
    "MCPToolExecution",
    "TrustedUserScopeError",
    "UserContext",
    "UserContextError",
    "call_mcp_tool",
    "clear_tools_cache",
    "create_mcp_client",
    "enforce_trusted_user_scope",
    "execute_remote_tool",
    "fetch_user_context",
    "list_remote_tools",
    "load_mcp_config",
    "mcp_session",
    "require_current_user_id",
    "resolve_tool_call",
]
