"""Failure modes of the MCP boundary, stated without transport detail.

Every module in this package raises from this small set so a caller never has
to know whether a failure came from configuration, transport, or the server's
own answer. The messages are deliberately generic: they reach logs and, through
the graph, user-visible copy.
"""

from __future__ import annotations


class MCPConfigurationError(ValueError):
    """Raised when the remote MCP connection is not configured safely."""


class UserContextError(RuntimeError):
    """Raised when the MCP server is unreachable or a user has no seeded data."""


class TrustedUserScopeError(UserContextError):
    """A scoped MCP operation has no valid server-authenticated UUID."""
