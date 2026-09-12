"""Remote FastMCP client for Supabase-backed user context.

All transport and Horizon authentication details live in this module. The
Horizon credential is read only by the server-side orchestrator and is passed
directly to FastMCP; it is never added to graph state or model-visible data.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from fastmcp import Client

MCPAuthMode = Literal["horizon", "none"]


class MCPConfigurationError(ValueError):
    """Raised when the remote MCP connection is not configured safely."""


class UserContextError(RuntimeError):
    """Raised when the MCP server is unreachable or a user has no seeded data."""


@dataclass(frozen=True, slots=True)
class MCPConfig:
    """Validated remote MCP configuration with a redacted representation."""

    url: str
    auth_mode: MCPAuthMode
    _horizon_api_key: str | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        key = "<redacted>" if self._horizon_api_key is not None else None
        return f"MCPConfig(url={self.url!r}, auth_mode={self.auth_mode!r}, horizon_api_key={key!r})"


def _normalize_mcp_url(raw_url: str | None) -> str:
    if raw_url is None or not raw_url.strip():
        raise MCPConfigurationError("MCP_SERVER_URL is required")

    parsed = urlsplit(raw_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise MCPConfigurationError("MCP_SERVER_URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise MCPConfigurationError("MCP_SERVER_URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise MCPConfigurationError("MCP_SERVER_URL must not contain a query or fragment")

    path = parsed.path.rstrip("/")
    if path in {"", "/mcp"}:
        path = "/mcp"
    else:
        raise MCPConfigurationError("MCP_SERVER_URL must use the /mcp endpoint path")

    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def load_mcp_config(environment: Mapping[str, str] | None = None) -> MCPConfig:
    """Load and validate the MCP connection without exposing its credential."""
    env = os.environ if environment is None else environment
    auth_mode = env.get("MCP_AUTH_MODE", "horizon").strip().lower()
    if auth_mode not in {"horizon", "none"}:
        raise MCPConfigurationError("MCP_AUTH_MODE must be either 'horizon' or 'none'")

    url = _normalize_mcp_url(env.get("MCP_SERVER_URL"))
    if auth_mode == "none":
        return MCPConfig(url=url, auth_mode="none")

    token = env.get("HORIZON_API_KEY", "").strip()
    if not token:
        raise MCPConfigurationError("HORIZON_API_KEY is required when MCP_AUTH_MODE is 'horizon'")
    if not token.startswith("fmcp_"):
        raise MCPConfigurationError("HORIZON_API_KEY must begin with 'fmcp_'")

    return MCPConfig(url=url, auth_mode="horizon", _horizon_api_key=token)


def create_mcp_client(config: MCPConfig | None = None) -> Client:
    """Create the configured FastMCP client.

    FastMCP accepts the raw Horizon token and adds the HTTP ``Bearer`` prefix.
    """
    resolved = config or load_mcp_config()
    if resolved.auth_mode == "horizon":
        return Client(resolved.url, auth=resolved._horizon_api_key)
    return Client(resolved.url)


async def _select(client: Client, table: str, column: str, value: str) -> list[dict[str, object]]:
    result = await client.call_tool(
        "select_rows",
        {
            "schema": "public",
            "table": table,
            "filters": [{"column": column, "operator": "eq", "value": value}],
        },
    )
    return result.data.rows


def _overdraft_risk(available_balance: float, recurring_expenses: float) -> float:
    """Return the uncovered share of recurring expenses, clamped to [0, 1]."""
    if recurring_expenses <= 0:
        return 0.0
    shortfall = 1 - (available_balance / recurring_expenses)
    return max(0.0, min(1.0, shortfall))


async def fetch_user_context(email: str) -> dict[str, object]:
    """Fetch context for the seeded demo user matching this login email,
    entirely through remote MCP tool calls.

    The email is the client-supplied identifier: it's what the mobile app's
    Supabase Auth session already carries, and it's a column that already
    exists on `public.users` - no new schema or client-side user id needed.
    """
    try:
        async with create_mcp_client() as client:
            user_rows = await _select(client, "users", "email", email)
            if not user_rows:
                raise UserContextError(f"no seeded user found for email {email}")
            user_id = user_rows[0]["id"]
            prefs_rows = await _select(client, "accessibility_preferences", "user_id", user_id)
            account_rows = await _select(client, "accounts", "user_id", user_id)
            subscription_rows = await _select(client, "subscriptions", "user_id", user_id)
    except MCPConfigurationError:
        raise
    except UserContextError:
        raise
    except Exception:  # noqa: BLE001 - expose a stable error without transport secrets
        raise UserContextError("could not reach the remote MCP server") from None

    if not prefs_rows:
        raise UserContextError(f"no accessibility preferences seeded for user {user_id}")

    prefs = prefs_rows[0]
    available_balance = sum(float(row["available_balance"]) for row in account_rows)
    recurring_expenses = sum(
        float(row["amount"]) for row in subscription_rows if row["status"] == "active"
    )

    return {
        "literacy_level": prefs["literacy_level"],
        "font_scale": prefs["font_scale"],
        "contrast": prefs["contrast"],
        "hit_target": prefs["hit_target"],
        "available_balance": available_balance,
        "recurring_expenses": recurring_expenses,
        "overdraft_risk": _overdraft_risk(available_balance, recurring_expenses),
    }
