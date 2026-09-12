"""Fetches real financial and accessibility context through the read-only
Supabase MCP tool server (hackmty2026-mcp), never touching Supabase directly.

Prefers the deployed Horizon endpoint (MCP_SERVER_URL, authenticated with
HORIZON_API_KEY) so the agent talks to the same MCP instance in every
environment. Falls back to spawning the sibling repository's server as a
local stdio subprocess when no remote URL is configured, for offline
development.
"""

from __future__ import annotations

import os

from fastmcp import Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
_DEFAULT_MCP_DIR = os.path.normpath(os.path.join(_REPO_ROOT, "..", "hackmty2026-mcp"))


class UserContextError(RuntimeError):
    """Raised when the MCP server is unreachable or a user has no seeded data."""


def _transport() -> StreamableHttpTransport | StdioTransport:
    server_url = os.environ.get("MCP_SERVER_URL")
    if server_url:
        auth_mode = os.environ.get("MCP_AUTH_MODE", "horizon")
        if auth_mode == "none":
            return StreamableHttpTransport(server_url)
        api_key = os.environ.get("HORIZON_API_KEY")
        if not api_key:
            raise UserContextError(
                "MCP_SERVER_URL is set but HORIZON_API_KEY is missing "
                f"(MCP_AUTH_MODE={auth_mode!r})"
            )
        return StreamableHttpTransport(server_url, auth=api_key)

    mcp_dir = os.environ.get("SUPABASE_MCP_DIR", _DEFAULT_MCP_DIR)
    python = os.environ.get(
        "SUPABASE_MCP_PYTHON", os.path.join(mcp_dir, ".venv", "bin", "python")
    )
    return StdioTransport(command=python, args=["-m", "supabase_mcp.server"], cwd=mcp_dir)


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
    """Share of upcoming recurring expenses the current balance would fail to
    cover, clamped to [0, 1]. 0 means fully covered; 1 means no coverage."""
    if recurring_expenses <= 0:
        return 0.0
    shortfall = 1 - (available_balance / recurring_expenses)
    return max(0.0, min(1.0, shortfall))


async def fetch_user_context(user_id: str) -> dict[str, object]:
    """Fetch accessibility preferences and derived financial metrics for one
    seeded demo user, entirely through MCP `select_rows` calls."""
    try:
        async with Client(_transport()) as client:
            prefs_rows = await _select(client, "accessibility_preferences", "user_id", user_id)
            account_rows = await _select(client, "accounts", "user_id", user_id)
            subscription_rows = await _select(client, "subscriptions", "user_id", user_id)
    except UserContextError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced as a typed, catchable error
        raise UserContextError(f"could not reach the Supabase MCP server: {exc}") from exc

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
