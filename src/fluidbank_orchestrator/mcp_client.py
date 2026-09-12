"""Remote FastMCP client for Supabase-backed user context.

All transport and Horizon authentication details live in this module. The
Horizon credential is read only by the server-side orchestrator and is passed
directly to FastMCP; it is never added to graph state or model-visible data.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal, cast
from urllib.parse import urlsplit, urlunsplit

from fastmcp import Client
from fastmcp.client.client import CallToolResult

from .schemas.a2ui import A2UIBundle
from .services.a2ui_bridge import A2UIBridge, A2UIBridgeError
from .state import UserProfile

logger = logging.getLogger(__name__)

MCPAuthMode = Literal["horizon", "none"]


class MCPConfigurationError(ValueError):
    """Raised when the remote MCP connection is not configured safely."""


class UserContextError(RuntimeError):
    """Raised when the MCP server is unreachable or a user has no seeded data."""


@dataclass(frozen=True, slots=True)
class MCPToolExecution:
    """One MCP result plus its optional, independently validated presentation."""

    result: CallToolResult
    a2ui: A2UIBundle | None
    presentation_error: bool = False


@dataclass(frozen=True, slots=True)
class MCPToolDefinition:
    """Detached, JSON-safe tool definition loaded from the active MCP endpoint."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
        }


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


def create_mcp_client(config: MCPConfig | None = None) -> Client[Any]:
    """Create the configured FastMCP client.

    FastMCP accepts the raw Horizon token and adds the HTTP ``Bearer`` prefix.
    """
    resolved = config or load_mcp_config()
    if resolved.auth_mode == "horizon":
        return Client(resolved.url, auth=resolved._horizon_api_key)
    return Client(resolved.url)


DEFAULT_A2UI_BRIDGE = A2UIBridge()

MODEL_TOOL_NAMES = frozenset(
    {
        "health_check",
        "list_allowed_tables",
        "describe_table",
        "select_rows",
        "database_overview",
        "visualize_allowed_data",
    }
)


async def list_remote_tools() -> list[MCPToolDefinition]:
    """Load the real read-only tool collection from the configured endpoint."""
    config = load_mcp_config()
    try:
        async with create_mcp_client(config) as client:
            listed = await client.list_tools()
    except MCPConfigurationError:
        raise
    except Exception:
        raise UserContextError("could not load tools from the remote MCP server") from None

    definitions: list[MCPToolDefinition] = []
    for tool in listed:
        if tool.name not in MODEL_TOOL_NAMES:
            continue
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
            )
        )
    logger.info(
        "Loaded MCP model tools count=%d visualization_available=%s",
        len(definitions),
        any(tool.name == "visualize_allowed_data" for tool in definitions),
    )
    return definitions


async def call_mcp_tool(
    client: Client[Any],
    server_identity: str,
    name: str,
    arguments: Mapping[str, Any] | None = None,
    *,
    bridge: A2UIBridge = DEFAULT_A2UI_BRIDGE,
) -> MCPToolExecution:
    """Call any MCP tool and process optional A2UI metadata through one bridge."""
    logger.info("MCP tool selected name=%s", name)
    result = await client.call_tool(
        name,
        dict(arguments) if arguments is not None else None,
        raise_on_error=False,
    )
    logger.info("MCP tool completed name=%s is_error=%s", name, result.is_error)
    try:
        a2ui = await bridge.build_bundle(client, result, server_identity=server_identity)
    except A2UIBridgeError as exc:
        logger.warning("MCP A2UI presentation rejected code=%s", exc.code)
        return MCPToolExecution(result=result, a2ui=None, presentation_error=True)
    except Exception as exc:  # noqa: BLE001 - retain the safe MCP fallback on bridge defects
        logger.warning("MCP A2UI presentation failed (%s)", type(exc).__name__)
        return MCPToolExecution(result=result, a2ui=None, presentation_error=True)
    return MCPToolExecution(result=result, a2ui=a2ui)


async def execute_remote_tool(
    name: str,
    arguments: Mapping[str, Any] | None = None,
    *,
    bridge: A2UIBridge = DEFAULT_A2UI_BRIDGE,
) -> MCPToolExecution:
    """Execute a tool over the configured remote/local MCP connection."""
    config = load_mcp_config()
    try:
        async with create_mcp_client(config) as client:
            return await call_mcp_tool(
                client,
                config.url,
                name,
                arguments,
                bridge=bridge,
            )
    except MCPConfigurationError:
        raise
    except Exception:  # noqa: BLE001 - expose no transport or credential details
        raise UserContextError("could not reach the remote MCP server") from None


async def _select(
    client: Client[Any],
    server_identity: str,
    table: str,
    current_user_id: str,
) -> list[dict[str, object]]:
    execution = await call_mcp_tool(
        client,
        server_identity,
        "select_rows",
        {
            "schema": "public",
            "table": table,
            "scope": {"user_id": current_user_id},
        },
    )
    rows = getattr(execution.result.data, "rows", None)
    if not isinstance(rows, list):
        raise UserContextError("the MCP selection result was invalid")
    validated: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(key, str) for key in row):
            raise UserContextError("the MCP selection result was invalid")
        validated.append(cast("dict[str, object]", dict(row)))
    return validated


def _overdraft_risk(available_balance: float, recurring_expenses: float) -> float:
    """Return the uncovered share of recurring expenses, clamped to [0, 1]."""
    if recurring_expenses <= 0:
        return 0.0
    shortfall = 1 - (available_balance / recurring_expenses)
    return max(0.0, min(1.0, shortfall))


def _string_value(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise UserContextError("the MCP user context was invalid")
    return value


def _float_value(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise UserContextError("the MCP user context was invalid")
    try:
        parsed = float(value)
    except (OverflowError, ValueError):
        raise UserContextError("the MCP user context was invalid") from None
    if not isfinite(parsed):
        raise UserContextError("the MCP user context was invalid")
    return parsed


async def fetch_user_context(current_user_id: str) -> UserProfile:
    """Fetch one configured demo user's context through mandatory MCP scope."""
    try:
        config = load_mcp_config()
        async with create_mcp_client(config) as client:
            user_rows = await _select(client, config.url, "users", current_user_id)
            if not user_rows:
                raise UserContextError("no seeded user found for the supplied user id")
            prefs_rows = await _select(
                client, config.url, "accessibility_preferences", current_user_id
            )
            account_rows = await _select(client, config.url, "accounts", current_user_id)
            subscription_rows = await _select(client, config.url, "subscriptions", current_user_id)
    except MCPConfigurationError:
        raise
    except UserContextError:
        raise
    except Exception:  # noqa: BLE001 - expose a stable error without transport secrets
        raise UserContextError("could not reach the remote MCP server") from None

    if not prefs_rows:
        raise UserContextError("no accessibility preferences seeded for the selected user")

    prefs = prefs_rows[0]
    available_balance = sum(_float_value(row, "available_balance") for row in account_rows)
    recurring_expenses = sum(
        _float_value(row, "amount")
        for row in subscription_rows
        if _string_value(row, "status") == "active"
    )

    return {
        "literacy_level": _string_value(prefs, "literacy_level"),
        "font_scale": _string_value(prefs, "font_scale"),
        "contrast": _string_value(prefs, "contrast"),
        "hit_target": _string_value(prefs, "hit_target"),
        "available_balance": available_balance,
        "recurring_expenses": recurring_expenses,
        "overdraft_risk": _overdraft_risk(available_balance, recurring_expenses),
    }
