"""Remote FastMCP client for Supabase-backed user context.

All transport and Horizon authentication details live in this module. The
Horizon credential is read only by the server-side orchestrator and is passed
directly to FastMCP; it is never added to graph state or model-visible data.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal, cast
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

from fastmcp import Client
from fastmcp.client.client import CallToolResult

from .observability import preview, stage
from .schemas.a2ui import A2UIBundle
from .services.a2ui_bridge import A2UIBridge, A2UIBridgeError
from .state import UserProfile

logger = logging.getLogger(__name__)

MCPAuthMode = Literal["horizon", "none"]


class MCPConfigurationError(ValueError):
    """Raised when the remote MCP connection is not configured safely."""


class UserContextError(RuntimeError):
    """Raised when the MCP server is unreachable or a user has no seeded data."""


class TrustedUserScopeError(UserContextError):
    """A scoped MCP operation has no valid server-authenticated UUID."""


@dataclass(frozen=True, slots=True)
class MCPToolExecution:
    """One MCP result plus its optional, independently validated presentation."""

    result: CallToolResult
    a2ui: A2UIBundle | None
    presentation_error: bool = False


@dataclass(frozen=True, slots=True)
class UserContext:
    """One user's derived profile plus the scoped rows it was built from.

    The rows travel with the profile so a later domain read does not fetch the
    same scoped table a second time in the same turn.
    """

    profile: UserProfile
    rows: dict[str, list[dict[str, object]]]


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


_TURN_SESSION: ContextVar[tuple[Client[Any], str] | None] = ContextVar(
    "fluidbank_mcp_session", default=None
)


@asynccontextmanager
async def mcp_session() -> AsyncIterator[None]:
    """Hold one MCP session open for everything inside this block.

    A session handshake against the remote endpoint costs roughly as much as a
    query, and a turn used to pay it three times: once to list tools, once for
    user context, and once per tool call. Helpers below join this session when
    one is active and fall back to opening their own when it is not, so tests
    and one-off calls keep working unchanged.
    """
    config = load_mcp_config()
    async with AsyncExitStack() as stack:
        async with stage("mcp.connect", purpose="turn", shared=True):
            client = await stack.enter_async_context(create_mcp_client(config))
        token = _TURN_SESSION.set((client, config.url))
        try:
            yield
        finally:
            _TURN_SESSION.reset(token)


@asynccontextmanager
async def _session(purpose: str) -> AsyncIterator[tuple[Client[Any], str]]:
    """Yield the turn's shared session, or a private one when none is active."""
    joined = _TURN_SESSION.get()
    if joined is not None:
        yield joined
        return
    config = load_mcp_config()
    async with AsyncExitStack() as stack:
        async with stage("mcp.connect", purpose=purpose, shared=False):
            client = await stack.enter_async_context(create_mcp_client(config))
        yield client, config.url


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

SCOPED_TOOL_NAMES = frozenset(
    {
        "select_rows",
        "visualize_allowed_data",
        "a2ui_action",
    }
)

_OWNERSHIP_FILTER_COLUMNS = frozenset(
    {
        "user_id",
        "customer_id",
        "account_id",
        "from_account_id",
        "to_account_id",
        "owner_id",
        "persona_id",
    }
)


def require_current_user_id(value: object) -> UUID:
    """Return only a UUID established by the authentication boundary."""
    if not isinstance(value, UUID):
        raise TrustedUserScopeError("a valid authenticated user id is required")
    return value


def _business_filters(value: object, *, users_table: bool = False) -> list[object]:
    filters = list(value) if isinstance(value, list) else []
    ownership_columns = _OWNERSHIP_FILTER_COLUMNS | ({"id"} if users_table else set())
    return [
        item
        for item in filters
        if not (isinstance(item, Mapping) and item.get("column") in ownership_columns)
    ]


def enforce_trusted_user_scope(
    tool_name: str,
    arguments: Mapping[str, Any] | None,
    current_user_id: UUID | None,
) -> dict[str, Any] | None:
    """Detach tool arguments and overwrite every server-owned identity field.

    Scoped tools fail closed unless the caller supplies the UUID produced by the
    authentication boundary. Unscoped tools retain their original schema shape.
    """
    if tool_name not in SCOPED_TOOL_NAMES:
        return deepcopy(dict(arguments)) if arguments is not None else None

    canonical_user_id = str(require_current_user_id(current_user_id))
    scoped = deepcopy(dict(arguments)) if arguments is not None else {}
    canonical_scope = {"user_id": canonical_user_id}

    if tool_name == "select_rows":
        scoped["scope"] = canonical_scope
        scoped["filters"] = _business_filters(
            scoped.get("filters"),
            users_table=scoped.get("schema") == "public" and scoped.get("table") == "users",
        )
    elif tool_name == "visualize_allowed_data":
        raw_request = scoped.get("request")
        request = deepcopy(dict(raw_request)) if isinstance(raw_request, Mapping) else {}
        request["scope"] = canonical_scope
        source = request.get("source")
        request["filters"] = _business_filters(
            request.get("filters"),
            users_table=(
                isinstance(source, Mapping)
                and source.get("schema") == "public"
                and source.get("table") == "users"
            ),
        )
        scoped["request"] = request
    else:
        scoped["trustedScope"] = canonical_scope
    return scoped


async def list_remote_tools() -> list[MCPToolDefinition]:
    """Load the real read-only tool collection from the configured endpoint."""
    try:
        async with _session("list_tools") as (client, _identity):
            async with stage("mcp.list_tools"):
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
    preview(f"mcp.call.{name}.result", result.structured_content)
    async with stage("mcp.a2ui_bridge", name=name) as step:
        try:
            a2ui = await bridge.build_bundle(client, result, server_identity=server_identity)
        except A2UIBridgeError as exc:
            step.set(outcome="rejected", code=exc.code)
            logger.warning("MCP A2UI presentation rejected code=%s", exc.code)
            return MCPToolExecution(result=result, a2ui=None, presentation_error=True)
        except Exception as exc:  # noqa: BLE001 - retain the safe MCP fallback on bridge defects
            step.set(outcome="failed", reason=type(exc).__name__)
            logger.warning("MCP A2UI presentation failed (%s)", type(exc).__name__)
            return MCPToolExecution(result=result, a2ui=None, presentation_error=True)
        step.set(
            outcome="bundled" if a2ui is not None else "no_presentation",
            messages=len(a2ui.messages) if a2ui is not None else None,
        )
    return MCPToolExecution(result=result, a2ui=a2ui)


async def execute_remote_tool(
    name: str,
    arguments: Mapping[str, Any] | None = None,
    *,
    current_user_id: UUID | None = None,
    bridge: A2UIBridge = DEFAULT_A2UI_BRIDGE,
) -> MCPToolExecution:
    """Execute a tool over the configured remote/local MCP connection."""
    if name in SCOPED_TOOL_NAMES:
        require_current_user_id(current_user_id)
    try:
        async with _session(name) as (client, identity):
            return await call_mcp_tool(
                client,
                identity,
                name,
                arguments,
                current_user_id=current_user_id,
                bridge=bridge,
            )
    except MCPConfigurationError:
        raise
    except Exception:  # noqa: BLE001 - expose no transport or credential details
        raise UserContextError("could not reach the remote MCP server") from None
    raise UserContextError("the remote MCP session closed without a result")


async def _select(
    client: Client[Any],
    server_identity: str,
    table: str,
    current_user_id: UUID,
) -> list[dict[str, object]]:
    execution = await call_mcp_tool(
        client,
        server_identity,
        "select_rows",
        {
            "schema": "public",
            "table": table,
        },
        current_user_id=current_user_id,
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


def _owned_balances(account_rows: list[dict[str, object]]) -> dict[str, float]:
    """Aggregate owned cash by currency without treating credit as money."""
    balances: dict[str, float] = {}
    for row in account_rows:
        account_type = _string_value(row, "account_type")
        if account_type == "credit":
            continue
        if account_type not in {"checking", "savings"}:
            raise UserContextError("the MCP user context was invalid")
        currency = _string_value(row, "currency")
        if currency not in {"MXN", "USD"}:
            raise UserContextError("the MCP user context was invalid")
        balances[currency] = balances.get(currency, 0.0) + _float_value(row, "available_balance")
    return balances


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


# Accessible defaults for an account that has not chosen presentation settings
# yet. A new user simply has no row, which is not an error.
_DEFAULT_PREFERENCES: dict[str, str] = {
    "literacy_level": "medium",
    "font_scale": "lg",
    "contrast": "high",
    "hit_target": "large",
}


async def fetch_user_context(current_user_id: UUID) -> UserContext:
    """Fetch one signed-in user's context through mandatory MCP scope."""
    current_user_id = require_current_user_id(current_user_id)
    try:
        async with _session("user_context") as (client, identity):
            # The four reads are independent, so the profile costs one round
            # trip instead of four. Membership is still enforced: an id that
            # belongs to nobody returns no user row and fails below, and MCP
            # scopes every one of these selects server-side regardless.
            async with stage("mcp.user_context", selects=4, concurrent=True):
                user_rows, prefs_rows, account_rows, subscription_rows = await asyncio.gather(
                    _select(client, identity, "users", current_user_id),
                    _select(client, identity, "accessibility_preferences", current_user_id),
                    _select(client, identity, "accounts", current_user_id),
                    _select(client, identity, "subscriptions", current_user_id),
                )
            if not user_rows:
                raise UserContextError("no user found for the supplied user id")
    except MCPConfigurationError:
        raise
    except UserContextError:
        raise
    except Exception:  # noqa: BLE001 - expose a stable error without transport secrets
        raise UserContextError("could not reach the remote MCP server") from None

    # An account with no stored preferences reads with the accessible defaults
    # rather than losing its real balances to the generic fallback profile.
    prefs = prefs_rows[0] if prefs_rows else _DEFAULT_PREFERENCES
    owned_balances = _owned_balances(account_rows)
    available_balance = next(iter(owned_balances.values())) if len(owned_balances) == 1 else None
    recurring_expenses = sum(
        _float_value(row, "amount")
        for row in subscription_rows
        if _string_value(row, "status") == "active"
    )

    profile: UserProfile = {
        "literacy_level": _string_value(prefs, "literacy_level")
        or _DEFAULT_PREFERENCES["literacy_level"],
        "font_scale": _string_value(prefs, "font_scale") or _DEFAULT_PREFERENCES["font_scale"],
        "contrast": _string_value(prefs, "contrast") or _DEFAULT_PREFERENCES["contrast"],
        "hit_target": _string_value(prefs, "hit_target") or _DEFAULT_PREFERENCES["hit_target"],
        "available_balance": available_balance,
        "recurring_expenses": recurring_expenses,
        "overdraft_risk": (
            _overdraft_risk(available_balance, recurring_expenses)
            if available_balance is not None
            else None
        ),
        "owned_balances": owned_balances,
    }
    # Only the domain tables a later financial read would ask for again. The
    # user and preference rows stay out: nothing re-reads them, and they carry
    # identity fields that have no business travelling through graph state.
    return UserContext(
        profile=profile,
        rows={"accounts": account_rows, "subscriptions": subscription_rows},
    )
