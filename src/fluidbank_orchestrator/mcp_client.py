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
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from time import monotonic
from typing import Any, Literal, cast
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

from fastmcp import Client
from fastmcp.client.client import CallToolResult

from .observability import event, preview, stage
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

FINANCIAL_DOMAIN_TOOL_NAMES = frozenset(
    {
        "get_financial_overview",
        "get_accounts",
        "get_transactions",
        "analyze_spending",
        "get_cash_flow",
        "get_budget_progress",
        "get_savings_progress",
        "get_debt_overview",
        "get_upcoming_payments",
        "get_financial_alerts",
        "get_bank_statements",
        "get_payment_activity",
        "get_beneficiaries",
        "get_transaction_disputes",
        "compare_debt_scenarios",
    }
)

# The MCP server replaced its `tools/list` with progressive discovery, so the
# model receives these two synthetic tools and finds everything else through
# them. This is a discovery contract, not an allowlist: it says nothing about
# what may execute, only how a model reaches a capability.
SEARCH_TOOL_NAME = "search_tools"
CALL_TOOL_NAME = "call_tool"
DISCOVERY_TOOL_NAMES = frozenset({SEARCH_TOOL_NAME, CALL_TOOL_NAME})

# Security boundary, kept deliberately separate from discovery: every tool here
# reads user-owned rows, so its identity fields are overwritten with the UUID
# the authentication boundary produced. A tool reaching MCP without passing
# through this set would be trusting model-supplied identity.
SCOPED_TOOL_NAMES = frozenset(
    {
        "select_rows",
        "visualize_allowed_data",
        "a2ui_action",
        "a2ui_form",
        *FINANCIAL_DOMAIN_TOOL_NAMES,
    }
)


#: A model that re-wraps the envelope nests it once, so two levels is enough to
#: unwrap any real call while still bounding a malicious or looping payload.
_MAX_ENVELOPE_DEPTH = 4


def resolve_tool_call(
    name: str, arguments: Mapping[str, Any] | None
) -> tuple[str, Mapping[str, Any] | None]:
    """Unwrap the discovery proxy to the domain tool a call actually reaches.

    A model that discovered `get_transactions` calls it as
    `call_tool(name="get_transactions", arguments={...})`. Scoping, logging and
    routing all care about the inner tool, never about the envelope.

    Unwrapping repeats because Gemini re-wraps the envelope in roughly one call
    in five, emitting `call_tool(name="call_tool", arguments=<the real call>)`.
    That nesting has exactly one valid reading, so it is resolved rather than
    bounced off the server and retried a turn later. Pointing the proxy at
    `search_tools` has no such reading and is left for the server to refuse.
    """
    for _ in range(_MAX_ENVELOPE_DEPTH):
        if name != CALL_TOOL_NAME or arguments is None:
            return name, arguments
        target = arguments.get("name")
        if not isinstance(target, str):
            # No name at this level. If the whole call sits one level down,
            # descend; otherwise there is no way to know what was meant.
            nested = arguments.get("arguments")
            if not isinstance(nested, Mapping):
                return name, arguments
            arguments = nested
            continue
        if target == SEARCH_TOOL_NAME:
            return name, arguments
        inner = arguments.get("arguments")
        if not isinstance(inner, Mapping):
            # `{"name": X, <the call's own fields>}` with no `arguments` level.
            # The siblings are the call; without this they are silently dropped
            # and the tool runs on defaults instead of the user's filters.
            siblings = {k: v for k, v in arguments.items() if k not in {"name", "arguments"}}
            inner = siblings or None
        name, arguments = target, inner if isinstance(inner, Mapping) else None
    return name, arguments


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

    A `call_tool` envelope is scoped on the tool it targets and then rebuilt, so
    discovery cannot be used to smuggle a scoped call past this function.
    """
    if tool_name == CALL_TOOL_NAME:
        target, inner = resolve_tool_call(tool_name, arguments)
        if target == tool_name:
            return deepcopy(dict(arguments)) if arguments is not None else None
        scoped_inner = enforce_trusted_user_scope(target, inner, current_user_id)
        envelope = deepcopy(dict(arguments)) if arguments is not None else {}
        envelope["name"] = target
        envelope["arguments"] = scoped_inner if scoped_inner is not None else {}
        return envelope

    if tool_name not in SCOPED_TOOL_NAMES:
        return deepcopy(dict(arguments)) if arguments is not None else None

    canonical_user_id = str(require_current_user_id(current_user_id))
    scoped = deepcopy(dict(arguments)) if arguments is not None else {}
    canonical_scope = {"user_id": canonical_user_id}

    if tool_name in FINANCIAL_DOMAIN_TOOL_NAMES:
        raw_request = scoped.get("request")
        if not isinstance(raw_request, Mapping):
            # The model dropped the single `request` wrapper and sent its fields
            # at the top level. Every financial tool takes exactly one `request`
            # object and rejects unknown fields, so promoting the stray keys is
            # the only reading that can validate; the alternative is losing the
            # user's filters.
            raw_request = {key: value for key, value in scoped.items() if key != "request"}
        while isinstance(raw_request.get("request"), Mapping):
            # ...or it applied the wrapper twice. No request model has a
            # `request` field, so a nested one is always the repeated wrapper.
            raw_request = raw_request["request"]
        request = deepcopy(dict(raw_request))
        request["scope"] = canonical_scope
        request.pop("user_id", None)
        request.pop("email", None)
        scoped = {"request": request}
    elif tool_name == "select_rows":
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
        if tool_name == "a2ui_action":
            # Server-to-server proof is never part of the A2UI context or model input.
            import hashlib
            import hmac

            scoped.pop("actionProof", None)
            secret = os.environ.get("MCP_ACTIONS_SECRET", "")
            if len(secret) >= 32:
                signed = {
                    key: scoped.get(key)
                    for key in ("name", "surfaceId", "sourceComponentId", "timestamp", "context")
                }
                signed["user_id"] = canonical_user_id
                encoded = json.dumps(
                    signed, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode()
                scoped["actionProof"] = hmac.new(
                    secret.encode(), encoded, hashlib.sha256
                ).hexdigest()
    return scoped


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
        )
        for tool in tools
    ]


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
        async with _session("list_tools") as (client, _identity):
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
            )
        )
    logger.info(
        "Loaded MCP model tools count=%d discovery=%s",
        len(definitions),
        DISCOVERY_TOOL_NAMES <= {tool.name for tool in definitions},
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
    effective_name, _ = resolve_tool_call(name, arguments)
    if effective_name in SCOPED_TOOL_NAMES:
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
    # Read the wire payload rather than `.data`: `select_rows` is not advertised
    # in `tools/list` under progressive discovery, so the client has no output
    # schema to deserialize it into a typed object.
    structured = execution.result.structured_content
    rows = structured.get("rows") if isinstance(structured, Mapping) else None
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
