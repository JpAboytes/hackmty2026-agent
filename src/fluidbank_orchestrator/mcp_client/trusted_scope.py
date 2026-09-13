"""The identity boundary: nothing scoped reaches MCP with model-chosen identity.

Everything here is pure. Given a tool name, the arguments a model or a
deterministic planner produced, and the UUID the authentication boundary
established, it returns the arguments that may actually be sent. The
authenticated UUID is the only identity source; any ownership field the caller
supplied is discarded or overwritten.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections.abc import Mapping
from copy import deepcopy
from typing import Any
from uuid import UUID

from .errors import TrustedUserScopeError
from .tool_names import (
    CALL_TOOL_NAME,
    FINANCIAL_DOMAIN_TOOL_NAMES,
    SCOPED_TOOL_NAMES,
    SEARCH_TOOL_NAME,
    USER_CONTEXT_TOOL_NAME,
)

#: A model that re-wraps the envelope nests it once, so two levels is enough to
#: unwrap any real call while still bounding a malicious or looping payload.
_MAX_ENVELOPE_DEPTH = 4

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

#: Length below which the shared action secret is treated as absent rather than
#: as a weak key, so an unsigned request is refused by MCP instead of accepted.
_MIN_ACTIONS_SECRET_LENGTH = 32

_SIGNED_ACTION_FIELDS = ("name", "surfaceId", "sourceComponentId", "timestamp", "context")


def require_current_user_id(value: object) -> UUID:
    """Return only a UUID established by the authentication boundary."""
    if not isinstance(value, UUID):
        raise TrustedUserScopeError("a valid authenticated user id is required")
    return value


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


def _business_filters(value: object, *, users_table: bool = False) -> list[object]:
    filters = list(value) if isinstance(value, list) else []
    ownership_columns = _OWNERSHIP_FILTER_COLUMNS | ({"id"} if users_table else set())
    return [
        item
        for item in filters
        if not (isinstance(item, Mapping) and item.get("column") in ownership_columns)
    ]


def _financial_request(scoped: dict[str, Any]) -> dict[str, Any]:
    """Recover the single `request` object every financial tool expects."""
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
    return deepcopy(dict(raw_request))


def _action_proof(scoped: dict[str, Any], canonical_user_id: str) -> str | None:
    """Sign the event plus the verified user id for the MCP write allowlist.

    Server-to-server proof only: it is never part of the A2UI context or of
    anything a model can see.
    """
    secret = os.environ.get("MCP_ACTIONS_SECRET", "")
    if len(secret) < _MIN_ACTIONS_SECRET_LENGTH:
        return None
    signed: dict[str, Any] = {key: scoped.get(key) for key in _SIGNED_ACTION_FIELDS}
    signed["user_id"] = canonical_user_id
    encoded = json.dumps(signed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hmac.new(secret.encode(), encoded, hashlib.sha256).hexdigest()


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
        request = _financial_request(scoped)
        request["scope"] = canonical_scope
        request.pop("user_id", None)
        request.pop("email", None)
        scoped = {"request": request}
    elif tool_name == USER_CONTEXT_TOOL_NAME:
        scoped = {"scope": canonical_scope}
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
            scoped.pop("actionProof", None)
            proof = _action_proof(scoped, canonical_user_id)
            if proof is not None:
                scoped["actionProof"] = proof
    return scoped
