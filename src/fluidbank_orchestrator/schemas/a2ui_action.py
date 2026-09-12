"""Strict parser for Expo's temporary action-over-chat representation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from math import isfinite
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, field_validator

from .a2ui import Identifier

ACTION_CONTEXT_MAX_BYTES = 16_384
ACTION_MAX_BYTES = 20_000
ACTION_MAX_DEPTH = 16
_ACTION_INTRO = (
    "Actualiza esta consulta financiera de solo lectura o simulación usando la acción de "
    "interfaz adjunta."
)
_ACTION_PREFIX = f"{_ACTION_INTRO}\nAcción A2UI: "


class A2UIActionPayloadError(ValueError):
    """The query claimed to be an A2UI action but failed strict validation."""


class _A2UIAction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, populate_by_name=True)

    name: Identifier
    surface_id: Identifier = Field(alias="surfaceId")
    source_component_id: Identifier = Field(alias="sourceComponentId")
    timestamp: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=64)]
    context: dict[
        Annotated[str, StringConstraints(strict=True, min_length=1, max_length=128)], JsonValue
    ]

    @field_validator("timestamp")
    @classmethod
    def _timestamp_has_offset(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be RFC 3339") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timestamp must include an offset")
        return value


def _reject_json_constant(_: str) -> None:
    raise ValueError("non-finite JSON number")


def _validate_context_tree(value: Any, *, depth: int = 0) -> None:
    if depth > ACTION_MAX_DEPTH:
        raise A2UIActionPayloadError("A2UI action context is too deeply nested")
    if isinstance(value, str):
        if len(value) > 4_000:
            raise A2UIActionPayloadError("A2UI action context contains an oversized string")
        return
    if value is None or isinstance(value, bool | int):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise A2UIActionPayloadError("A2UI action context contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _validate_context_tree(item, depth=depth + 1)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 128:
                raise A2UIActionPayloadError("A2UI action context has an invalid key")
            _validate_context_tree(item, depth=depth + 1)
        return
    raise A2UIActionPayloadError("A2UI action context must contain JSON values only")


def parse_legacy_a2ui_action(query: str) -> dict[str, Any] | None:
    """Return the five action fields, or None when this is an ordinary chat query."""
    if not query.startswith(_ACTION_INTRO):
        return None
    if not query.startswith(_ACTION_PREFIX):
        raise A2UIActionPayloadError("Malformed A2UI action representation")
    if len(query.encode("utf-8")) > ACTION_MAX_BYTES:
        raise A2UIActionPayloadError("A2UI action representation is too large")

    encoded = query[len(_ACTION_PREFIX) :]
    try:
        raw = json.loads(encoded, parse_constant=_reject_json_constant)
    except (ValueError, TypeError, RecursionError) as exc:
        raise A2UIActionPayloadError("Malformed A2UI action JSON") from exc
    if not isinstance(raw, dict):
        raise A2UIActionPayloadError("A2UI action must be a JSON object")
    try:
        action = _A2UIAction.model_validate(raw, strict=True)
    except Exception as exc:
        raise A2UIActionPayloadError("Invalid A2UI action fields") from exc

    context = action.context
    _validate_context_tree(context)
    if (
        len(json.dumps(context, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        > ACTION_CONTEXT_MAX_BYTES
    ):
        raise A2UIActionPayloadError("A2UI action context is too large")

    # Return detached original values so accepted fields reach MCP unchanged.
    return {
        "name": raw["name"],
        "surfaceId": raw["surfaceId"],
        "sourceComponentId": raw["sourceComponentId"],
        "timestamp": raw["timestamp"],
        "context": deepcopy(raw["context"]),
    }
