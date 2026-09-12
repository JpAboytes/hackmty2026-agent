"""Resolve MCP-owned A2UI resources into self-contained Expo responses."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from functools import partial
from typing import Any
from urllib.parse import urlsplit

from fastmcp import Client
from fastmcp.client.client import CallToolResult
from mcp.types import EmbeddedResource, TextResourceContents

from fluidbank_orchestrator.schemas.a2ui import (
    A2UI_MIME_TYPE,
    MAX_RESOURCE_BYTES,
    A2UIBundle,
    A2UIValidationError,
    validate_complete_sequence,
    validate_dynamic_updates,
    validate_static_template,
)

_URI_PART = re.compile(r"^[A-Za-z0-9._~-]+$")
logger = logging.getLogger(__name__)


class A2UIBridgeError(ValueError):
    """A safe, classified A2UI presentation failure."""

    def __init__(self, code: str) -> None:
        super().__init__("The visual response could not be validated.")
        self.code = code


def _parse_resource_uri(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 2_048:
        raise A2UIBridgeError("invalid_resource_uri")
    parsed = urlsplit(value)
    parts = [parsed.netloc, *parsed.path.split("/")[1:]]
    if (
        parsed.scheme != "a2ui"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(
            not part or part in {".", ".."} or _URI_PART.fullmatch(part) is None for part in parts
        )
    ):
        raise A2UIBridgeError("invalid_resource_uri")
    return value


def _reject_json_constant(_: str) -> None:
    raise ValueError("non-finite JSON number")


def _parse_message_array(text: str, *, error_code: str) -> list[dict[str, Any]]:
    if len(text.encode("utf-8")) > MAX_RESOURCE_BYTES:
        raise A2UIBridgeError(error_code)
    try:
        value = json.loads(text, parse_constant=_reject_json_constant)
    except (ValueError, TypeError, RecursionError) as exc:
        raise A2UIBridgeError(error_code) from exc
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, dict) for item in value)
    ):
        raise A2UIBridgeError(error_code)
    return value


class A2UIBridge:
    """Validate and cache static MCP A2UI templates with per-key single-flight reads."""

    def __init__(self, *, max_cache_entries: int = 64) -> None:
        if max_cache_entries < 1 or max_cache_entries > 256:
            raise ValueError("max_cache_entries must be between 1 and 256")
        self._max_cache_entries = max_cache_entries
        self._cache: OrderedDict[tuple[str, str], tuple[str, tuple[dict[str, Any], ...]]] = (
            OrderedDict()
        )
        self._inflight: dict[
            tuple[str, str], asyncio.Task[tuple[str, tuple[dict[str, Any], ...]]]
        ] = {}

    async def build_bundle(
        self,
        mcp_client: Client[Any],
        tool_result: CallToolResult,
        *,
        server_identity: str,
    ) -> A2UIBundle | None:
        """Build a complete static-plus-dynamic bundle from one MCP tool result."""
        meta = tool_result.meta
        if meta is None or "ui" not in meta:
            return None
        ui = meta.get("ui")
        if not isinstance(ui, Mapping):
            raise A2UIBridgeError("invalid_ui_metadata")
        resource_uri = _parse_resource_uri(ui.get("resourceUri"))
        logger.info("MCP A2UI resource discovered uri=%s", resource_uri)
        if ui.get("mimeType") != A2UI_MIME_TYPE:
            raise A2UIBridgeError("invalid_ui_mime_type")
        if not server_identity or len(server_identity) > 2_048:
            raise A2UIBridgeError("invalid_server_identity")

        surface_id, template_messages = await self._get_template(
            mcp_client, server_identity, resource_uri
        )
        logger.info("MCP A2UI resource loaded uri=%s", resource_uri)
        dynamic_messages = self._extract_dynamic_messages(tool_result, surface_id)
        complete = [*template_messages, *dynamic_messages]
        try:
            validate_complete_sequence(complete, surface_id)
        except A2UIValidationError as exc:
            raise A2UIBridgeError("invalid_a2ui_sequence") from exc
        logger.info("MCP A2UI validated surface=%s", surface_id)
        logger.info("MCP A2UI ready for client emission surface=%s", surface_id)
        return A2UIBundle(resource_uri=resource_uri, messages=deepcopy(complete))

    async def _get_template(
        self, mcp_client: Client[Any], server_identity: str, resource_uri: str
    ) -> tuple[str, list[dict[str, Any]]]:
        key = (server_identity, resource_uri)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            surface_id, messages = cached
            return surface_id, deepcopy(list(messages))

        task = self._inflight.get(key)
        if task is None:
            if len(self._inflight) >= self._max_cache_entries:
                raise A2UIBridgeError("resource_resolution_busy")
            task = asyncio.create_task(self._read_template(mcp_client, key))
            self._inflight[key] = task
            task.add_done_callback(partial(self._finish_read, key))
        try:
            surface_id, messages = await asyncio.shield(task)
        except A2UIBridgeError:
            raise
        except Exception as exc:
            raise A2UIBridgeError("resource_read_failed") from exc
        return surface_id, deepcopy(list(messages))

    def _finish_read(
        self,
        key: tuple[str, str],
        task: asyncio.Task[tuple[str, tuple[dict[str, Any], ...]]],
    ) -> None:
        if self._inflight.get(key) is task:
            self._inflight.pop(key, None)
        if task.cancelled():
            return
        # Retrieve an exception when every waiter was cancelled, without logging payloads.
        task.exception()

    async def _read_template(
        self, mcp_client: Client[Any], key: tuple[str, str]
    ) -> tuple[str, tuple[dict[str, Any], ...]]:
        _server_identity, resource_uri = key
        try:
            contents = await mcp_client.read_resource(resource_uri)
        except Exception as exc:
            raise A2UIBridgeError("resource_read_failed") from exc
        if len(contents) != 1:
            raise A2UIBridgeError("invalid_template_resource")
        resource = contents[0]
        if not isinstance(resource, TextResourceContents) or resource.mime_type != A2UI_MIME_TYPE:
            raise A2UIBridgeError("invalid_template_resource")
        raw_messages = _parse_message_array(resource.text, error_code="invalid_template_json")
        try:
            surface_id, messages = validate_static_template(raw_messages)
        except A2UIValidationError as exc:
            raise A2UIBridgeError("invalid_template_messages") from exc

        cached_messages = tuple(deepcopy(messages))
        self._cache[key] = (surface_id, cached_messages)
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_cache_entries:
            self._cache.popitem(last=False)
        return surface_id, cached_messages

    @staticmethod
    def _extract_dynamic_messages(
        tool_result: CallToolResult, surface_id: str
    ) -> list[dict[str, Any]]:
        raw_messages: list[dict[str, Any]] = []
        found = False
        total_bytes = 0
        for content in tool_result.content:
            if not isinstance(content, EmbeddedResource):
                continue
            resource = content.resource
            if resource.mime_type != A2UI_MIME_TYPE:
                continue
            found = True
            if not isinstance(resource, TextResourceContents):
                raise A2UIBridgeError("invalid_embedded_resource")
            total_bytes += len(resource.text.encode("utf-8"))
            if total_bytes > MAX_RESOURCE_BYTES:
                raise A2UIBridgeError("invalid_embedded_json")
            raw_messages.extend(
                _parse_message_array(resource.text, error_code="invalid_embedded_json")
            )
        if not found:
            raise A2UIBridgeError("missing_dynamic_messages")
        try:
            return validate_dynamic_updates(raw_messages, expected_surface_id=surface_id)
        except A2UIValidationError as exc:
            raise A2UIBridgeError("invalid_dynamic_messages") from exc


async def build_a2ui_bundle(
    mcp_client: Client[Any],
    tool_result: CallToolResult,
    *,
    server_identity: str,
    bridge: A2UIBridge | None = None,
) -> A2UIBundle | None:
    """Convenience entry point for callers that do not own a bridge service."""
    return await (bridge or A2UIBridge()).build_bundle(
        mcp_client, tool_result, server_identity=server_identity
    )
