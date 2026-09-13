"""Offline contract tests for MCP A2UI resource resolution and caching."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import Any

import pytest
from fastmcp.client.client import CallToolResult
from mcp.types import EmbeddedResource, TextContent, TextResourceContents

from fluidbank_orchestrator.api.responses import response_from_tool
from fluidbank_orchestrator.mcp_client import MCPToolExecution, call_mcp_tool
from fluidbank_orchestrator.schemas.a2ui import A2UI_BASIC_CATALOG, A2UI_MIME_TYPE
from fluidbank_orchestrator.services.a2ui_bridge import A2UIBridge, A2UIBridgeError

RESOURCE_URI = "a2ui://database/overview"
SURFACE_ID = "database-overview"
SERVER_ID = "https://example.fastmcp.app/mcp"

TEMPLATE_MESSAGES = [
    {
        "version": "v0.9.1",
        "createSurface": {"surfaceId": SURFACE_ID, "catalogId": A2UI_BASIC_CATALOG},
    },
    {
        "version": "v0.9.1",
        "updateComponents": {
            "surfaceId": SURFACE_ID,
            "components": [
                {"id": "root", "component": "Card", "child": "overview_column"},
                {
                    "id": "overview_column",
                    "component": "Column",
                    "children": ["title", "objects", "refresh_button"],
                },
                {"id": "title", "component": "Text", "text": {"path": "/title"}},
                {"id": "objects", "component": "Text", "text": {"path": "/objectsText"}},
                {"id": "refresh_label", "component": "Text", "text": "Refresh"},
                {
                    "id": "refresh_button",
                    "component": "Button",
                    "child": "refresh_label",
                    "action": {
                        "event": {
                            "name": "refresh_database_overview",
                            "context": {"limit": {"path": "/limit"}},
                        }
                    },
                },
            ],
        },
    },
]
DYNAMIC_MESSAGES = [
    {
        "version": "v0.9.1",
        "updateDataModel": {
            "surfaceId": SURFACE_ID,
            "path": "/",
            "value": {
                "title": "Database overview",
                "objectsText": "public.customers (table)",
                "limit": 50,
            },
        },
    }
]


def _tool_result(
    *,
    meta: dict[str, Any] | None = None,
    embedded_text: str | None = None,
    embedded_mime: str = A2UI_MIME_TYPE,
) -> CallToolResult:
    resolved_meta = (
        {"ui": {"resourceUri": RESOURCE_URI, "mimeType": A2UI_MIME_TYPE}} if meta is None else meta
    )
    content: list[Any] = [TextContent(text="Database overview loaded.")]
    if embedded_text is not None or embedded_mime:
        content.append(
            EmbeddedResource(
                resource=TextResourceContents(
                    uri=f"{RESOURCE_URI}/data",
                    mime_type=embedded_mime,
                    text=embedded_text or json.dumps(DYNAMIC_MESSAGES),
                )
            )
        )
    return CallToolResult(
        content=content,
        structured_content={"ok": True, "object_count": 1},
        meta=resolved_meta,
        data=None,
    )


class FakeClient:
    def __init__(self, template: Any = None, *, failures: int = 0, delay: float = 0) -> None:
        self.template = deepcopy(TEMPLATE_MESSAGES) if template is None else template
        self.failures = failures
        self.delay = delay
        self.read_count = 0
        self.call_count = 0
        self.result = _tool_result()

    async def read_resource(self, uri: str) -> list[TextResourceContents]:
        assert uri == RESOURCE_URI
        self.read_count += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.read_count <= self.failures:
            raise RuntimeError("sensitive upstream detail")
        text = self.template if isinstance(self.template, str) else json.dumps(self.template)
        return [TextResourceContents(uri=RESOURCE_URI, mime_type=A2UI_MIME_TYPE, text=text)]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        *,
        raise_on_error: bool = True,
    ) -> CallToolResult:
        assert raise_on_error is False
        self.call_count += 1
        return self.result


@pytest.mark.asyncio
async def test_valid_result_becomes_exact_expo_contract_with_object_messages() -> None:
    bridge = A2UIBridge()
    client = FakeClient()
    bundle = await bridge.build_bundle(client, client.result, server_identity=SERVER_ID)  # type: ignore[arg-type]

    assert bundle is not None
    response = response_from_tool(MCPToolExecution(result=client.result, a2ui=bundle))
    assert response.model_dump() == {
        "message": "Database overview loaded.",
        "data": {"ok": True, "object_count": 1},
        "a2ui": {
            "resource_uri": RESOURCE_URI,
            "messages": [*TEMPLATE_MESSAGES, *DYNAMIC_MESSAGES],
        },
    }
    assert all(isinstance(message, dict) for message in bundle.messages)
    assert [next(key for key in message if key != "version") for message in bundle.messages] == [
        "createSurface",
        "updateComponents",
        "updateDataModel",
    ]


@pytest.mark.asyncio
async def test_cache_reads_once_returns_complete_detached_bundle_each_time() -> None:
    bridge = A2UIBridge()
    client = FakeClient()
    first = await bridge.build_bundle(client, client.result, server_identity=SERVER_ID)  # type: ignore[arg-type]
    assert first is not None
    first.messages[0]["createSurface"]["surfaceId"] = "mutated"

    second = await bridge.build_bundle(client, client.result, server_identity=SERVER_ID)  # type: ignore[arg-type]
    assert second is not None
    assert client.read_count == 1
    assert second.messages == [*TEMPLATE_MESSAGES, *DYNAMIC_MESSAGES]


@pytest.mark.asyncio
async def test_cache_never_reuses_dynamic_tool_data() -> None:
    bridge = A2UIBridge()
    client = FakeClient()
    first = await bridge.build_bundle(client, client.result, server_identity=SERVER_ID)  # type: ignore[arg-type]
    updated = deepcopy(DYNAMIC_MESSAGES)
    updated[0]["updateDataModel"]["value"]["title"] = "Fresh tool data"
    second = await bridge.build_bundle(
        client,
        _tool_result(embedded_text=json.dumps(updated)),
        server_identity=SERVER_ID,
    )  # type: ignore[arg-type]

    assert first is not None and second is not None
    assert client.read_count == 1
    assert first.messages[-1]["updateDataModel"]["value"]["title"] == "Database overview"
    assert second.messages[-1]["updateDataModel"]["value"]["title"] == "Fresh tool data"


@pytest.mark.asyncio
async def test_missing_ui_metadata_returns_none_without_resource_read() -> None:
    bridge = A2UIBridge()
    client = FakeClient()
    result = _tool_result(meta={})
    assert await bridge.build_bundle(client, result, server_identity=SERVER_ID) is None  # type: ignore[arg-type]
    assert client.read_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("meta", "expected_code"),
    [
        (
            {"ui": {"resourceUri": RESOURCE_URI, "mimeType": "application/json"}},
            "invalid_ui_mime_type",
        ),
        (
            {
                "ui": {
                    "resourceUri": "https://example.com/template.json",
                    "mimeType": A2UI_MIME_TYPE,
                }
            },
            "invalid_resource_uri",
        ),
    ],
)
async def test_invalid_metadata_is_rejected_without_resource_read(
    meta: dict[str, Any], expected_code: str
) -> None:
    bridge = A2UIBridge()
    client = FakeClient()
    with pytest.raises(A2UIBridgeError) as caught:
        await bridge.build_bundle(client, _tool_result(meta=meta), server_identity=SERVER_ID)  # type: ignore[arg-type]
    assert caught.value.code == expected_code
    assert client.read_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutate",
    [
        lambda messages: messages[0].update(version="v1.0"),
        lambda messages: messages[0]["createSurface"].update(catalogId="https://bad.invalid"),
        lambda messages: messages[1]["updateComponents"]["components"][0].update(
            component="WebView"
        ),
    ],
)
async def test_wrong_version_catalog_or_component_is_rejected(mutate: Any) -> None:
    template = deepcopy(TEMPLATE_MESSAGES)
    mutate(template)
    bridge = A2UIBridge()
    client = FakeClient(template)
    with pytest.raises(A2UIBridgeError, match="visual response"):
        await bridge.build_bundle(client, client.result, server_identity=SERVER_ID)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_malformed_template_json_is_rejected() -> None:
    bridge = A2UIBridge()
    client = FakeClient("not-json")
    with pytest.raises(A2UIBridgeError) as caught:
        await bridge.build_bundle(client, client.result, server_identity=SERVER_ID)  # type: ignore[arg-type]
    assert caught.value.code == "invalid_template_json"


@pytest.mark.asyncio
async def test_malformed_embedded_json_is_rejected() -> None:
    bridge = A2UIBridge()
    client = FakeClient()
    with pytest.raises(A2UIBridgeError) as caught:
        await bridge.build_bundle(
            client,
            _tool_result(embedded_text="not-json"),
            server_identity=SERVER_ID,
        )  # type: ignore[arg-type]
    assert caught.value.code == "invalid_embedded_json"


@pytest.mark.asyncio
async def test_surface_mismatch_is_rejected() -> None:
    dynamic = deepcopy(DYNAMIC_MESSAGES)
    dynamic[0]["updateDataModel"]["surfaceId"] = "different-surface"
    bridge = A2UIBridge()
    client = FakeClient()
    with pytest.raises(A2UIBridgeError) as caught:
        await bridge.build_bundle(
            client,
            _tool_result(embedded_text=json.dumps(dynamic)),
            server_identity=SERVER_ID,
        )  # type: ignore[arg-type]
    assert caught.value.code == "invalid_dynamic_messages"


@pytest.mark.asyncio
async def test_failed_resource_read_is_not_cached() -> None:
    bridge = A2UIBridge()
    client = FakeClient(failures=1)
    with pytest.raises(A2UIBridgeError) as caught:
        await bridge.build_bundle(client, client.result, server_identity=SERVER_ID)  # type: ignore[arg-type]
    assert caught.value.code == "resource_read_failed"

    bundle = await bridge.build_bundle(client, client.result, server_identity=SERVER_ID)  # type: ignore[arg-type]
    assert bundle is not None
    assert client.read_count == 2


@pytest.mark.asyncio
async def test_concurrent_uncached_resolution_is_single_flight() -> None:
    bridge = A2UIBridge()
    client = FakeClient(delay=0.02)
    bundles = await asyncio.gather(
        *(
            bridge.build_bundle(client, client.result, server_identity=SERVER_ID)  # type: ignore[arg-type]
            for _ in range(12)
        )
    )
    assert client.read_count == 1
    assert all(bundle is not None and len(bundle.messages) == 3 for bundle in bundles)


@pytest.mark.asyncio
async def test_text_and_structured_data_survive_presentation_failure() -> None:
    bridge = A2UIBridge()
    client = FakeClient("not-json")
    execution = await call_mcp_tool(  # type: ignore[arg-type]
        client, SERVER_ID, "database_overview", {"limit": 50}, bridge=bridge
    )
    response = response_from_tool(execution)
    assert client.call_count == 1
    assert execution.presentation_error is True
    assert response.message == "Database overview loaded."
    assert response.data == {"ok": True, "object_count": 1}
    assert response.a2ui is None


@pytest.mark.asyncio
async def test_database_overview_uses_only_frontend_supported_components() -> None:
    bridge = A2UIBridge()
    client = FakeClient()
    bundle = await bridge.build_bundle(client, client.result, server_identity=SERVER_ID)  # type: ignore[arg-type]
    assert bundle is not None
    components = bundle.messages[1]["updateComponents"]["components"]
    assert {component["component"] for component in components} == {
        "Text",
        "Button",
        "Card",
        "Column",
    }
