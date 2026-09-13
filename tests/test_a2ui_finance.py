"""Finance catalog validation, forwarding, and cross-repository parity tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from fastmcp.client.client import CallToolResult
from mcp.types import EmbeddedResource, TextContent, TextResourceContents

from fluidbank_orchestrator.schemas.a2ui import (
    A2UI_BASIC_CATALOG,
    A2UI_FINANCE_CATALOG,
    A2UI_FINANCE_V2_CATALOG,
    A2UI_MIME_TYPE,
    A2UIValidationError,
    validate_complete_sequence,
    validate_static_template,
)
from fluidbank_orchestrator.services.a2ui_bridge import A2UIBridge
from fluidbank_orchestrator.services.financial_presentation import build_financial_presentation
from fluidbank_orchestrator.state import UserProfile

RESOURCE_URI = "a2ui://finance/data-chart"
SURFACE_ID = "data-chart"

FINANCE_TEMPLATE = [
    {
        "version": "v0.9.1",
        "createSurface": {"surfaceId": SURFACE_ID, "catalogId": A2UI_FINANCE_CATALOG},
    },
    {
        "version": "v0.9.1",
        "updateComponents": {
            "surfaceId": SURFACE_ID,
            "components": [
                {"id": "root", "component": "Card", "child": "column"},
                {"id": "column", "component": "Column", "children": ["chart"]},
                {"id": "chart", "component": "Chart", "chart": {"path": "/chart"}},
            ],
        },
    },
]


def _fixtures() -> dict[str, list[dict[str, Any]]]:
    path = Path(__file__).with_name("fixtures") / "chart-contract.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _sequence(chart: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        *deepcopy(FINANCE_TEMPLATE),
        {
            "version": "v0.9.1",
            "updateDataModel": {
                "surfaceId": SURFACE_ID,
                "path": "/",
                "value": {"chart": deepcopy(chart)},
            },
        },
    ]


def test_finance_and_basic_catalog_rules_are_strict() -> None:
    surface_id, messages = validate_static_template(FINANCE_TEMPLATE)
    assert surface_id == SURFACE_ID
    assert messages == FINANCE_TEMPLATE

    unknown = deepcopy(FINANCE_TEMPLATE)
    unknown[0]["createSurface"]["catalogId"] = "https://example.invalid/catalog"
    with pytest.raises(A2UIValidationError):
        validate_static_template(unknown)

    basic_chart = deepcopy(FINANCE_TEMPLATE)
    basic_chart[0]["createSurface"]["catalogId"] = A2UI_BASIC_CATALOG
    with pytest.raises(A2UIValidationError):
        validate_static_template(basic_chart)


def test_agent_chart_validation_matches_representative_contract_fixtures() -> None:
    fixtures = _fixtures()
    for chart in fixtures["accepted"]:
        validate_complete_sequence(_sequence(chart), SURFACE_ID)
    for chart in fixtures["rejected"]:
        with pytest.raises(A2UIValidationError):
            validate_complete_sequence(_sequence(chart), SURFACE_ID)

    oversized = deepcopy(fixtures["accepted"][0])
    oversized["props"]["data"] = [{"label": f"P{index}", "values": [index]} for index in range(241)]
    with pytest.raises(A2UIValidationError):
        validate_complete_sequence(_sequence(oversized), SURFACE_ID)

    nonfinite = deepcopy(fixtures["accepted"][0])
    nonfinite["props"]["data"] = [{"label": "Bad", "values": [float("nan")]}]
    with pytest.raises(A2UIValidationError):
        validate_complete_sequence(_sequence(nonfinite), SURFACE_ID)


def test_bound_chart_is_validated_after_incremental_data_updates() -> None:
    chart = deepcopy(_fixtures()["accepted"][0])
    direct = [
        *deepcopy(FINANCE_TEMPLATE),
        {
            "version": "v0.9.1",
            "updateDataModel": {
                "surfaceId": SURFACE_ID,
                "path": "/chart",
                "value": chart,
            },
        },
    ]
    validate_complete_sequence(direct, SURFACE_ID)

    tampered = [
        *_sequence(chart),
        {
            "version": "v0.9.1",
            "updateDataModel": {
                "surfaceId": SURFACE_ID,
                "path": "/chart/style",
                "value": {"color": "red"},
            },
        },
    ]
    with pytest.raises(A2UIValidationError):
        validate_complete_sequence(tampered, SURFACE_ID)


class FinanceClient:
    def __init__(self) -> None:
        self.read_count = 0

    async def read_resource(self, uri: str) -> list[TextResourceContents]:
        assert uri == RESOURCE_URI
        self.read_count += 1
        return [
            TextResourceContents(
                uri=RESOURCE_URI,
                mime_type=A2UI_MIME_TYPE,
                text=json.dumps(FINANCE_TEMPLATE),
            )
        ]


def _result(chart: dict[str, Any]) -> CallToolResult:
    dynamic = _sequence(chart)[-1:]
    return CallToolResult(
        content=[
            TextContent(text="Chart loaded."),
            EmbeddedResource(
                resource=TextResourceContents(
                    uri=f"{RESOURCE_URI}/data",
                    mime_type=A2UI_MIME_TYPE,
                    text=json.dumps(dynamic),
                )
            ),
        ],
        structured_content={"ok": True},
        meta={"ui": {"resourceUri": RESOURCE_URI, "mimeType": A2UI_MIME_TYPE}},
        data=None,
    )


@pytest.mark.asyncio
async def test_finance_template_is_cached_and_forwarded_as_ordered_objects() -> None:
    bridge = A2UIBridge()
    client = FinanceClient()
    chart = _fixtures()["accepted"][0]
    first = await bridge.build_bundle(client, _result(chart), server_identity="test-server")  # type: ignore[arg-type]
    second = await bridge.build_bundle(client, _result(chart), server_identity="test-server")  # type: ignore[arg-type]

    assert first is not None and second is not None
    assert client.read_count == 1
    assert all(isinstance(message, dict) for message in first.messages)
    assert [next(key for key in message if key != "version") for message in first.messages] == [
        "createSurface",
        "updateComponents",
        "updateDataModel",
    ]
    assert first.messages == second.messages


def test_cross_repository_catalog_id_schema_and_fixture_parity() -> None:
    repo = Path(__file__).resolve().parents[1]
    monorepo = repo.parent
    agent_catalog = json.loads(
        (repo / "src/fluidbank_orchestrator/a2ui_catalogs/finance_v1.json").read_text(
            encoding="utf-8"
        )
    )
    mcp_catalog = json.loads(
        (
            monorepo / "hackmty2026-mcp/src/supabase_mcp/a2ui_support/catalogs/finance_v1.json"
        ).read_text(encoding="utf-8")
    )
    mobile_types = (monorepo / "HackMTY2026_Mobile/src/features/a2ui/types.ts").read_text(
        encoding="utf-8"
    )
    mobile_fixtures = json.loads(
        (monorepo / "HackMTY2026_Mobile/tests/fixtures/chart-contract.json").read_text(
            encoding="utf-8"
        )
    )

    assert agent_catalog == mcp_catalog
    assert agent_catalog["catalogId"] == A2UI_FINANCE_CATALOG
    assert A2UI_FINANCE_CATALOG in mobile_types
    assert mobile_fixtures == _fixtures()


def test_finance_v2_banking_view_accepts_valid_summary_and_rejects_unknown_props() -> None:
    profile: UserProfile = {
        "literacy_level": "medium",
        "font_scale": "lg",
        "contrast": "high",
        "hit_target": "large",
        "overdraft_risk": 0.0,
        "recurring_expenses": 0.0,
        "available_balance": 100.0,
        "owned_balances": {"MXN": 100.0},
    }
    observations = [
        {
            "name": "select_rows",
            "arguments": {"table": "accounts"},
            "is_error": False,
            "data": {
                "rows": [
                    {
                        "id": "checking",
                        "account_type": "checking",
                        "currency": "MXN",
                        "available_balance": 100,
                    }
                ]
            },
        }
    ]
    bundle = build_financial_presentation("financial-summary", observations, profile).a2ui
    assert bundle.messages[0]["createSurface"]["catalogId"] == A2UI_FINANCE_V2_CATALOG
    assert bundle.resource_uri == "a2ui://finance/view"
    assert bundle.messages[0]["createSurface"]["surfaceId"] == "financial-view"
    canonical_template = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "hackmty2026-mcp/src/supabase_mcp/a2ui_support/templates/financial_view.json"
        ).read_text(encoding="utf-8")
    )
    assert bundle.messages[:2] == canonical_template
    assert bundle.messages[1]["updateComponents"]["components"] == [
        {
            "id": "root",
            "component": "Column",
            "children": ["banking_view", "request_financial_view_button"],
            "accessibility": {"label": {"path": "/viewLabel"}},
        },
        {
            "id": "banking_view",
            "component": "BankingView",
            "view": {"path": "/view"},
            "accessibility": {"label": {"path": "/viewLabel"}},
        },
        {
            "id": "request_financial_view_label",
            "component": "Text",
            "text": {"path": "/actionLabel"},
        },
        {
            "id": "request_financial_view_button",
            "component": "Button",
            "child": "request_financial_view_label",
            "variant": "primary",
            "accessibility": {"label": {"path": "/actionLabel"}},
            "action": {
                "event": {
                    "name": "request_financial_view",
                    "context": {"intent": {"path": "/requestIntent"}},
                }
            },
        },
    ]
    # A generated surface announces itself: the accessible name is bound to the
    # data model, so it names the view the user is actually looking at.
    assert bundle.messages[2]["updateDataModel"]["value"]["viewLabel"] == "Tu panorama financiero"
    assert bundle.messages[2]["updateDataModel"]["value"]["actionLabel"] == (
        "Ver gastos del último mes"
    )
    validate_complete_sequence(bundle.messages, "financial-view")

    invalid = deepcopy(bundle.messages)
    invalid[-1]["updateDataModel"]["value"]["view"]["style"] = {"color": "red"}
    with pytest.raises(A2UIValidationError):
        validate_complete_sequence(invalid, "financial-view")


def test_all_shared_mobile_banking_view_examples_validate() -> None:
    monorepo = Path(__file__).resolve().parents[2]
    examples = json.loads(
        (monorepo / "HackMTY2026_Mobile/docs/a2ui/banking-view.examples.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(examples) == 13
    for example in examples:
        messages = example["a2ui"]["messages"]
        surface_id = messages[0]["createSurface"]["surfaceId"]
        validate_complete_sequence(messages, surface_id)
