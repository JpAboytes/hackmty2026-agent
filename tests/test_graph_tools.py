"""Deterministic model/tool-loop tests for visualization routing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent

import fluidbank_orchestrator.graph as graph_module
from fluidbank_orchestrator.graph import ModelTurn, ToolAwareModel, build_graph
from fluidbank_orchestrator.mcp_client import MCPToolDefinition, MCPToolExecution
from fluidbank_orchestrator.schemas.a2ui import A2UIBundle
from fluidbank_orchestrator.state import UserProfile

EMAIL = "ana.demo@fluidbank.test"
PROFILE: UserProfile = {
    "literacy_level": "medium",
    "font_scale": "lg",
    "contrast": "high",
    "hit_target": "large",
    "overdraft_risk": 0.1,
    "recurring_expenses": 200.0,
    "available_balance": 1000.0,
}


class FakeModel(ToolAwareModel):
    def __init__(self, message: str = "Respuesta textual.") -> None:
        self.message = message
        self.bound_tool_names: list[set[str]] = []

    async def generate(
        self,
        *,
        query: str,
        profile: UserProfile,
        tools: Sequence[MCPToolDefinition],
        observations: Sequence[Mapping[str, Any]],
    ) -> ModelTurn:
        del query, profile, observations
        self.bound_tool_names.append({tool.name for tool in tools})
        return ModelTurn(message=self.message)


class SelectUsersModel(ToolAwareModel):
    async def generate(
        self,
        *,
        query: str,
        profile: UserProfile,
        tools: Sequence[MCPToolDefinition],
        observations: Sequence[Mapping[str, Any]],
    ) -> ModelTurn:
        del query, profile, tools
        if observations:
            return ModelTurn(message="Usuario cargado.")
        return ModelTurn(
            message="",
            tool_calls=(
                {
                    "name": "select_rows",
                    "arguments": {
                        "schema": "public",
                        "table": "users",
                        "filters": [
                            {"column": "email", "operator": "eq", "value": "wrong@example.com"},
                            {"column": "active", "operator": "eq", "value": True},
                        ],
                    },
                },
            ),
        )


def _tools(*, visualization: bool = True) -> list[MCPToolDefinition]:
    names = ["list_allowed_tables", "describe_table"]
    if visualization:
        names.append("visualize_allowed_data")
    return [
        MCPToolDefinition(name=name, description=f"{name} tool", input_schema={"type": "object"})
        for name in names
    ]


def _execution(
    name: str, arguments: Mapping[str, Any], *, numeric: bool = True
) -> MCPToolExecution:
    if name == "list_allowed_tables":
        structured: dict[str, Any] = {
            "ok": True,
            "object_count": 1,
            "objects": [{"schema": "public", "table": "transactions", "kind": "table"}],
        }
        message = "Allowed objects loaded."
        meta = None
    elif name == "describe_table":
        columns = [
            {
                "name": "occurred_at",
                "data_type": "DATE",
                "nullable": False,
                "primary_key": False,
            },
            {"name": "merchant", "data_type": "TEXT", "nullable": False, "primary_key": False},
        ]
        if numeric:
            columns.append(
                {
                    "name": "amount",
                    "data_type": "NUMERIC(14, 2)",
                    "nullable": False,
                    "primary_key": False,
                }
            )
        structured = {
            "ok": True,
            "schema": arguments["schema"],
            "table": arguments["table"],
            "columns": columns,
        }
        message = "Table described."
        meta = None
    else:
        request = arguments["request"]
        kind = request["visualization"]["kind"]
        structured = {"ok": True, "row_count": 2, "chart": {"kind": kind}}
        message = f"{kind} chart loaded."
        meta = {
            "ui": {
                "resourceUri": "a2ui://finance/data-chart",
                "mimeType": "application/a2ui+json",
            }
        }
        chart = (
            {
                "kind": "area",
                "accessibleSummary": "Area chart.",
                "props": {
                    "data": [{"label": "2026-09-12", "values": [10.0]}],
                    "series": [{"id": "amount", "label": "amount"}],
                },
            }
            if kind == "area"
            else {
                "kind": "heatmap",
                "accessibleSummary": "Heatmap chart.",
                "props": {
                    "data": [{"date": "2026-09-12", "value": 10.0}],
                    "initialView": "month",
                },
            }
        )
        a2ui = A2UIBundle(
            resource_uri="a2ui://finance/data-chart",
            messages=[
                {
                    "version": "v0.9.1",
                    "createSurface": {
                        "surfaceId": "data-chart",
                        "catalogId": "https://fluidbank.app/a2ui/catalogs/finance/v1",
                    },
                },
                {
                    "version": "v0.9.1",
                    "updateComponents": {
                        "surfaceId": "data-chart",
                        "components": [
                            {
                                "id": "root",
                                "component": "Chart",
                                "chart": {"path": "/chart"},
                            }
                        ],
                    },
                },
                {
                    "version": "v0.9.1",
                    "updateDataModel": {
                        "surfaceId": "data-chart",
                        "path": "/",
                        "value": {"chart": chart},
                    },
                },
            ],
        )
    if name != "visualize_allowed_data":
        a2ui = None
    return MCPToolExecution(
        result=CallToolResult(
            content=[TextContent(text=message)],
            structured_content=structured,
            meta=meta,
            data=None,
        ),
        a2ui=a2ui,
    )


async def _run(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    *,
    visualization: bool = True,
    numeric: bool = True,
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]], FakeModel]:
    calls: list[tuple[str, dict[str, Any]]] = []
    model = FakeModel()

    async def fake_profile(_email: str) -> UserProfile:
        return PROFILE.copy()

    async def load_tools() -> list[MCPToolDefinition]:
        return _tools(visualization=visualization)

    async def execute(name: str, arguments: Mapping[str, Any] | None) -> MCPToolExecution:
        resolved = dict(arguments or {})
        calls.append((name, resolved))
        return _execution(name, resolved, numeric=numeric)

    monkeypatch.setattr(graph_module, "fetch_user_context", fake_profile)
    result = await build_graph(
        model=model,
        tool_loader=load_tools,
        tool_executor=execute,
    ).ainvoke({"user_query": query, "user_email": EMAIL})
    return result, calls, model


@pytest.mark.asyncio
async def test_explicit_trend_discovers_schema_then_calls_area_visualization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls, model = await _run(monkeypatch, "Show me a chart of account activity over time")

    assert [name for name, _arguments in calls] == [
        "list_allowed_tables",
        "describe_table",
        "visualize_allowed_data",
    ]
    request = calls[-1][1]["request"]
    assert request["source"] == {"schema": "public", "table": "transactions"}
    assert request["visualization"] == {
        "kind": "area",
        "x_column": "occurred_at",
        "y_columns": ["amount"],
    }
    execution = result["final_tool_execution"]
    assert isinstance(execution, MCPToolExecution)
    assert execution.result.meta is not None
    assert execution.result.meta["ui"]["resourceUri"] == "a2ui://finance/data-chart"
    assert execution.a2ui is not None
    assert [
        next(key for key in message if key != "version") for message in execution.a2ui.messages
    ] == [
        "createSurface",
        "updateComponents",
        "updateDataModel",
    ]
    assert all("visualize_allowed_data" in names for names in model.bound_tool_names)


@pytest.mark.asyncio
async def test_calendar_activity_selects_heatmap(monkeypatch: pytest.MonkeyPatch) -> None:
    _result, calls, _model = await _run(
        monkeypatch, "Show account activity by day as a calendar heatmap."
    )
    visualization = calls[-1][1]["request"]["visualization"]
    assert visualization == {
        "kind": "heatmap",
        "date_column": "occurred_at",
        "value_column": "amount",
        "initial_view": "month",
    }


@pytest.mark.asyncio
async def test_unknown_columns_always_trigger_discovery_before_visualization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _result, calls, _model = await _run(
        monkeypatch, "Graph made_up_date against imaginary_total over time"
    )
    assert [name for name, _arguments in calls[:2]] == [
        "list_allowed_tables",
        "describe_table",
    ]
    assert calls[-1][1]["request"]["visualization"]["x_column"] == "occurred_at"
    assert "made_up_date" not in str(calls[-1][1])


@pytest.mark.asyncio
async def test_non_visual_question_does_not_call_visualization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, calls, _model = await _run(monkeypatch, "¿Cuál es mi saldo disponible?")
    assert calls == []
    assert result["message"] == "Respuesta textual."


@pytest.mark.asyncio
async def test_select_users_is_scoped_to_request_email(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_profile(_email: str) -> UserProfile:
        return PROFILE.copy()

    async def load_tools() -> list[MCPToolDefinition]:
        return [
            MCPToolDefinition(
                name="select_rows",
                description="Select rows.",
                input_schema={"type": "object"},
            )
        ]

    async def execute(name: str, arguments: Mapping[str, Any] | None) -> MCPToolExecution:
        resolved = dict(arguments or {})
        calls.append((name, resolved))
        return MCPToolExecution(
            result=CallToolResult(
                content=[TextContent(text="One user loaded.")],
                structured_content={"ok": True, "rows": [{"id": "user-id"}]},
                meta=None,
                data=None,
            ),
            a2ui=None,
        )

    monkeypatch.setattr(graph_module, "fetch_user_context", fake_profile)
    result = await build_graph(
        model=SelectUsersModel(),
        tool_loader=load_tools,
        tool_executor=execute,
    ).ainvoke({"user_query": "Carga mi usuario", "user_email": EMAIL})

    assert result["message"] == "Usuario cargado."
    assert calls == [
        (
            "select_rows",
            {
                "schema": "public",
                "table": "users",
                "filters": [
                    {"column": "active", "operator": "eq", "value": True},
                    {"column": "email", "operator": "eq", "value": EMAIL},
                ],
            },
        )
    ]


@pytest.mark.asyncio
async def test_unavailable_or_invalid_chart_data_has_safe_text_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_tool, calls, _model = await _run(
        monkeypatch, "Visualiza la actividad diaria", visualization=False
    )
    assert calls == []
    assert "no ofrece visualize_allowed_data" in missing_tool["message"]

    invalid_data, calls, _model = await _run(
        monkeypatch, "Visualiza la actividad diaria", numeric=False
    )
    assert [name for name, _arguments in calls] == ["list_allowed_tables", "describe_table"]
    assert "columnas de fecha y valor numérico" in invalid_data["message"]
