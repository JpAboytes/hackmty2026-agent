"""Deterministic bilingual routing for MCP financial domain tools."""

from __future__ import annotations

from uuid import UUID

import pytest

from fluidbank_orchestrator.graph import _financial_data_turn, _model_tool_schema
from fluidbank_orchestrator.mcp_client import (
    FINANCIAL_DOMAIN_TOOL_NAMES,
    MODEL_TOOL_NAMES,
    MCPToolDefinition,
    enforce_trusted_user_scope,
)

USER_A = UUID("11111111-1111-1111-1111-111111111111")
USER_B = UUID("22222222-2222-2222-2222-222222222222")


def _state(query: str) -> dict[str, object]:
    return {
        "user_query": query,
        "available_tools": [
            MCPToolDefinition(name, f"ES / EN {name}", {"type": "object"}).as_dict()
            for name in FINANCIAL_DOMAIN_TOOL_NAMES
        ],
        "tool_observations": [],
    }


@pytest.mark.parametrize(
    ("query", "intent", "expected"),
    [
        ("¿Cómo van mis finanzas?", "financial-summary", "get_financial_overview"),
        ("¿En qué gasté más este mes?", "spending-analysis", "analyze_spending"),
        ("Enséñame mis compras de Uber", "transactions", "get_transactions"),
        ("Compara mis ingresos y gastos de seis meses", "cash-flow", "get_cash_flow"),
        ("¿Cuánto me queda de presupuesto de comida?", "budgets", "get_budget_progress"),
        ("¿Cómo va mi meta para vacaciones?", "savings-goals", "get_savings_progress"),
        ("¿Cuánto debo y cuándo pago?", "debts", "get_debt_overview"),
        ("¿Qué pagos tengo en los próximos 15 días?", "recurring-payments", "get_upcoming_payments"),
        ("Show my Uber purchases", "transactions", "get_transactions"),
        ("Compare income and expenses for six months", "cash-flow", "get_cash_flow"),
    ],
)
def test_financial_intent_routes_to_one_domain_tool(
    query: str, intent: str, expected: str
) -> None:
    turn = _financial_data_turn(_state(query), intent)  # type: ignore[arg-type]
    assert [call["name"] for call in turn.tool_calls] == [expected]
    assert turn.tool_calls[0]["name"] != "select_rows"


def test_compare_scenarios_resolves_owned_debt_then_calls_comparator() -> None:
    state = _state("Compara mis escenarios para liquidar esta deuda")
    state["tool_observations"] = [
        {
            "name": "get_debt_overview",
            "is_error": False,
            "data": {"debts": [{"id": "33333333-3333-3333-3333-333333333333"}]},
        }
    ]
    turn = _financial_data_turn(state, "debts")  # type: ignore[arg-type]
    assert [call["name"] for call in turn.tool_calls] == ["compare_debt_scenarios"]


def test_authenticated_uuid_overwrites_model_identity_for_every_financial_tool() -> None:
    for name in FINANCIAL_DOMAIN_TOOL_NAMES:
        original = {
            "request": {
                "scope": {"user_id": str(USER_B)},
                "user_id": str(USER_B),
                "email": "attacker@example.test",
            }
        }
        scoped = enforce_trusted_user_scope(name, original, USER_A)
        assert scoped is not None
        assert scoped["request"]["scope"] == {"user_id": str(USER_A)}
        assert "user_id" not in scoped["request"]
        assert "email" not in scoped["request"]
        assert original["request"]["scope"] == {"user_id": str(USER_B)}


def test_generic_schema_tools_are_not_in_the_model_financial_path() -> None:
    assert FINANCIAL_DOMAIN_TOOL_NAMES <= MODEL_TOOL_NAMES
    assert {"select_rows", "describe_table", "list_allowed_tables"}.isdisjoint(MODEL_TOOL_NAMES)


def test_financial_model_schema_hides_nested_scope() -> None:
    definition = MCPToolDefinition(
        "get_transactions",
        "Movimientos / Transactions",
        {
            "type": "object",
            "properties": {
                "request": {
                    "type": "object",
                    "properties": {
                        "scope": {"type": "object"},
                        "period": {"type": "string"},
                    },
                    "required": ["scope", "period"],
                }
            },
        },
    )
    request_schema = _model_tool_schema(definition)["properties"]["request"]
    assert "scope" not in request_schema["properties"]
    assert request_schema["required"] == ["period"]
