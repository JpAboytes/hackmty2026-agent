"""Deterministic bilingual routing for MCP financial domain tools."""

from __future__ import annotations

from uuid import UUID

import pytest

from fluidbank_orchestrator.graph import _financial_data_turn, _model_tool_schema
from fluidbank_orchestrator.mcp_client import (
    CALL_TOOL_NAME,
    DISCOVERY_TOOL_NAMES,
    FINANCIAL_DOMAIN_TOOL_NAMES,
    SCOPED_TOOL_NAMES,
    SEARCH_TOOL_NAME,
    MCPToolDefinition,
    enforce_trusted_user_scope,
)
from fluidbank_orchestrator.services.financial_presentation import build_financial_presentation
from fluidbank_orchestrator.state import UserProfile

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
        (
            "¿Qué pagos tengo en los próximos 15 días?",
            "recurring-payments",
            "get_upcoming_payments",
        ),
        ("Show my Uber purchases", "transactions", "get_transactions"),
        ("Compare income and expenses for six months", "cash-flow", "get_cash_flow"),
    ],
)
def test_financial_intent_routes_to_one_domain_tool(query: str, intent: str, expected: str) -> None:
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


def test_every_financial_tool_stays_inside_the_scoping_boundary() -> None:
    """Discovery changed how tools are found, not which ones carry a user scope."""
    assert FINANCIAL_DOMAIN_TOOL_NAMES <= SCOPED_TOOL_NAMES
    assert DISCOVERY_TOOL_NAMES.isdisjoint(SCOPED_TOOL_NAMES)


def test_the_deterministic_router_ignores_the_model_facing_catalog() -> None:
    """The classifier routes on capability, not on what the model was shown.

    With progressive discovery the model sees only the search pair, so gating
    this path on that list would make every classified financial query fail.
    """
    state = _state("¿Cuánto debo y cuándo pago?")
    state["available_tools"] = [
        MCPToolDefinition(name, f"ES / EN {name}", {"type": "object"}).as_dict()
        for name in (SEARCH_TOOL_NAME, CALL_TOOL_NAME)
    ]

    turn = _financial_data_turn(state, "debts")  # type: ignore[arg-type]

    assert [call["name"] for call in turn.tool_calls] == ["get_debt_overview"]


def test_an_unreachable_mcp_server_does_not_invent_a_financial_answer() -> None:
    state = _state("¿Cuánto debo y cuándo pago?")
    state["available_tools"] = []

    turn = _financial_data_turn(state, "debts")  # type: ignore[arg-type]

    assert turn.tool_calls == ()
    assert "No está disponible" in turn.message


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


_PROFILE: UserProfile = {
    "literacy_level": "medium",
    "font_scale": "lg",
    "contrast": "high",
    "hit_target": "large",
    "overdraft_risk": 0.0,
    "recurring_expenses": 0.0,
    "available_balance": 100.0,
    "owned_balances": {"MXN": 100.0},
}


def test_spending_categories_are_summed_before_the_view_is_built() -> None:
    """The demo database has more categories than the contract has buckets.

    `groceries` and `dining` are both food, and `subscription`, `credit_card`
    and `housing` are all other. Mapping the rows one by one produced repeated
    buckets, which the BankingView contract rejects, so every spending answer
    failed validation and the user got no view at all.
    """
    by_category = [
        {"currency": "MXN", "category": name, "amount": amount}
        for name, amount in (
            ("groceries", 300.0),
            ("dining", 200.0),
            ("subscription", 90.0),
            ("credit_card", 60.0),
            ("housing", 50.0),
            ("transport", 120.0),
        )
    ]
    observations = [
        {
            "name": "analyze_spending",
            "arguments": {},
            "is_error": False,
            "data": {
                "ok": True,
                "summary_by_currency": [{"currency": "MXN", "expenses": 820.0}],
                "by_category": by_category,
                "daily_series": [{"currency": "MXN", "date": "2026-09-11", "amount": 820.0}],
            },
        }
    ]

    presentation = build_financial_presentation("spending-analysis", observations, _PROFILE)
    view = presentation.a2ui.messages[2]["updateDataModel"]["value"]["view"]
    amounts = {row["category"]: row["amount"] for row in view["categories"]}

    assert len(amounts) == len(view["categories"])
    assert amounts["food"] == 500.0
    assert amounts["other"] == 200.0
    assert amounts["transport"] == 120.0
    # The breakdown may be truncated, but it can never exceed the headline.
    assert sum(amounts.values()) <= view["totalSpent"] + 0.01


def test_spending_categories_stay_within_the_eight_the_contract_allows() -> None:
    by_category = [
        {"currency": "MXN", "category": name, "amount": 10.0 * index}
        for index, name in enumerate(
            (
                "groceries",
                "dining",
                "transport",
                "health",
                "utilities",
                "shopping",
                "entertainment",
                "transfer",
                "subscription",
                "credit_card",
                "housing",
            ),
            start=1,
        )
    ]
    observations = [
        {
            "name": "analyze_spending",
            "arguments": {},
            "is_error": False,
            "data": {
                "ok": True,
                "summary_by_currency": [{"currency": "MXN", "expenses": 660.0}],
                "by_category": by_category,
                "daily_series": [{"currency": "MXN", "date": "2026-09-11", "amount": 660.0}],
            },
        }
    ]

    view = build_financial_presentation("spending-analysis", observations, _PROFILE).a2ui.messages[
        2
    ]["updateDataModel"]["value"]["view"]

    assert len(view["categories"]) <= 8
    assert len({row["category"] for row in view["categories"]}) == len(view["categories"])
