"""Deterministic planning of the smallest financial-domain read for a request.

A classified financial question never reaches the model: this planner knows
which capability each intent needs, so it skips discovery entirely and emits at
most one tool call. Same query plus same intent plus same observations always
yields the same plan, which is what makes the financial fast path reproducible.

What it must still check is that MCP answered at all. The capability set is the
scoped-execution boundary, never the (now tiny) set of schemas the model was
handed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from ..mcp_client import addressable_financial_tool_names
from ..schemas.banking_view import FinancialIntent
from ..state import GraphState
from .model import ModelTurn
from .observations import has_table_observation, has_tool_observation, normalized_query
from .tool_visibility import tool_definitions

#: The single capability each intent reads from, before per-query refinements.
_TOOL_BY_INTENT: dict[str, str] = {
    "financial-summary": "get_financial_overview",
    "transactions": "get_transactions",
    "spending-analysis": "analyze_spending",
    "cash-flow": "get_cash_flow",
    "budgets": "get_budget_progress",
    "recurring-payments": "get_upcoming_payments",
    "credit-card": "get_accounts",
    "debts": "get_debt_overview",
    "transfers": "get_payment_activity",
    "card-security": "get_transaction_disputes",
    "savings-goals": "get_savings_progress",
    "banking-information": "get_accounts",
}

_CURRENT_MONTH_TOOLS = {"get_financial_overview", "get_transactions", "analyze_spending"}
_MAX_DAYS_AHEAD = 365
_DEFAULT_DAYS_AHEAD = 30


def _banking_information_tool(text: str) -> str:
    if any(term in text for term in ("estado de cuenta", "bank statement", "statement")):
        return "get_bank_statements"
    if any(term in text for term in ("beneficiario", "beneficiary", "destinatario")):
        return "get_beneficiaries"
    return "get_accounts"


def _tool_for_intent(intent: FinancialIntent, text: str) -> str | None:
    if intent == "banking-information":
        return _banking_information_tool(text)
    if intent == "financial-summary" and any(
        term in text for term in ("alerta", "alert", "aviso", "warning")
    ):
        return "get_financial_alerts"
    return _TOOL_BY_INTENT.get(intent)


def _request_arguments(tool_name: str, text: str) -> dict[str, Any]:
    """Bilingual query refinements, one tool at a time."""
    request: dict[str, Any] = {}
    if tool_name in _CURRENT_MONTH_TOOLS:
        request["period"] = "current_month"
    elif tool_name == "get_cash_flow":
        request["period"] = (
            "last_6_months"
            if any(term in text for term in ("seis", "six", "6 meses", "6 months"))
            else "last_12_months"
        )
    elif tool_name == "get_payment_activity":
        request["period"] = "last_90_days"
    if tool_name == "get_transactions" and "uber" in text:
        request["merchant_query"] = "Uber"
    if tool_name == "get_budget_progress" and any(
        term in text for term in ("comida", "alimentos", "food")
    ):
        request["category"] = "food"
    if tool_name == "get_upcoming_payments":
        match = re.search(r"\b(\d{1,3})\s+(?:dias|days)\b", text)
        request["days_ahead"] = (
            min(int(match.group(1)), _MAX_DAYS_AHEAD) if match else _DEFAULT_DAYS_AHEAD
        )
    return request


def _debt_scenario_turn(state: GraphState, available: frozenset[str]) -> ModelTurn | None:
    """Comparing payoff scenarios needs an owned debt id first.

    The comparison runs only against a debt this user's own overview returned,
    so the overview is read first and the id is taken from it.
    """
    overview = next(
        (
            observation
            for observation in state.get("tool_observations", [])
            if observation.get("name") == "get_debt_overview"
            and observation.get("is_error") is not True
        ),
        None,
    )
    if overview is None:
        return None
    data = overview.get("data")
    debts = data.get("debts") if isinstance(data, Mapping) else None
    first_debt = debts[0] if isinstance(debts, list) and debts else None
    debt_id = first_debt.get("id") if isinstance(first_debt, Mapping) else None
    if not isinstance(debt_id, str):
        return ModelTurn(message="No encontré una deuda guardada para comparar.")
    tool_name = "compare_debt_scenarios"
    if tool_name not in available or has_tool_observation(state, tool_name):
        return ModelTurn(message="")
    return ModelTurn(
        message="",
        tool_calls=(
            {
                "name": tool_name,
                "arguments": {"request": {"debt_id": debt_id, "order_by": "lowest_total_interest"}},
            },
        ),
    )


def financial_data_turn(state: GraphState, intent: FinancialIntent) -> ModelTurn:
    """Plan the smallest bilingual financial-domain read for the request."""
    available = addressable_financial_tool_names(tool.name for tool in tool_definitions(state))
    text = normalized_query(state["user_query"])
    tool_name = _tool_for_intent(intent, text)

    # User context already made and retained the scoped accounts read. That is
    # the complete verified source used by the summary view, so asking the MCP
    # for a broader overview would be a duplicate deterministic read.
    if tool_name == "get_financial_overview" and has_table_observation(state, "accounts"):
        return ModelTurn(message="")

    compare_requested = intent == "debts" and any(
        term in text for term in ("compara", "comparar", "escenario", "compare", "scenario")
    )
    if compare_requested:
        scenario_turn = _debt_scenario_turn(state, available)
        if scenario_turn is not None:
            return scenario_turn
        tool_name = "get_debt_overview"

    if tool_name is None:
        return ModelTurn(message="No está disponible la consulta financiera requerida.")
    if has_tool_observation(state, tool_name):
        return ModelTurn(message="")
    if tool_name not in available:
        return ModelTurn(message="No está disponible la consulta financiera requerida.")
    request = _request_arguments(tool_name, text)
    if intent == "credit-card":
        request["account_type"] = "credit"
    arguments = {"request": request}
    return ModelTurn(message="", tool_calls=({"name": tool_name, "arguments": arguments},))


def financial_data_observed(state: GraphState, intent: FinancialIntent) -> bool:
    """Whether retrieval reached a terminal observation for this request.

    Errors count as completed attempts, which lets the builder render an
    explicit verified failure without confusing it with a tool never called.
    """
    text = normalized_query(state["user_query"])
    tool_name = _tool_for_intent(intent, text)
    if tool_name == "get_financial_overview" and has_table_observation(state, "accounts"):
        return True
    if tool_name is None:
        return False
    compare_requested = intent == "debts" and any(
        term in text for term in ("compara", "comparar", "escenario", "compare", "scenario")
    )
    if not compare_requested:
        return has_tool_observation(state, tool_name)
    if has_tool_observation(state, "compare_debt_scenarios"):
        return True
    overview = next(
        (
            observation
            for observation in state.get("tool_observations", [])
            if observation.get("name") == "get_debt_overview"
        ),
        None,
    )
    if overview is None:
        return False
    if overview.get("is_error") is True:
        return True
    data = overview.get("data")
    debts = data.get("debts") if isinstance(data, Mapping) else None
    first_debt = debts[0] if isinstance(debts, list) and debts else None
    debt_id = first_debt.get("id") if isinstance(first_debt, Mapping) else None
    return not isinstance(debt_id, str)
