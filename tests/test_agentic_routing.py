"""The model decides; the graph only enforces invariants.

These tests replace the old phrase-router tests. What they pin down is that the
same query can take different paths depending on what the model decides, that
discovery can return several candidates and the model may pick any number of
them, that results always return to the model before a surface is produced, and
that the parts which must *not* be the model's choice still are not.
"""

from __future__ import annotations

from typing import Any

import pytest
from scripted_model import (
    USER_A,
    Recorder,
    ScriptedModel,
    answers,
    calls,
    candidates_result,
    error_result,
    prepares,
    presents,
    rows_result,
    run_turn,
    searches,
)

from fluidbank_orchestrator.agent.nodes import MAX_TOOL_TURNS

_ACCOUNTS = [
    {"id": "checking", "account_type": "checking", "currency": "MXN", "available_balance": 100},
    {"id": "savings", "account_type": "savings", "currency": "MXN", "available_balance": 50},
]
_TRANSACTIONS = [
    {
        "id": "t1",
        "amount": 20,
        "direction": "debit",
        "category": "dining",
        "merchant": "Café",
        "occurred_at": "2026-09-11T15:00:00+00:00",
    }
]


# --------------------------------------------------------------------------
# No deterministic router
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_same_query_takes_whichever_path_the_model_chooses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proof there is no phrase table: one query, three different outcomes."""
    query = "¿Cuánto dinero tengo?"

    data_executor = Recorder(results={"get_accounts": rows_result(_ACCOUNTS, "accounts")})
    data_result = await run_turn(
        monkeypatch,
        query,
        ScriptedModel(calls(("get_accounts", {"request": {}})), presents("financial-summary")),
        data_executor,
    )
    assert data_executor.names == ["get_accounts"]
    assert data_result["financial_presentation"].intent == "financial-summary"

    chat_executor = Recorder()
    chat_result = await run_turn(
        monkeypatch, query, ScriptedModel(answers("Te explico cómo verlo.")), chat_executor
    )
    assert chat_executor.names == []
    assert chat_result.get("financial_presentation") is None
    assert chat_result["message"] == "Te explico cómo verlo."

    form_executor = Recorder()
    form_result = await run_turn(
        monkeypatch, query, ScriptedModel(prepares("budget.create")), form_executor
    )
    assert form_executor.names == ["a2ui_form"]
    assert form_result.get("financial_presentation") is None


@pytest.mark.asyncio
async def test_no_module_maps_a_phrase_to_a_tool_any_more() -> None:
    """The deterministic routers are gone, not merely unused."""
    for module in (
        "fluidbank_orchestrator.agent.retrieval",
        "fluidbank_orchestrator.api.query_routing",
        "fluidbank_orchestrator.a2ui_actions.routing",
    ):
        with pytest.raises(ModuleNotFoundError):
            __import__(module)

    from fluidbank_orchestrator.services import financial_presentation

    assert not hasattr(financial_presentation, "classify_financial_request")


# --------------------------------------------------------------------------
# Discovery: candidates in, 1..N selections out
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_single_useful_candidate_is_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    executor = Recorder(
        results={
            "search_tools": candidates_result("get_transactions"),
            "get_transactions": rows_result(_TRANSACTIONS, "transactions"),
        }
    )
    model = ScriptedModel(
        searches("movimientos recientes"),
        calls(("get_transactions", {"request": {"period": "current_month"}})),
        presents("transactions"),
    )
    result = await run_turn(monkeypatch, "Muéstrame mis movimientos", model, executor)

    assert executor.names == ["search_tools", "get_transactions"]
    assert result["financial_presentation"].intent == "transactions"


@pytest.mark.asyncio
async def test_several_candidates_are_all_offered_and_one_may_be_enough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multiple candidates must never be collapsed, and never forced either."""
    executor = Recorder(
        results={
            "search_tools": candidates_result(
                "analyze_spending", "get_transactions", "get_cash_flow", "get_budget_progress"
            ),
            "analyze_spending": rows_result(_TRANSACTIONS, "transactions"),
        }
    )
    model = ScriptedModel(
        searches("gasto por categoría"),
        calls(("analyze_spending", {"request": {"period": "current_month"}})),
        presents("spending-analysis"),
    )
    await run_turn(monkeypatch, "¿En qué gasté este mes?", model, executor)

    offered = model.seen[1]["observations"][0]["data"]["result"]
    assert [entry["name"] for entry in offered] == [
        "analyze_spending",
        "get_transactions",
        "get_cash_flow",
        "get_budget_progress",
    ]
    # Every candidate keeps the schema the model needs to evaluate it.
    assert all("inputSchema" in entry for entry in offered)
    # One was sufficient, so exactly one ran. Nothing forced a second.
    assert executor.names == ["search_tools", "analyze_spending"]


@pytest.mark.asyncio
async def test_the_model_may_select_several_tools_in_one_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A budget suggested from real spending genuinely needs two capabilities."""
    executor = Recorder(
        results={
            "search_tools": candidates_result(
                "analyze_spending", "get_budget_progress", "get_accounts"
            ),
            "analyze_spending": rows_result(_TRANSACTIONS, "transactions"),
            "get_budget_progress": rows_result([{"id": "b1", "category": "dining"}], "budgets"),
        }
    )
    model = ScriptedModel(
        searches("gasto en comida y presupuesto"),
        calls(
            ("analyze_spending", {"request": {"period": "current_month"}}),
            ("get_budget_progress", {"request": {"category": "dining"}}),
        ),
        prepares("budget.create"),
    )
    await run_turn(
        monkeypatch, "Crea un presupuesto de comida según lo que suelo gastar", model, executor
    )

    assert executor.names == [
        "search_tools",
        "analyze_spending",
        "get_budget_progress",
        "a2ui_form",
    ]
    # Both results reached the model before it prepared the action.
    observed = {entry["name"] for entry in model.seen[2]["observations"]}
    assert {"analyze_spending", "get_budget_progress"} <= observed


@pytest.mark.asyncio
async def test_irrelevant_candidates_do_not_become_a_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = Recorder(results={"search_tools": candidates_result("get_beneficiaries")})
    model = ScriptedModel(
        searches("clima de mañana"),
        answers("Eso no es algo que pueda consultar en tu banca."),
    )
    result = await run_turn(monkeypatch, "¿Va a llover mañana?", model, executor)

    assert executor.names == []
    assert model.seen == []
    assert result.get("financial_presentation") is None
    assert result["policy_reason"] == "out_of_scope"
    assert result["message"] == (
        "Solo puedo ayudar con consultas y operaciones bancarias permitidas."
    )


@pytest.mark.asyncio
async def test_no_candidate_at_all_still_ends_the_turn_without_figures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = Recorder(results={"search_tools": candidates_result()})
    model = ScriptedModel(searches("algo inexistente"), answers("No encontré cómo consultarlo."))
    result = await run_turn(monkeypatch, "Consulta un dato bancario inexistente", model, executor)

    assert executor.names == ["search_tools"]
    assert result["message"] == "No encontré cómo consultarlo."
    assert result.get("financial_presentation") is None


@pytest.mark.asyncio
async def test_a_second_discovery_round_is_allowed_when_the_first_was_useless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = Recorder(
        results={
            "search_tools": candidates_result("get_debt_overview"),
            "get_debt_overview": rows_result([{"id": "d1", "balance": 10}], "debts"),
        }
    )
    model = ScriptedModel(
        searches("pagos que vienen"),
        searches("deudas pendientes"),
        calls(("get_debt_overview", {"request": {}})),
        presents("debts"),
    )
    await run_turn(monkeypatch, "¿Cuánto debo?", model, executor)

    assert executor.names == ["search_tools", "search_tools", "get_debt_overview"]


@pytest.mark.asyncio
async def test_repeating_the_identical_search_is_skipped_not_re_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second round must differ; an identical one buys nothing."""
    executor = Recorder(results={"search_tools": candidates_result("get_debt_overview")})
    model = ScriptedModel(
        searches("deudas pendientes"),
        searches("deudas pendientes"),
        answers("Ya tenía esos resultados."),
    )
    await run_turn(monkeypatch, "¿Cuánto debo?", model, executor)

    assert executor.names == ["search_tools"]


# --------------------------------------------------------------------------
# Results return to the model; failures are readable
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_results_reach_the_model_before_any_surface_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = Recorder(results={"get_accounts": rows_result(_ACCOUNTS, "accounts")})
    model = ScriptedModel(calls(("get_accounts", {"request": {}})), presents("financial-summary"))
    result = await run_turn(monkeypatch, "¿Cuánto tengo?", model, executor)

    assert model.seen[0]["observations"] == []
    read = model.seen[1]["observations"]
    assert [entry["name"] for entry in read] == ["get_accounts"]
    assert read[0]["data"]["accounts"] == _ACCOUNTS
    assert result["financial_presentation"].data["owned_balance"] == 150


@pytest.mark.asyncio
async def test_a_tool_failure_is_handed_back_as_structured_information(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = Recorder(results={"get_accounts": error_result("La consulta excedió el tiempo.")})
    model = ScriptedModel(
        calls(("get_accounts", {"request": {}})),
        presents("financial-summary"),
    )
    result = await run_turn(monkeypatch, "¿Cuánto tengo?", model, executor)

    failure = model.seen[1]["observations"][0]
    assert failure["is_error"] is True
    assert failure["data"]["error"]["code"] == "database_timeout"
    assert failure["data"]["error"]["retryable"] is True
    # A failed read renders an explicit empty view, never an estimate.
    view = result["financial_presentation"].a2ui.messages[-1]["updateDataModel"]["value"]["view"]
    assert view.get("accounts", []) == []


@pytest.mark.asyncio
async def test_the_loop_is_bounded_even_if_the_model_keeps_asking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = Recorder(results={"search_tools": candidates_result("get_accounts")})
    model = ScriptedModel(*[searches(f"intento {index}") for index in range(MAX_TOOL_TURNS + 4)])
    result = await run_turn(monkeypatch, "Busca datos bancarios sin parar", model, executor)

    assert result["tool_loop_count"] == MAX_TOOL_TURNS
    assert "límite seguro de pasos" in result["message"]


# --------------------------------------------------------------------------
# What is still not the model's choice
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_hallucinated_presentation_intent_produces_no_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fluidbank_orchestrator.agent.model import ModelTurn

    executor = Recorder()
    model = ScriptedModel(ModelTurn(message="Listo.", presentation_intent="crypto-portfolio"))  # type: ignore[arg-type]
    result = await run_turn(monkeypatch, "Muéstrame mi cripto", model, executor)

    assert result.get("financial_presentation") is None
    assert result["message"] == "Listo."


@pytest.mark.asyncio
async def test_an_approved_action_pins_the_view_the_model_cannot_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`requested_intent` is trusted input and outranks the model's choice."""
    from scripted_model import Recorder as _Recorder
    from scripted_model import discovery_tools, graph_for

    executor = _Recorder(results={"get_accounts": rows_result(_ACCOUNTS, "accounts")})
    model = ScriptedModel(presents("spending-analysis"))
    graph = graph_for(monkeypatch, model, executor, context_rows={"accounts": _ACCOUNTS})
    result: dict[str, Any] = await graph.ainvoke(
        {
            "user_query": "request_financial_view:financial-summary",
            "requested_intent": "financial-summary",
            "action_requested": True,
            "current_user_id": USER_A,
        }
    )

    assert discovery_tools is not None
    assert result["financial_presentation"].intent == "financial-summary"


@pytest.mark.asyncio
async def test_a_missing_user_context_answers_without_figures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripted_model import discovery_tools

    import fluidbank_orchestrator.agent.nodes as nodes_module
    from fluidbank_orchestrator.graph import build_graph
    from fluidbank_orchestrator.mcp_client import UserContextError

    async def broken(_current_user_id: Any) -> Any:
        raise UserContextError("unavailable")

    monkeypatch.setattr(nodes_module, "fetch_user_context", broken)
    executor = Recorder()
    model = ScriptedModel(calls(("get_accounts", {"request": {}})))
    result: dict[str, Any] = await build_graph(
        model=model, tool_loader=discovery_tools, tool_executor=executor
    ).ainvoke({"user_query": "¿Cuánto tengo?", "current_user_id": USER_A})

    # The model is never consulted, and no tool runs, when identity is unverified.
    assert model.seen == []
    assert executor.names == []
    assert "no puedo mostrarte cifras" in result["message"]
