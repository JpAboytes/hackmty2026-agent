"""The coarse lifecycle phases a turn reports, and what they must never carry.

Two separate promises are tested here. The first is that the phases a client
sees describe the shape of the turn: a data request, a direct action and an
action needing data each report a different, recognisable sequence. The second
is that a phase is *only* an identifier - no prompt, no reasoning, no tool
arguments and no rows can travel on this channel, whatever a node writes.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from scripted_model import (
    Recorder,
    ScriptedModel,
    answers,
    calls,
    candidates_result,
    prepares,
    presents,
    rows_result,
    searches,
    statuses_for,
)

from fluidbank_orchestrator.agent.status import (
    AGENT_STATUS_EVENT,
    AGENT_STATUSES,
    emit_status,
    status_event,
)
from fluidbank_orchestrator.api import _status_line

_ACCOUNTS = [
    {"id": "checking", "account_type": "checking", "currency": "MXN", "available_balance": 150},
]
_SPEND = [
    {
        "id": "t1",
        "amount": 20,
        "direction": "debit",
        "category": "dining",
        "merchant": "Café",
        "occurred_at": "2026-09-11T15:00:00+00:00",
    }
]


def _ids(payloads: list[dict[str, Any]]) -> list[str]:
    return [payload["status"] for payload in payloads]


# --------------------------------------------------------------------------
# The vocabulary
# --------------------------------------------------------------------------


def test_the_vocabulary_is_exactly_the_eight_documented_phases() -> None:
    assert AGENT_STATUSES == {
        "interpreting",
        "discovering_tools",
        "selecting_tools",
        "executing_tools",
        "interpreting_results",
        "preparing_action",
        "building_ui",
        "validating_ui",
    }


def test_an_event_carries_a_discriminator_and_a_status_and_nothing_else() -> None:
    assert status_event("executing_tools") == {
        "type": "agent_status",
        "status": "executing_tools",
    }
    assert AGENT_STATUS_EVENT == "agent_status"


def test_a_status_outside_the_vocabulary_cannot_be_built() -> None:
    with pytest.raises(ValueError):
        status_event("thinking_about_the_users_debt")  # type: ignore[arg-type]


def test_emitting_outside_a_stream_is_a_no_op_not_a_failure() -> None:
    """The plain POST route runs the same nodes without a writer attached."""
    emit_status("interpreting")


# --------------------------------------------------------------------------
# The sequences a client actually sees
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_data_request_reports_discovery_selection_execution_then_ui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = Recorder(
        results={
            "search_tools": candidates_result("get_accounts", "get_financial_overview"),
            "get_accounts": rows_result(_ACCOUNTS, "accounts"),
        }
    )
    model = ScriptedModel(
        # interpreting
        searches("saldo disponible"),
        # selecting_tools, having read the candidates
        calls(("get_accounts", {"request": {}})),
        # interpreting_results, then the view
        presents("financial-summary"),
    )
    payloads, final = await statuses_for(monkeypatch, "¿Cuánto dinero tengo?", model, executor)

    assert _ids(payloads) == [
        "interpreting",
        "interpreting",
        "discovering_tools",
        "selecting_tools",
        "executing_tools",
        "interpreting_results",
        "building_ui",
        "validating_ui",
    ]
    assert final["financial_presentation"].intent == "financial-summary"


@pytest.mark.asyncio
async def test_a_direct_action_skips_discovery_and_reports_preparing_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = Recorder(default=rows_result([]))
    model = ScriptedModel(prepares("budget.create"))
    payloads, _final = await statuses_for(
        monkeypatch, "Crea un presupuesto", model, executor
    )

    ids = _ids(payloads)
    assert "discovering_tools" not in ids
    assert "executing_tools" not in ids
    assert ids[-2:] == ["preparing_action", "building_ui"]
    assert ids[0] == "interpreting"


@pytest.mark.asyncio
async def test_an_action_that_needs_data_reports_both_halves_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = Recorder(
        results={
            "search_tools": candidates_result("analyze_spending", "get_budget_progress"),
            "analyze_spending": rows_result(_SPEND, "transactions"),
        }
    )
    model = ScriptedModel(
        searches("gasto en comida"),
        calls(("analyze_spending", {"request": {"period": "current_month"}})),
        prepares("budget.create"),
    )
    payloads, _final = await statuses_for(
        monkeypatch,
        "Crea un presupuesto de comida según lo que suelo gastar",
        model,
        executor,
    )

    ids = _ids(payloads)
    for earlier, later in (
        ("discovering_tools", "selecting_tools"),
        ("selecting_tools", "executing_tools"),
        ("executing_tools", "interpreting_results"),
        ("interpreting_results", "preparing_action"),
        ("preparing_action", "building_ui"),
    ):
        assert ids.index(earlier) < ids.index(later), ids


@pytest.mark.asyncio
async def test_a_conversational_turn_reports_no_ui_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not every request passes through every phase."""
    payloads, _final = await statuses_for(
        monkeypatch, "¿Qué es una tasa de interés?", ScriptedModel(answers("Es…")), Recorder()
    )

    ids = _ids(payloads)
    assert set(ids) == {"interpreting"}
    assert "building_ui" not in ids
    assert "validating_ui" not in ids


# --------------------------------------------------------------------------
# What may never travel on this channel
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_phase_ever_carries_reasoning_arguments_or_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = Recorder(
        results={
            "search_tools": candidates_result("get_accounts"),
            "get_accounts": rows_result(_ACCOUNTS, "accounts"),
        }
    )
    model = ScriptedModel(
        searches("saldo disponible"),
        calls(("get_accounts", {"request": {"secret_note": "internal analysis"}})),
        presents("financial-summary", message="Tienes 150.00 MXN."),
    )
    payloads, _final = await statuses_for(monkeypatch, "¿Cuánto dinero tengo?", model, executor)

    for payload in payloads:
        assert set(payload) == {"type", "status"}
        assert payload["type"] == "agent_status"
        assert payload["status"] in AGENT_STATUSES

    serialized = json.dumps(payloads, ensure_ascii=False)
    for leak in ("internal analysis", "150", "checking", "saldo disponible", "Tienes"):
        assert leak not in serialized


def test_the_http_boundary_drops_anything_that_is_not_a_known_status() -> None:
    """The route allowlists; a node cannot widen the stream by writing more."""
    assert _status_line({"type": "agent_status", "status": "building_ui"}) == (
        '{"type":"agent_status","status":"building_ui"}'
    )
    # A richer payload is reduced to the two fields, never forwarded verbatim.
    assert _status_line(
        {"type": "agent_status", "status": "building_ui", "thought": "the user is overdrawn"}
    ) == '{"type":"agent_status","status":"building_ui"}'
    # Anything else is not a lifecycle event at all.
    assert _status_line({"type": "reasoning", "text": "step 1…"}) is None
    assert _status_line({"type": "agent_status", "status": "pondering"}) is None
    assert _status_line({"status": "building_ui"}) is None
