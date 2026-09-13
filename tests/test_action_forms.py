"""Action templates and authenticated routing, without network or an LLM."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from mcp.types import TextContent

from fluidbank_orchestrator import api
from fluidbank_orchestrator.a2ui_actions.routing import requested_form, requested_form_arguments
from fluidbank_orchestrator.mcp_client import SCOPED_TOOL_NAMES, enforce_trusted_user_scope
from fluidbank_orchestrator.schemas.a2ui import (
    A2UI_BASIC_CATALOG,
    validate_complete_sequence,
    validate_dynamic_updates,
)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Crea un presupuesto", "budget.create"),
        ("Quiero editar mi presupuesto", "budget.load"),
        ("Crear una meta de ahorro", "savings_goal.create"),
        ("Actualiza mi meta de ahorro", "savings_goal.load"),
        ("Transfiere $500 a Ana", "transfer.execute"),
        ("Quiero hacer una transferencia", "transfer.execute"),
        ("Mueve dinero a mi cuenta de ahorro", "transfer.execute"),
        ("Quiero pagar mi tarjeta de crédito", "credit_card.pay"),
        ("¿Cuándo debo pagar mi tarjeta?", None),
        ("¿Cómo crear un presupuesto?", None),
        ("No quiero crear una meta de ahorro", None),
    ],
)
def test_form_routing_only_prepares_forms(query, expected):
    assert requested_form(query) == expected
    # `a2ui_action` is never a model-facing tool. Under progressive discovery
    # that is enforced on the server, which declares it app-only so neither
    # search nor the `call_tool` proxy can reach it; here the invariant that
    # remains client-side is that it is always scoped to the authenticated user.
    assert "a2ui_action" in SCOPED_TOOL_NAMES


def test_transfer_form_prefills_explicit_amount_and_recipient():
    assert requested_form_arguments("Transfiere $1,250.50 a Ana", "transfer.execute") == {
        "initial_amount": 1250.5,
        "initial_recipient": "Ana",
    }
    assert requested_form_arguments("Transfiere $1250 a Ana", "transfer.execute") == {
        "initial_amount": 1250.0,
        "initial_recipient": "Ana",
    }


def test_all_forms_pass_official_sdk_and_agent_validation():
    root = Path(__file__).resolve().parents[2]
    mcp_root = root / "hackmty2026-mcp"
    registry = json.loads(
        (mcp_root / "src/supabase_mcp/a2ui_actions/actions.json").read_text()
    )
    for action in registry["actions"]:
        messages = json.loads(
            (
                mcp_root
                / "src/supabase_mcp/a2ui_support/templates"
                / f"{action['surfaceId']}.json"
            ).read_text()
        )
        model = {field["key"]: field["default"] for field in action["inputs"]}
        data_model = {"form": model, "help": "Revisa y confirma"}
        dynamic_start = len(messages)
        if action["name"] == "transfer.execute":
            components = deepcopy(messages[1]["updateComponents"]["components"])
            for component in components:
                if component["id"] == "source_account":
                    component["options"] = [
                        {"label": "Cuenta principal · •••• 1111", "value": "Cuenta principal"}
                    ]
                elif component["id"] == "recipient":
                    component["options"] = [
                        {"label": "Ana · Banco receptor · •••• 4321", "value": "Ana"}
                    ]
            messages.append(
                {
                    "version": "v0.9.1",
                    "updateComponents": {
                        "surfaceId": action["surfaceId"],
                        "components": components,
                    },
                }
            )
        if action.get("preview"):
            data_model["preview"] = {
                "intent": "credit-card",
                "title": "Tu tarjeta",
                "currency": "MXN",
                "cardName": "Tarjeta oro",
                "lastFour": "1234",
                "debt": 5000,
                "availableCredit": 5000,
                "minimumPayment": 300,
                "interestFreePayment": 2000,
                "dueDate": "2026-10-01",
            }
        messages.append(
            {
                "version": "v0.9.1",
                "updateDataModel": {
                    "surfaceId": action["surfaceId"],
                    "value": data_model,
                },
            }
        )
        validate_complete_sequence(messages, action["surfaceId"])
        if action["name"] == "transfer.execute":
            validate_dynamic_updates(
                messages[dynamic_start:],
                expected_surface_id=action["surfaceId"],
                catalog_id=A2UI_BASIC_CATALOG,
            )


@pytest.mark.asyncio
async def test_form_request_routes_to_mcp_with_verified_user(monkeypatch):
    uid = UUID("f52827d7-0213-4df4-9621-14775d6228d4")
    called = []

    async def execute(name, arguments, *, current_user_id):
        called.append((name, arguments, current_user_id))
        return SimpleNamespace(
            result=SimpleNamespace(content=[TextContent(text="Formulario")], structured_content={}),
            a2ui=None,
        )

    monkeypatch.setattr(api, "execute_remote_tool", execute)
    result = await api._route_request(None, "Crea un presupuesto", uid)
    assert result.message == "Formulario"
    assert called == [("a2ui_form", {"name": "budget.create"}, uid)]


@pytest.mark.asyncio
async def test_transfer_request_passes_only_bounded_prefill(monkeypatch):
    uid = UUID("f52827d7-0213-4df4-9621-14775d6228d4")
    called = []

    async def execute(name, arguments, *, current_user_id):
        called.append((name, arguments, current_user_id))
        return SimpleNamespace(
            result=SimpleNamespace(content=[TextContent(text="Formulario")], structured_content={}),
            a2ui=None,
        )

    monkeypatch.setattr(api, "execute_remote_tool", execute)
    await api._route_request(None, "Transfiere $500 a Ana", uid)
    assert called == [
        (
            "a2ui_form",
            {"name": "transfer.execute", "initial_amount": 500.0, "initial_recipient": "Ana"},
            uid,
        )
    ]


def test_form_tool_scope_is_replaced_by_authenticated_subject():
    uid = UUID("f52827d7-0213-4df4-9621-14775d6228d4")
    scoped = enforce_trusted_user_scope(
        "a2ui_form", {"name": "budget.create", "trustedScope": {"user_id": "spoofed"}}, uid
    )
    assert scoped["trustedScope"] == {"user_id": str(uid)}


def test_write_proof_is_derived_server_side_and_not_from_model(monkeypatch):
    import hashlib
    import hmac

    secret = "test-only-secret-for-offline-tests-12345"
    monkeypatch.setenv("MCP_ACTIONS_SECRET", secret)
    uid = UUID("f52827d7-0213-4df4-9621-14775d6228d4")
    raw = {
        "name": "budget.create",
        "surfaceId": "budget-create",
        "sourceComponentId": "submit",
        "timestamp": "2026-09-13T12:00:00Z",
        "context": {"name": "Comida"},
        "actionProof": "forged",
    }
    scoped = enforce_trusted_user_scope("a2ui_action", raw, uid)
    payload = {
        key: raw[key] for key in ("name", "surfaceId", "sourceComponentId", "timestamp", "context")
    }
    payload["user_id"] = str(uid)
    expected = hmac.new(
        secret.encode(),
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(),
        hashlib.sha256,
    ).hexdigest()
    assert scoped["actionProof"] == expected
    assert raw["actionProof"] == "forged"
