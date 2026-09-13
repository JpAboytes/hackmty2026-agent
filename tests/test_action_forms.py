"""Action templates and authenticated routing, without network or an LLM."""

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from mcp.types import TextContent

from fluidbank_orchestrator import api
from fluidbank_orchestrator.a2ui_actions.routing import requested_form
from fluidbank_orchestrator.mcp_client import MODEL_TOOL_NAMES, enforce_trusted_user_scope
from fluidbank_orchestrator.schemas.a2ui import validate_complete_sequence


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Crea un presupuesto", "budget.create"),
        ("Quiero editar mi presupuesto", "budget.load"),
        ("Crear una meta de ahorro", "savings_goal.create"),
        ("Actualiza mi meta de ahorro", "savings_goal.load"),
        ("¿Cómo crear un presupuesto?", None),
        ("No quiero crear una meta de ahorro", None),
    ],
)
def test_form_routing_only_prepares_forms(query, expected):
    assert requested_form(query) == expected
    assert "a2ui_action" not in MODEL_TOOL_NAMES


def test_all_forms_pass_official_sdk_and_agent_validation():
    root = Path(__file__).resolve().parents[2]
    registry = json.loads(
        (root / "hackmty2026-mcp/src/supabase_mcp/a2ui_actions/actions.json").read_text()
    )
    for action in registry["actions"]:
        messages = json.loads(
            (
                root
                / "hackmty2026-mcp/src/supabase_mcp/a2ui_support/templates"
                / f"{action['surfaceId']}.json"
            ).read_text()
        )
        model = {field["key"]: field["default"] for field in action["inputs"]}
        messages.append(
            {
                "version": "v0.9.1",
                "updateDataModel": {
                    "surfaceId": action["surfaceId"],
                    "value": {"form": model, "help": "Revisa y confirma"},
                },
            }
        )
        validate_complete_sequence(messages, action["surfaceId"])


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
