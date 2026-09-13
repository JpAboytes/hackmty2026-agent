"""Action templates and authenticated routing, without network or an LLM."""

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from mcp.types import TextContent

from fluidbank_orchestrator.a2ui_actions.forms import (
    ACTION_FORM_NAMES,
    declared_action_names,
    normalize_form_name,
)
from fluidbank_orchestrator.agent.nodes import make_prepare_action_node
from fluidbank_orchestrator.mcp_client import SCOPED_TOOL_NAMES, enforce_trusted_user_scope
from fluidbank_orchestrator.schemas.a2ui import validate_complete_sequence


@pytest.mark.parametrize(
    ("chosen", "expected"),
    [
        ("budget.create", "budget.create"),
        ("budget.load", "budget.load"),
        ("savings_goal.create", "savings_goal.create"),
        ("savings_goal.load", "savings_goal.load"),
        # A form the model invented, one it may not prepare directly, and
        # non-string junk all collapse to "no form" rather than reaching MCP.
        ("budget.delete", None),
        ("budget.update", None),
        ("Crea un presupuesto", None),
        (None, None),
        (17, None),
    ],
)
def test_only_the_declared_form_vocabulary_reaches_mcp(chosen, expected):
    assert normalize_form_name(chosen) == expected


def test_preparable_forms_are_a_subset_of_the_shared_contract():
    """The model selects from names the contract actually declares.

    `.update` is absent on purpose: MCP derives an update form from the
    corresponding `.load`, so it is a contract action but not a preparable one.
    """
    declared = declared_action_names()
    assert set(ACTION_FORM_NAMES) < declared
    assert {name for name in declared if name.endswith(".update")}.isdisjoint(ACTION_FORM_NAMES)


def test_action_handler_is_never_a_model_facing_tool():
    # `a2ui_action` is never a model-facing tool. Under progressive discovery
    # that is enforced on the server, which declares it app-only so neither
    # search nor the `call_tool` proxy can reach it; here the invariant that
    # remains client-side is that it is always scoped to the authenticated user.
    assert "a2ui_action" in SCOPED_TOOL_NAMES


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
async def test_preparing_a_form_calls_mcp_with_the_verified_user_and_saves_nothing():
    """Preparing is a read: `a2ui_form` is the only tool the node may call."""
    uid = UUID("f52827d7-0213-4df4-9621-14775d6228d4")
    called = []

    async def execute(name, arguments, *, current_user_id):
        called.append((name, arguments, current_user_id))
        return SimpleNamespace(
            result=SimpleNamespace(
                content=[TextContent(text="Formulario")],
                structured_content={},
                is_error=False,
            ),
            a2ui=None,
        )

    node = make_prepare_action_node(execute)
    result = await node({"action_form": "budget.create", "current_user_id": uid})

    assert result["message"] == "Formulario"
    assert called == [("a2ui_form", {"name": "budget.create"}, uid)]
    assert not any(name == "a2ui_action" for name, _, _ in called)


@pytest.mark.asyncio
async def test_a_form_name_the_model_invented_never_reaches_mcp():
    called = []

    async def execute(name, arguments, *, current_user_id):
        called.append(name)
        raise AssertionError("MCP must not be called for an unknown form")

    node = make_prepare_action_node(execute)
    result = await node(
        {
            "action_form": "budget.delete",
            "current_user_id": UUID("f52827d7-0213-4df4-9621-14775d6228d4"),
        }
    )

    assert called == []
    assert "no está disponible" in result["message"]
    assert "final_tool_execution" not in result


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
