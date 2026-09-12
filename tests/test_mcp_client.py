"""Focused tests for remote MCP configuration; no network calls are made."""

from __future__ import annotations

from unittest.mock import patch
from uuid import UUID

import pytest
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent

from fluidbank_orchestrator import mcp_client
from fluidbank_orchestrator.mcp_client import (
    MODEL_TOOL_NAMES,
    MCPConfig,
    MCPConfigurationError,
    TrustedUserScopeError,
    UserContextError,
    call_mcp_tool,
    create_mcp_client,
    fetch_user_context,
    load_mcp_config,
)

_URL = "https://example.fastmcp.app/mcp"
_TOKEN = "fmcp_test_placeholder_not_a_real_key"


def test_chat_message_is_not_a_model_escape_hatch() -> None:
    assert "chat_message" not in MODEL_TOOL_NAMES


def test_horizon_mode_rejects_missing_api_key() -> None:
    with pytest.raises(MCPConfigurationError, match="HORIZON_API_KEY is required"):
        load_mcp_config({"MCP_SERVER_URL": _URL, "MCP_AUTH_MODE": "horizon"})


def test_horizon_mode_rejects_malformed_api_key_without_echoing_it() -> None:
    malformed = "definitely-not-a-horizon-key"
    with pytest.raises(MCPConfigurationError) as caught:
        load_mcp_config(
            {
                "MCP_SERVER_URL": _URL,
                "MCP_AUTH_MODE": "horizon",
                "HORIZON_API_KEY": malformed,
            }
        )

    assert malformed not in str(caught.value)


def test_horizon_mode_rejects_missing_url() -> None:
    with pytest.raises(MCPConfigurationError, match="MCP_SERVER_URL is required"):
        load_mcp_config({"MCP_AUTH_MODE": "horizon", "HORIZON_API_KEY": _TOKEN})


def test_horizon_mode_accepts_fmcp_key_and_redacts_config() -> None:
    config = load_mcp_config(
        {
            "MCP_SERVER_URL": "https://example.fastmcp.app/",
            "MCP_AUTH_MODE": "horizon",
            "HORIZON_API_KEY": _TOKEN,
        }
    )

    assert config.url == _URL
    assert config.auth_mode == "horizon"
    assert _TOKEN not in repr(config)
    assert "<redacted>" in repr(config)


def test_none_mode_does_not_require_or_attach_token() -> None:
    config = load_mcp_config(
        {
            "MCP_SERVER_URL": "http://localhost:8000",
            "MCP_AUTH_MODE": "none",
            "HORIZON_API_KEY": _TOKEN,
        }
    )

    with patch("fluidbank_orchestrator.mcp_client.Client") as client_class:
        created = create_mcp_client(config)

    assert created is client_class.return_value
    client_class.assert_called_once_with("http://localhost:8000/mcp")
    assert _TOKEN not in repr(config)


def test_horizon_client_uses_expected_remote_url_and_raw_token() -> None:
    config = load_mcp_config({"MCP_SERVER_URL": _URL, "HORIZON_API_KEY": _TOKEN})

    with patch("fluidbank_orchestrator.mcp_client.Client") as client_class:
        created = create_mcp_client(config)

    assert created is client_class.return_value
    client_class.assert_called_once_with(_URL, auth=_TOKEN)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.fastmcp.app/mcp?debug=true",
        "https://example.fastmcp.app/not-mcp",
        "example.fastmcp.app/mcp",
    ],
)
def test_invalid_mcp_endpoint_is_rejected(url: str) -> None:
    with pytest.raises(MCPConfigurationError):
        load_mcp_config({"MCP_SERVER_URL": url, "HORIZON_API_KEY": _TOKEN})


class _FakeClient:
    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


@pytest.mark.asyncio
async def test_fetch_user_context_scopes_every_selection_to_user_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MCPConfig(url=_URL, auth_mode="horizon", _horizon_api_key=_TOKEN)
    user_id = UUID("11111111-1111-1111-1111-111111111111")
    calls: list[tuple[str, UUID]] = []

    async def fake_select(
        client: _FakeClient,
        server_identity: str,
        table: str,
        current_user_id: UUID,
    ) -> list[dict[str, object]]:
        del client
        assert server_identity == _URL
        calls.append((table, current_user_id))
        rows: dict[str, list[dict[str, object]]] = {
            "users": [{"id": str(user_id)}],
            "accessibility_preferences": [
                {
                    "literacy_level": "standard",
                    "font_scale": "1.0",
                    "contrast": "standard",
                    "hit_target": "standard",
                }
            ],
            "accounts": [
                {
                    "account_type": "checking",
                    "currency": "MXN",
                    "available_balance": "1250.50",
                },
                {
                    "account_type": "credit",
                    "currency": "MXN",
                    "available_balance": "9000.00",
                },
            ],
            "subscriptions": [{"amount": "19.99", "status": "active"}],
        }
        return rows[table]

    monkeypatch.setattr(mcp_client, "load_mcp_config", lambda: config)
    monkeypatch.setattr(mcp_client, "create_mcp_client", lambda _config: _FakeClient())
    monkeypatch.setattr(mcp_client, "_select", fake_select)

    profile = (await fetch_user_context(user_id)).profile

    assert calls == [
        ("users", user_id),
        ("accessibility_preferences", user_id),
        ("accounts", user_id),
        ("subscriptions", user_id),
    ]
    assert profile["available_balance"] == 1250.50
    assert profile["owned_balances"] == {"MXN": 1250.50}
    assert profile["recurring_expenses"] == 19.99


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e10000"])
def test_float_value_rejects_non_finite_values(value: str) -> None:
    with pytest.raises(UserContextError, match="invalid"):
        mcp_client._float_value({"amount": value}, "amount")


@pytest.mark.asyncio
async def test_a_new_user_without_preferences_keeps_its_own_balances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh account has no preferences row, which must not discard its data."""
    config = MCPConfig(url=_URL, auth_mode="horizon", _horizon_api_key=_TOKEN)
    user_id = UUID("c72428ad-ebaf-4709-b832-2c0f5094d685")

    async def fake_select(
        client: _FakeClient,
        server_identity: str,
        table: str,
        current_user_id: UUID,
    ) -> list[dict[str, object]]:
        del client, server_identity, current_user_id
        rows: dict[str, list[dict[str, object]]] = {
            "users": [{"id": str(user_id)}],
            "accessibility_preferences": [],
            "accounts": [
                {
                    "account_type": "savings",
                    "currency": "MXN",
                    "available_balance": "42.00",
                }
            ],
            "subscriptions": [],
        }
        return rows[table]

    monkeypatch.setattr(mcp_client, "load_mcp_config", lambda: config)
    monkeypatch.setattr(mcp_client, "create_mcp_client", lambda _config: _FakeClient())
    monkeypatch.setattr(mcp_client, "_select", fake_select)

    profile = (await fetch_user_context(user_id)).profile

    assert profile["available_balance"] == 42.00
    assert profile["recurring_expenses"] == 0.0
    assert profile["literacy_level"] == "medium"
    assert profile["hit_target"] == "large"


def test_owned_money_excludes_credit_and_never_mixes_currencies() -> None:
    rows = [
        {"account_type": "checking", "currency": "MXN", "available_balance": 100},
        {"account_type": "savings", "currency": "MXN", "available_balance": 50},
        {"account_type": "credit", "currency": "MXN", "available_balance": 900},
        {"account_type": "checking", "currency": "USD", "available_balance": 20},
    ]
    assert mcp_client._owned_balances(rows) == {"MXN": 150.0, "USD": 20.0}


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object] | None]] = []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, object] | None,
        *,
        raise_on_error: bool = True,
    ) -> CallToolResult:
        assert raise_on_error is False
        self.calls.append((name, arguments))
        return CallToolResult(
            content=[TextContent(text="ok")],
            structured_content={"ok": True},
            meta=None,
            data=None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "model_arguments", "scope_path"),
    [
        (
            "select_rows",
            {
                "schema": "public",
                "table": "transactions",
                "scope": {"user_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"},
            },
            ("scope",),
        ),
        (
            "visualize_allowed_data",
            {"request": {"scope": {"user_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"}}},
            ("request", "scope"),
        ),
        (
            "a2ui_action",
            {
                "name": "refresh",
                "trustedScope": {"user_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"},
            },
            ("trustedScope",),
        ),
    ],
)
async def test_mcp_boundary_overwrites_untrusted_user_scope(
    tool_name: str,
    model_arguments: dict[str, object],
    scope_path: tuple[str, ...],
) -> None:
    current_user_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    client = _RecordingClient()

    await call_mcp_tool(  # type: ignore[arg-type]
        client,
        _URL,
        tool_name,
        model_arguments,
        current_user_id=current_user_id,
    )

    received = client.calls[0][1]
    assert received is not None
    value: object = received
    for key in scope_path:
        assert isinstance(value, dict)
        value = value[key]
    assert value == {"user_id": str(current_user_id)}


@pytest.mark.asyncio
async def test_scoped_mcp_call_without_identity_fails_closed_before_calling_client() -> None:
    client = _RecordingClient()

    with pytest.raises(TrustedUserScopeError):
        await call_mcp_tool(  # type: ignore[arg-type]
            client,
            _URL,
            "select_rows",
            {"schema": "public", "table": "transactions"},
        )

    assert client.calls == []
