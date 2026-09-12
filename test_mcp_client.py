"""Focused tests for remote MCP configuration; no network calls are made."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mcp_client import MCPConfigurationError, create_mcp_client, load_mcp_config

_URL = "https://example.fastmcp.app/mcp"
_TOKEN = "fmcp_test_placeholder_not_a_real_key"


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

    with patch("mcp_client.Client") as client_class:
        created = create_mcp_client(config)

    assert created is client_class.return_value
    client_class.assert_called_once_with("http://localhost:8000/mcp")
    assert _TOKEN not in repr(config)


def test_horizon_client_uses_expected_remote_url_and_raw_token() -> None:
    config = load_mcp_config({"MCP_SERVER_URL": _URL, "HORIZON_API_KEY": _TOKEN})

    with patch("mcp_client.Client") as client_class:
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
