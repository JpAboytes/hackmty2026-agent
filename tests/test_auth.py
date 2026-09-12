"""Focused Supabase authentication-boundary tests."""

from __future__ import annotations

import pytest

from fluidbank_orchestrator.auth import (
    AuthenticationError,
    _authenticated_subject,
    _bearer_token,
    load_supabase_auth_config,
)

USER_ID = "68dc4d66-07b8-5893-95f1-07f06989a552"


def test_auth_config_requires_secure_supabase_url_and_public_key() -> None:
    config = load_supabase_auth_config(
        {
            "SUPABASE_URL": "https://project.supabase.co/",
            "SUPABASE_PUBLISHABLE_KEY": "public-test-key",
        }
    )
    assert config.url == "https://project.supabase.co"
    with pytest.raises(AuthenticationError):
        load_supabase_auth_config(
            {"SUPABASE_URL": "http://project.supabase.co", "SUPABASE_ANON_KEY": "key"}
        )


def test_bearer_header_is_strict() -> None:
    assert _bearer_token("Bearer header.payload.signature") == "header.payload.signature"
    for value in (None, "token", "Basic token", "Bearer two tokens"):
        with pytest.raises(AuthenticationError):
            _bearer_token(value)


def test_authenticated_subject_rejects_anonymous_or_unconfirmed_accounts() -> None:
    assert (
        _authenticated_subject(
            {"id": USER_ID, "is_anonymous": False, "email_confirmed_at": "2026-01-01"}
        )
        == USER_ID
    )
    with pytest.raises(AuthenticationError):
        _authenticated_subject(
            {"id": USER_ID, "is_anonymous": True, "email_confirmed_at": "2026-01-01"}
        )
    with pytest.raises(AuthenticationError):
        _authenticated_subject({"id": USER_ID, "is_anonymous": False})
