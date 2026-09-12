"""Focused Supabase authentication-boundary tests."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest

from fluidbank_orchestrator import auth
from fluidbank_orchestrator.auth import (
    AuthenticationError,
    SupabaseAuthConfig,
    _authenticated_subject,
    _bearer_token,
    load_supabase_auth_config,
    verify_supabase_access_token,
)

USER_ID = UUID("68dc4d66-07b8-5893-95f1-07f06989a552")


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
            {"id": str(USER_ID), "is_anonymous": False, "email_confirmed_at": "2026-01-01"}
        )
        == USER_ID
    )
    with pytest.raises(AuthenticationError):
        _authenticated_subject(
            {"id": str(USER_ID), "is_anonymous": True, "email_confirmed_at": "2026-01-01"}
        )
    with pytest.raises(AuthenticationError):
        _authenticated_subject({"id": str(USER_ID), "is_anonymous": False})


@pytest.mark.asyncio
async def test_verified_supabase_user_returns_canonical_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, dict[str, str]]] = []

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json() -> dict[str, Any]:
            return {
                "id": str(USER_ID),
                "is_anonymous": False,
                "email_confirmed_at": "2026-01-01",
            }

    class FakeClient:
        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def get(self, url: str, *, headers: dict[str, str]) -> FakeResponse:
            requests.append((url, headers))
            return FakeResponse()

    monkeypatch.setattr(
        auth,
        "load_supabase_auth_config",
        lambda: SupabaseAuthConfig(
            url="https://project.supabase.co", publishable_key="public-test-key"
        ),
    )
    monkeypatch.setattr(auth.httpx, "AsyncClient", lambda **_kwargs: FakeClient())

    subject = await verify_supabase_access_token("Bearer valid-test-token")

    assert subject == USER_ID
    assert isinstance(subject, UUID)
    assert requests == [
        (
            "https://project.supabase.co/auth/v1/user",
            {
                "Authorization": "Bearer valid-test-token",
                "apikey": "public-test-key",
                "Accept": "application/json",
            },
        )
    ]
