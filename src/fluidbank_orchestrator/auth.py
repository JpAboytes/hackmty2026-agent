"""Supabase access-token verification at the public API boundary."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

import httpx


class AuthenticationError(ValueError):
    """The caller could not be authenticated without exposing token details."""


@dataclass(frozen=True, slots=True)
class SupabaseAuthConfig:
    url: str
    publishable_key: str


def load_supabase_auth_config(
    environment: Mapping[str, str] | None = None,
) -> SupabaseAuthConfig:
    env = os.environ if environment is None else environment
    raw_url = env.get("SUPABASE_URL", "").strip()
    parsed = urlsplit(raw_url)
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if (
        not raw_url
        or parsed.scheme not in ({"http", "https"} if local else {"https"})
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.path.rstrip("/")
        or parsed.query
        or parsed.fragment
    ):
        raise AuthenticationError("Supabase authentication is not configured")
    key = (
        env.get("SUPABASE_PUBLISHABLE_KEY", "").strip() or env.get("SUPABASE_ANON_KEY", "").strip()
    )
    if not key or any(character.isspace() for character in key):
        raise AuthenticationError("Supabase authentication is not configured")
    base_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
    return SupabaseAuthConfig(url=base_url, publishable_key=key)


def _bearer_token(authorization: str | None) -> str:
    if authorization is None:
        raise AuthenticationError("A bearer token is required")
    scheme, separator, token = authorization.partition(" ")
    if (
        not separator
        or scheme.casefold() != "bearer"
        or not token
        or any(character.isspace() for character in token)
        or len(token) > 16_384
    ):
        raise AuthenticationError("The bearer token is invalid")
    return token


def _authenticated_subject(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        raise AuthenticationError("The Supabase user response is invalid")
    raw_id = payload.get("id")
    try:
        subject = UUID(raw_id) if isinstance(raw_id, str) else None
    except ValueError:
        subject = None
    if subject is None:
        raise AuthenticationError("The Supabase user response is invalid")
    if payload.get("is_anonymous") is True:
        raise AuthenticationError("Anonymous accounts cannot use the banking agent")
    if not payload.get("email_confirmed_at") and not payload.get("confirmed_at"):
        raise AuthenticationError("The Supabase account is not confirmed")
    return str(subject)


async def verify_supabase_access_token(authorization: str | None) -> str:
    """Resolve the trusted UUID through Supabase Auth's authenticated user endpoint."""
    token = _bearer_token(authorization)
    config = load_supabase_auth_config()
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
            response = await client.get(
                f"{config.url}/auth/v1/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "apikey": config.publishable_key,
                    "Accept": "application/json",
                },
            )
    except httpx.HTTPError:
        raise AuthenticationError("Supabase authentication is unavailable") from None
    if response.status_code != 200:
        raise AuthenticationError("The bearer token was rejected")
    try:
        payload = response.json()
    except ValueError:
        raise AuthenticationError("The Supabase user response is invalid") from None
    return _authenticated_subject(payload)
