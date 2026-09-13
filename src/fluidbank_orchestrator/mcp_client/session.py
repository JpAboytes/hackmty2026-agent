"""One MCP session per turn, shared by everything that runs inside it.

A session handshake against the remote endpoint costs roughly as much as a
query, and a turn used to pay it three times: once to list tools, once for user
context, and once per tool call. Callers join the turn's session when one is
active and fall back to opening their own when it is not, so tests and one-off
calls keep working unchanged.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from typing import Any

from fastmcp import Client

from ..observability import stage
from .config import create_mcp_client, load_mcp_config

_TURN_SESSION: ContextVar[tuple[Client[Any], str] | None] = ContextVar(
    "fluidbank_mcp_session", default=None
)


@asynccontextmanager
async def mcp_session() -> AsyncIterator[None]:
    """Hold one MCP session open for everything inside this block."""
    config = load_mcp_config()
    async with AsyncExitStack() as stack:
        async with stage("mcp.connect", purpose="turn", shared=True):
            client = await stack.enter_async_context(create_mcp_client(config))
        token = _TURN_SESSION.set((client, config.url))
        try:
            yield
        finally:
            _TURN_SESSION.reset(token)


@asynccontextmanager
async def session(purpose: str) -> AsyncIterator[tuple[Client[Any], str]]:
    """Yield the turn's shared session, or a private one when none is active."""
    joined = _TURN_SESSION.get()
    if joined is not None:
        yield joined
        return
    config = load_mcp_config()
    async with AsyncExitStack() as stack:
        async with stage("mcp.connect", purpose=purpose, shared=False):
            client = await stack.enter_async_context(create_mcp_client(config))
        yield client, config.url
