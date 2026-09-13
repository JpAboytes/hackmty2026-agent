"""The HTTP wire contract of the chat endpoint.

The request carries the UUID and selected account UUID needed by Expo. Neither
is trusted: identity comes from the bearer token and account ownership is
verified through MCP. The response carries only a message, structured data, and
an optional validated A2UI bundle.
"""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .a2ui import A2UIBundle
from .a2ui_action import A2UIActionRequest

MAX_QUERY_CHARS = 20_000


class ChatRequest(BaseModel):
    """A chat turn whose security identity comes only from the bearer token."""

    model_config = ConfigDict(extra="forbid")

    query: Annotated[str, Field(min_length=1, max_length=MAX_QUERY_CHARS)] | None = None
    action: A2UIActionRequest | None = None
    # Compatibility only. The deployed mobile client sends the caller's id, and
    # forbidding it outright made every request 422. It is never trusted:
    # identity comes from the bearer token, and a value that disagrees with the
    # token is refused rather than honoured.
    user_id: UUID | None = None
    # Financial context selected by Expo. It is only a claim until the API
    # verifies ownership through the scoped MCP connection.
    account_id: UUID | None = None

    @model_validator(mode="after")
    def exactly_one_input(self) -> ChatRequest:
        if (self.query is None) == (self.action is None):
            raise ValueError("exactly one of query or action is required")
        return self


class ChatResponse(BaseModel):
    """One self-contained answer: conversational text, data, optional surface."""

    message: str
    data: dict[str, Any]
    a2ui: A2UIBundle | None = None
