"""The port the graph depends on instead of on any particular model vendor.

One turn of a tool-aware model produces either tool calls or an answer, never
both. ``build_graph`` takes any object satisfying ``ToolAwareModel``, which is
what lets the tests drive the whole workflow with a scripted model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..mcp_client import MCPToolDefinition
from ..schemas.banking_view import FinancialIntent
from ..state import UserProfile


@dataclass(frozen=True, slots=True)
class ModelTurn:
    """One model decision: some tool calls, or a message and optional intent."""

    message: str
    tool_calls: tuple[dict[str, Any], ...] = ()
    months: int | None = None
    presentation_intent: FinancialIntent | None = None


class ToolAwareModel(Protocol):
    async def generate(
        self,
        *,
        query: str,
        profile: UserProfile,
        tools: Sequence[MCPToolDefinition],
        observations: Sequence[Mapping[str, Any]],
    ) -> ModelTurn: ...
