"""State contract for the FluidBank LangGraph workflow."""

from __future__ import annotations

from typing import Any, TypedDict
from uuid import UUID


class UserProfile(TypedDict):
    """Financial and accessibility context fetched via the Supabase MCP server
    (or a local fallback profile when it is unreachable)."""

    literacy_level: str
    font_scale: str
    contrast: str
    hit_target: str
    overdraft_risk: float | None
    recurring_expenses: float
    available_balance: float | None
    owned_balances: dict[str, float]


class GraphState(TypedDict, total=False):
    """Data passed between graph nodes."""

    user_query: str
    current_user_id: UUID
    requested_intent: str
    action_requested: bool
    financial_request_intent: str
    presentation_intent: str
    user_profile: UserProfile
    context_available: bool
    message: str
    months: int
    available_tools: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    tool_observations: list[dict[str, Any]]
    final_tool_execution: object
    financial_presentation: object
    tool_loop_count: int
