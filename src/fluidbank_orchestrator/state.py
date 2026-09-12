"""State contract for the FluidBank LangGraph workflow.

No A2UI type lives here: template/surface generation is owned by the MCP
server's A2UI implementation, not the agent.
"""

from __future__ import annotations

from typing import Any, TypedDict


class UserProfile(TypedDict):
    """Financial and accessibility context fetched via the Supabase MCP server
    (or a local fallback profile when it is unreachable)."""

    literacy_level: str
    font_scale: str
    contrast: str
    hit_target: str
    overdraft_risk: float
    recurring_expenses: float
    available_balance: float


class GraphState(TypedDict, total=False):
    """Data passed between graph nodes."""

    user_query: str
    user_email: str
    user_profile: UserProfile
    message: str
    months: int
    available_tools: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    tool_observations: list[dict[str, Any]]
    final_tool_execution: object
    tool_loop_count: int
