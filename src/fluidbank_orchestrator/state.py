"""State contract for the FluidBank LangGraph workflow.

No A2UI type lives here: template/surface generation is owned by the MCP
server's A2UI implementation, not the agent.
"""

from __future__ import annotations

from typing import TypedDict


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
    user_id: str
    user_profile: UserProfile
    message: str
    months: int
