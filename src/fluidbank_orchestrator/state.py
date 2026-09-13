"""State contract for the FluidBank LangGraph workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict
from uuid import UUID

from .schemas.banking_view import FinancialIntent

if TYPE_CHECKING:
    from .mcp_client import MCPToolExecution
    from .services.financial_presentation import FinancialPresentation
else:
    # LangGraph resolves TypedDict annotations at runtime. Static checking sees
    # the concrete boundary types, while runtime resolution only needs channel
    # placeholders and retains no dependency back into MCP or services.
    MCPToolExecution = object
    FinancialPresentation = object


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


class ToolDefinitionState(TypedDict):
    """Detached advertised MCP definition retained in graph state."""

    name: str
    description: str
    input_schema: dict[str, Any]
    model_visible: bool


class ToolCall(TypedDict):
    """One pending MCP call. The tools node consumes the whole list."""

    name: str
    arguments: dict[str, Any]


class ToolObservation(TypedDict):
    """One completed MCP attempt, including structured failures."""

    name: str
    arguments: dict[str, Any]
    is_error: bool
    data: dict[str, Any]
    text: str


class GraphState(TypedDict, total=False):
    """Data passed between graph nodes."""

    user_query: str
    current_user_id: UUID
    # Input-only action intent. It is ignored unless action_requested is true.
    requested_intent: FinancialIntent | None
    action_requested: bool
    # Effective classified/model-selected intent, then the post-data selection.
    financial_request_intent: FinancialIntent | None
    presentation_intent: FinancialIntent | None
    user_profile: UserProfile
    context_available: bool
    message: str
    months: int | None
    # These lists intentionally use LangGraph's default overwrite channel.
    # Nodes return the complete current value; no implicit accumulation occurs.
    available_tools: list[ToolDefinitionState]
    tool_calls: list[ToolCall]
    tool_observations: list[ToolObservation]
    context_observations: list[ToolObservation]
    # The two presentation owners are mutually exclusive terminal outputs.
    final_tool_execution: MCPToolExecution | None
    financial_presentation: FinancialPresentation | None
    tool_loop_count: int
