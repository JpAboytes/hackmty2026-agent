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
    color_vision_mode: str
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
    #: A view the user explicitly approved through an A2UI action. Trusted
    #: input, ignored unless `action_requested` is true, and the only thing
    #: that outranks the model's own choice.
    requested_intent: FinancialIntent | None
    action_requested: bool
    #: True when the deterministic pre-model safety/scope gate refused the
    #: query. A refused turn goes directly to END without loading context,
    #: exposing tool schemas, or invoking the model.
    policy_refused: bool
    policy_reason: str | None
    #: The one intent this turn presents. The model proposes it, the finite
    #: vocabulary validates it, and `requested_intent` pins it when set.
    presentation_intent: FinancialIntent | None
    #: The A2UI form the model asked MCP to prepare, from the finite set in
    #: `a2ui_actions.forms`.
    action_form: str | None
    #: Validated, bounded prefill for that form. The model may suggest values;
    #: `a2ui_actions.forms.normalize_form_arguments` is the only way one
    #: reaches MCP, and the rendered form stays authoritative.
    action_form_arguments: dict[str, Any]
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
