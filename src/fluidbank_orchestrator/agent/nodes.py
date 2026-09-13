"""What each step of the workflow does, and how the agent step decides.

Every node is built by a small factory so its external dependency is explicit
and injectable; ``graph`` wires them into the topology and nothing else.

The policy that matters lives in ``agent_node``, in this order:

1. no verified user context - answer without figures, never invent them;
2. a classified or action-requested financial intent - plan deterministically
   through ``retrieval`` and never consult the model;
3. an MCP-produced presentation already retained - relay it;
4. the loop limit - stop rather than keep calling tools;
5. otherwise the model turn, which may only choose tools or a bounded answer.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langgraph.graph import END

from ..mcp_client import (
    MCPConfigurationError,
    MCPToolDefinition,
    MCPToolExecution,
    UserContextError,
    fetch_user_context,
    require_current_user_id,
)
from ..observability import event, stage
from ..services.financial_presentation import (
    build_financial_presentation,
    classify_financial_request,
    normalize_action_intent,
    select_presentation_intent,
)
from ..state import GraphState, UserProfile
from .model import ModelTurn, ToolAwareModel
from .observations import (
    context_observations,
    retained_observations,
    text_from_execution,
)
from .retrieval import financial_data_turn
from .tool_loop import ToolExecutor, run_pending_tools
from .tool_visibility import model_tool_definitions

#: What the graph reports when MCP cannot describe the user. It carries no
#: balances on purpose: a fallback must never read as financial data.
FALLBACK_PROFILE: UserProfile = {
    "literacy_level": "medium",
    "font_scale": "lg",
    "contrast": "high",
    "hit_target": "large",
    "overdraft_risk": None,
    "recurring_expenses": 0.0,
    "available_balance": None,
    "owned_balances": {},
}

MAX_TOOL_TURNS = 8

Node = Callable[[GraphState], Awaitable[GraphState]]
ToolLoader = Callable[[], Awaitable[list[MCPToolDefinition]]]


async def validate_identity_node(state: GraphState) -> GraphState:
    """Fail the turn before any load or read unless identity is authenticated."""
    return {"current_user_id": require_current_user_id(state.get("current_user_id"))}


def make_load_tools_node(tool_loader: ToolLoader) -> Node:
    async def load_tools_node(_state: GraphState) -> GraphState:
        async with stage("node.load_tools") as step:
            try:
                tools = await tool_loader()
            except (MCPConfigurationError, UserContextError) as exc:
                step.set(outcome="unavailable", reason=type(exc).__name__, tools=0)
                return {"available_tools": []}
            step.set(outcome="loaded", tools=len(tools))
            return {"available_tools": [tool.as_dict() for tool in tools]}

    return load_tools_node


async def fetch_context_node(state: GraphState) -> GraphState:
    """Load the signed-in user's profile and retain the rows it was built from."""
    async with stage("node.fetch_context") as step:
        try:
            current_user_id = require_current_user_id(state.get("current_user_id"))
            context = await fetch_user_context(current_user_id)
        except (MCPConfigurationError, UserContextError) as exc:
            # Missing context must never become invented financial data.
            step.set(outcome="fallback", reason=type(exc).__name__)
            return {"user_profile": FALLBACK_PROFILE.copy(), "context_available": False}
        retained = context_observations(context.rows)
        # Presence, never amounts: a balance is the user's money, not a log line.
        step.set(
            outcome="resolved",
            accounts=len(context.profile["owned_balances"]),
            has_balance=context.profile["available_balance"] is not None,
            retained=",".join(sorted(context.rows)) or "none",
        )
        return {
            "user_profile": context.profile,
            "context_available": True,
            "context_observations": retained,
        }


def make_agent_node(model: ToolAwareModel) -> Node:
    async def agent_node(state: GraphState) -> GraphState:
        async with stage(
            "node.agent",
            turn=state.get("tool_loop_count", 0),
            observations=len(state.get("tool_observations", [])),
        ) as step:
            return await _agent_turn(state, step, model)

    return agent_node


async def _agent_turn(state: GraphState, step: stage, model: ToolAwareModel) -> GraphState:
    if state.get("context_available") is False:
        step.set(decision="no_context")
        return {
            "message": (
                "No pude identificar tu cuenta, así que no puedo mostrarte cifras. "
                "Vuelve a iniciar sesión e inténtalo de nuevo."
            ),
            "tool_calls": [],
        }
    observations = state.get("tool_observations", [])
    explicit_intent = normalize_action_intent(state.get("requested_intent"))
    financial_intent = explicit_intent or classify_financial_request(state["user_query"])
    if financial_intent is not None:
        candidate = financial_data_turn(state, financial_intent)
        # The deterministic financial path never reaches Gemini; seeing this
        # decision with no model.gemini stage after it is the fast path.
        step.set(
            decision="financial_retrieval" if candidate.tool_calls else "financial_ready",
            intent=financial_intent,
            source="action" if explicit_intent is not None else "classifier",
            calls=",".join(call["name"] for call in candidate.tool_calls) or None,
        )
        return {
            "financial_request_intent": financial_intent,
            "message": candidate.message,
            "tool_calls": [dict(call) for call in candidate.tool_calls],
        }

    final_execution = state.get("final_tool_execution")
    if isinstance(final_execution, MCPToolExecution) and final_execution.a2ui is not None:
        step.set(decision="tool_presentation")
        return {"message": text_from_execution(final_execution), "tool_calls": []}
    if state.get("tool_loop_count", 0) >= MAX_TOOL_TURNS:
        step.set(decision="loop_limit", limit=MAX_TOOL_TURNS)
        return {
            "message": "No pude completar la consulta dentro del límite seguro de pasos.",
            "tool_calls": [],
        }

    candidate = await model.generate(
        query=state["user_query"],
        profile=state["user_profile"],
        tools=model_tool_definitions(state),
        observations=observations,
    )
    return _from_model_turn(candidate, step)


def _from_model_turn(candidate: ModelTurn, step: stage) -> GraphState:
    """Accept a model turn as tool calls, an intent selection, or a message.

    The model may select from the finite presentation vocabulary; it cannot
    invent one, because the value is normalized again downstream.
    """
    if candidate.tool_calls:
        step.set(
            decision="model_tool_calls",
            calls=",".join(call["name"] for call in candidate.tool_calls),
        )
        result: GraphState = {
            "message": "",
            "tool_calls": [dict(call) for call in candidate.tool_calls],
        }
    elif candidate.presentation_intent is not None:
        step.set(decision="model_presentation", intent=candidate.presentation_intent)
        result = {
            "message": candidate.message,
            "tool_calls": [],
            "financial_request_intent": candidate.presentation_intent,
        }
    else:
        step.set(decision="model_message", message_chars=len(candidate.message.strip()))
        message_text = candidate.message.strip() or (
            "No tengo una respuesta para mostrar en este momento."
        )
        result = {"message": message_text, "tool_calls": []}
    if candidate.months is not None:
        result["months"] = candidate.months
    return result


def make_tools_node(tool_executor: ToolExecutor) -> Node:
    async def tools_node(state: GraphState) -> GraphState:
        pending = state.get("tool_calls", [])
        async with stage("node.tools", calls=len(pending)):
            return await run_pending_tools(state, pending, tool_executor)

    return tools_node


async def select_presentation_node(state: GraphState) -> GraphState:
    """Choose the presentation semantics, only once retrieval has happened."""
    async with stage("node.select_presentation") as step:
        requested = normalize_action_intent(state.get("financial_request_intent"))
        if requested is None:
            step.set(outcome="invalid")
            return {"message": "La presentación financiera solicitada no es válida."}
        selected = select_presentation_intent(
            requested,
            retained_observations(state),
            query=state["user_query"],
            action_requested=state.get("action_requested", False),
        )
        step.set(requested=requested, selected=selected)
        return {"presentation_intent": selected}


async def build_presentation_node(state: GraphState) -> GraphState:
    """Build the trusted Finance v2 surface from retained observations only."""
    async with stage("node.build_presentation") as step:
        intent = normalize_action_intent(state.get("presentation_intent"))
        if intent is None:
            step.set(outcome="invalid")
            return {"message": "La presentación financiera solicitada no es válida."}
        presentation = build_financial_presentation(
            intent,
            retained_observations(state),
            state["user_profile"],
        )
        step.set(
            intent=intent,
            a2ui_messages=len(presentation.a2ui.messages),
            data_keys=",".join(sorted(presentation.data)) or "none",
        )
        return {"message": presentation.message, "financial_presentation": presentation}


def route_after_agent(state: GraphState) -> str:
    """Tool calls loop back; a financial intent presents; anything else ends."""
    if state.get("tool_calls"):
        destination = "tools"
    elif normalize_action_intent(state.get("financial_request_intent")) is not None:
        destination = "select_presentation"
    else:
        destination = END
    event("graph.route", node="agent", next=destination)
    return destination
