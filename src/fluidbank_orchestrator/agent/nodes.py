"""What each step of the workflow does, and how the agent step decides.

Every node is built by a small factory so its external dependency is explicit
and injectable; ``graph`` wires them into the topology and nothing else.

The policy that matters lives in ``agent_node``. It is deliberately short,
because *deciding what the user needs is the model's job*: whether data is
required, which capability provides it, how many capabilities, whether another
discovery round is worthwhile, and whether the turn should end in a
presentation, an action form, or plain text. The node only enforces the
invariants the model must not be trusted with:

1. a deterministic pre-model policy gate rejects prompt injection, executable
   code, historical narration, and non-banking requests;
2. no verified user context - answer without figures, never invent them;
3. an MCP-owned presentation from the current tool batch - relay it;
4. an action re-entry whose approved view does not validate - refuse it;
5. the loop limit - stop rather than keep calling tools;
6. an explicitly requested view pins the presentation so an approved action
   cannot be silently upgraded to a different one;
7. anything the model returns as a protocol value is re-normalized before it
   can reach a builder or MCP.

Coarse lifecycle phases are reported through ``status.emit_status``. They carry
an identifier and nothing else - never prompt text, reasoning, arguments or
rows.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from langgraph.graph import END

from ..a2ui_actions.forms import normalize_form_arguments, normalize_form_name
from ..mcp_client import (
    SEARCH_TOOL_NAME,
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
    normalize_action_intent,
    select_presentation_intent,
)
from ..state import GraphState, ToolCall, UserProfile
from .model import ModelTurn, ToolAwareModel
from .observations import (
    context_observations,
    retained_observations,
    text_from_execution,
)
from .policy import evaluate_query_policy, safe_model_message
from .status import AgentStatus, emit_status
from .tool_loop import ToolExecutor, run_pending_tools
from .tool_visibility import model_tool_definitions

#: What the graph reports when MCP cannot describe the user. It carries no
#: balances on purpose: a fallback must never read as financial data.
FALLBACK_PROFILE: UserProfile = {
    "literacy_level": "medium",
    "font_scale": "lg",
    "contrast": "high",
    "color_vision_mode": "none",
    "hit_target": "large",
    "overdraft_risk": None,
    "recurring_expenses": 0.0,
    "available_balance": None,
    "owned_balances": {},
}

MAX_TOOL_TURNS = 8

Node = Callable[[GraphState], Awaitable[GraphState]]
ToolLoader = Callable[[], Awaitable[list[MCPToolDefinition]]]

#: The tool that prepares - never saves - an A2UI action form.
FORM_TOOL_NAME = "a2ui_form"


def _copy_tool_calls(calls: Sequence[ToolCall]) -> list[ToolCall]:
    return [{"name": call["name"], "arguments": dict(call["arguments"])} for call in calls]


async def validate_identity_node(state: GraphState) -> GraphState:
    """Authenticate the turn, then reset every turn-scoped workflow channel.

    The graph is checkpointed per user, so a channel left behind by the
    previous turn would otherwise be read as this turn's decision. Only
    ``requested_intent`` survives, and only when the client actually sent an
    approved action - it is trusted input, so it is normalized here once.
    """
    emit_status("interpreting")
    action_requested = state.get("action_requested") is True
    requested_intent = (
        normalize_action_intent(state.get("requested_intent")) if action_requested else None
    )
    return {
        "current_user_id": require_current_user_id(state.get("current_user_id")),
        "requested_intent": requested_intent,
        "action_requested": action_requested,
        "policy_refused": False,
        "policy_reason": None,
        "presentation_intent": None,
        "action_form": None,
        "action_form_arguments": {},
        "message": "",
        "months": None,
        "tool_calls": [],
        "tool_observations": [],
        "context_observations": [],
        "final_tool_execution": None,
        "financial_presentation": None,
        "tool_loop_count": 0,
    }


async def enforce_query_policy_node(state: GraphState) -> GraphState:
    """Reject unsafe or out-of-domain text before any model or data access."""
    if state.get("action_requested") is True:
        return {"policy_refused": False, "policy_reason": None}
    decision = evaluate_query_policy(state.get("user_query", ""))
    if decision.allowed:
        return {"policy_refused": False, "policy_reason": None}
    event("policy.refused", reason=decision.reason)
    return {
        "policy_refused": True,
        "policy_reason": decision.reason,
        "message": decision.message,
        "tool_calls": [],
    }


def route_after_policy(state: GraphState) -> str:
    """A refusal ends immediately; an allowed turn may load tools/context."""
    destination = END if state.get("policy_refused") is True else "load_tools"
    event("graph.route", node="query_policy", next=destination)
    return destination


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


def _phase_before_model(observations: Sequence[Mapping[str, Any]]) -> AgentStatus:
    """Which coarse phase the upcoming model turn represents.

    Reading only the shape of the ledger, never its contents: an empty ledger
    means the request is still being interpreted, a fresh discovery result
    means candidate tools are being chosen, and domain results mean they are
    being read.
    """
    if not observations:
        return "interpreting"
    if observations[-1].get("name") == SEARCH_TOOL_NAME:
        return "selecting_tools"
    return "interpreting_results"


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

    # Ownership comes from MCP: either the bridge produced a validated bundle,
    # or `_meta.ui` claimed the surface and its safe fallback is the answer.
    final_execution = state.get("final_tool_execution")
    if isinstance(final_execution, MCPToolExecution) and (
        final_execution.mcp_ui_owned or final_execution.a2ui is not None
    ):
        step.set(decision="tool_presentation")
        return {"message": text_from_execution(final_execution), "tool_calls": []}

    # An approved action re-enters with the view the user actually approved.
    # A value that does not survive the finite vocabulary is a protocol fault,
    # not a question for the model.
    if state.get("action_requested") is True and state.get("requested_intent") is None:
        step.set(decision="invalid_action_intent")
        return {
            "message": "La presentación financiera solicitada no es válida.",
            "tool_calls": [],
        }

    if state.get("tool_loop_count", 0) >= MAX_TOOL_TURNS:
        step.set(decision="loop_limit", limit=MAX_TOOL_TURNS)
        return {
            "message": "No pude completar la consulta dentro del límite seguro de pasos.",
            "tool_calls": [],
        }

    observations = state.get("tool_observations", [])
    emit_status(_phase_before_model(observations))
    candidate = await model.generate(
        query=state["user_query"],
        profile=state["user_profile"],
        tools=model_tool_definitions(state),
        observations=observations,
    )
    return _from_model_turn(candidate, state, step)


def _from_model_turn(candidate: ModelTurn, state: GraphState, step: stage) -> GraphState:
    """Accept a model turn as tool calls, a form, an intent, or a message.

    Every protocol value the model produced is re-normalized here, so a
    hallucinated intent, form name or prefill becomes ``None`` rather than
    reaching a builder or MCP. An explicitly requested view outranks the
    model's choice: an approved action must present the view it was approved
    for.
    """
    pinned = normalize_action_intent(state.get("requested_intent"))
    # Each branch below owns exactly one outcome, so the two routing channels
    # start empty and only the selected one is filled.
    result: GraphState = {
        "message": "",
        "tool_calls": [],
        "presentation_intent": None,
        "action_form": None,
        "action_form_arguments": {},
    }
    if (refusal := safe_model_message(candidate.message)) is not None:
        step.set(decision="model_output_refused")
        result["message"] = refusal
        return result
    if candidate.tool_calls:
        step.set(
            decision="model_tool_calls",
            calls=",".join(call["name"] for call in candidate.tool_calls),
        )
        result["tool_calls"] = _copy_tool_calls(candidate.tool_calls)
    elif (form_name := normalize_form_name(candidate.action_form)) is not None:
        arguments = normalize_form_arguments(
            form_name,
            amount=candidate.form_amount,
            recipient=candidate.form_recipient,
        )
        step.set(
            decision="model_action_form",
            form=form_name,
            prefilled=",".join(sorted(arguments)) or None,
        )
        result["message"] = candidate.message
        result["action_form"] = form_name
        result["action_form_arguments"] = arguments
    elif (intent := pinned or normalize_action_intent(candidate.presentation_intent)) is not None:
        step.set(
            decision="model_presentation",
            intent=intent,
            source="requested" if pinned is not None else "model",
        )
        result["message"] = candidate.message
        result["presentation_intent"] = intent
    else:
        step.set(decision="model_message", message_chars=len(candidate.message.strip()))
        result["message"] = candidate.message.strip() or (
            "No tengo una respuesta para mostrar en este momento."
        )
    if candidate.months is not None:
        result["months"] = candidate.months
    return result


def make_tools_node(tool_executor: ToolExecutor) -> Node:
    async def tools_node(state: GraphState) -> GraphState:
        pending = state.get("tool_calls", [])
        emit_status(
            "discovering_tools"
            if any(call.get("name") == SEARCH_TOOL_NAME for call in pending)
            else "executing_tools"
        )
        async with stage("node.tools", calls=len(pending)):
            return await run_pending_tools(state, pending, tool_executor)

    return tools_node


def make_prepare_action_node(tool_executor: ToolExecutor) -> Node:
    """Ask MCP to prepare the form the model selected.

    Preparing a form is not performing its write: ``a2ui_form`` reads current
    values and returns a surface, and only a later user Button event reaches an
    action handler. The name and every prefilled value are re-validated here
    because the model chose them, and the rendered form stays authoritative.
    """

    async def prepare_action_node(state: GraphState) -> GraphState:
        emit_status("preparing_action")
        form_name = normalize_form_name(state.get("action_form"))
        async with stage("node.prepare_action", form=form_name) as step:
            if form_name is None:
                step.set(outcome="invalid")
                return {"message": "La acción solicitada no está disponible."}
            raw_arguments = state.get("action_form_arguments") or {}
            arguments = normalize_form_arguments(
                form_name,
                amount=raw_arguments.get("initial_amount"),
                recipient=raw_arguments.get("initial_recipient"),
            )
            try:
                execution = await tool_executor(
                    FORM_TOOL_NAME,
                    {"name": form_name, **arguments},
                    current_user_id=require_current_user_id(state.get("current_user_id")),
                )
            except (MCPConfigurationError, UserContextError) as exc:
                step.set(outcome="unavailable", reason=type(exc).__name__)
                return {"message": "No pude preparar la acción en este momento."}
            emit_status("building_ui")
            step.set(
                outcome="prepared",
                is_error=bool(execution.result.is_error),
                a2ui=execution.a2ui is not None,
            )
            if execution.a2ui is not None:
                emit_status("validating_ui")
            return {
                "message": text_from_execution(execution),
                "final_tool_execution": execution,
            }

    return prepare_action_node


async def select_presentation_node(state: GraphState) -> GraphState:
    """Refine the presentation semantics against what retrieval actually returned."""
    async with stage("node.select_presentation") as step:
        requested = normalize_action_intent(state.get("presentation_intent"))
        if requested is None:
            step.set(outcome="invalid")
            return {
                "presentation_intent": None,
                "financial_presentation": None,
                "message": "La presentación financiera solicitada no es válida.",
            }
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
    emit_status("building_ui")
    async with stage("node.build_presentation") as step:
        intent = normalize_action_intent(state.get("presentation_intent"))
        if intent is None:
            step.set(outcome="invalid")
            return {
                "financial_presentation": None,
                "message": "La presentación financiera solicitada no es válida.",
            }
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
        emit_status("validating_ui")
        return {"message": presentation.message, "financial_presentation": presentation}


def route_after_agent(state: GraphState) -> str:
    """Tool calls loop back; a form prepares; an intent presents; else the turn ends.

    Presenting additionally requires that something was actually read this turn.
    A presentation intent with an empty ledger means the model named a view
    without any data behind it, and provenance is the one thing it does not get
    to assert: with nothing retained there is nothing to render, so the turn
    ends conversationally instead of publishing an empty surface.
    """
    if state.get("tool_calls"):
        destination = "tools"
    elif normalize_form_name(state.get("action_form")) is not None:
        destination = "prepare_action"
    elif normalize_action_intent(state.get("presentation_intent")) is not None:
        destination = "select_presentation" if retained_observations(state) else END
    else:
        destination = END
    event("graph.route", node="agent", next=destination)
    return destination
