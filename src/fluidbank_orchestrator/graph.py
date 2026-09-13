"""Composition root for the LangGraph workflow.

The topology, and nothing else:

    START -> validate_identity -> load_tools -> fetch_context -> agent
    agent -> tools -> agent          (while the model asks for tool calls)
    agent -> prepare_action -> END   (the model selected an A2UI form)
    agent -> select_presentation -> build_presentation -> END
    agent -> END                     (bounded conversational answers)

The model decides which of those the turn needs. The topology only makes each
outcome reachable; it does not classify the request.

What each node *does* lives in ``agent.nodes``; the model adapter in
``agent.gemini``; the MCP boundary in ``mcp_client``. Change behaviour there,
not here. This module is also the LangGraph entrypoint declared in
``langgraph.json`` (``graph.py:graph``).

The names re-exported at the bottom keep ``fluidbank_orchestrator.graph``
importable as it was before the workflow was split into ``agent``.
"""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from .agent.gemini import GeminiToolAwareModel
from .agent.model import ModelTurn, ToolAwareModel
from .agent.nodes import (
    ToolLoader,
    build_presentation_node,
    fetch_context_node,
    make_agent_node,
    make_load_tools_node,
    make_prepare_action_node,
    make_tools_node,
    route_after_agent,
    select_presentation_node,
    validate_identity_node,
)
from .agent.tool_loop import ToolExecutor
from .mcp_client import UserContextError, execute_remote_tool, list_remote_tools
from .state import GraphState


def build_graph(
    *,
    model: ToolAwareModel | None = None,
    tool_loader: ToolLoader = list_remote_tools,
    tool_executor: ToolExecutor = execute_remote_tool,
) -> Any:
    """Build an injectable graph with an explicit model -> tools -> model loop."""
    resolved_model = model or GeminiToolAwareModel()

    workflow = StateGraph(GraphState)
    workflow.add_node("validate_identity", cast("Any", validate_identity_node))
    workflow.add_node("load_tools", cast("Any", make_load_tools_node(tool_loader)))
    workflow.add_node("fetch_context", cast("Any", fetch_context_node))
    workflow.add_node("agent", cast("Any", make_agent_node(resolved_model)))
    workflow.add_node("tools", cast("Any", make_tools_node(tool_executor)))
    workflow.add_node("prepare_action", cast("Any", make_prepare_action_node(tool_executor)))
    workflow.add_node("select_presentation", cast("Any", select_presentation_node))
    workflow.add_node("build_presentation", cast("Any", build_presentation_node))
    workflow.add_edge(START, "validate_identity")
    workflow.add_edge("validate_identity", "load_tools")
    workflow.add_edge("load_tools", "fetch_context")
    workflow.add_edge("fetch_context", "agent")
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "prepare_action": "prepare_action",
            "select_presentation": "select_presentation",
            END: END,
        },
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("prepare_action", END)
    workflow.add_edge("select_presentation", "build_presentation")
    workflow.add_edge("build_presentation", END)
    return workflow.compile()


graph = build_graph()

__all__ = [
    "GeminiToolAwareModel",
    "ModelTurn",
    "ToolAwareModel",
    "ToolExecutor",
    "ToolLoader",
    "UserContextError",
    "build_graph",
    "graph",
]
