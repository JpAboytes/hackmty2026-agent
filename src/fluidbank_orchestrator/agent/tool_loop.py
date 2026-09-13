"""Executing the pending tool calls of one loop iteration, with provenance.

Every call that runs is recorded as an observation whatever its outcome, so a
later stage can tell "MCP said there is nothing" from "MCP was never asked".
Two independent gates decide what may run at all: the model may only use what
the server advertised as model-visible (under progressive discovery, the search
pair), and the deterministic planner may only use the scoped financial set.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Protocol
from uuid import UUID

from ..mcp_client import (
    SEARCH_TOOL_NAME,
    MCPConfigurationError,
    MCPToolExecution,
    UserContextError,
    addressable_financial_tool_names,
    require_current_user_id,
    resolve_tool_call,
)
from ..observability import event, preview, stage
from ..state import GraphState, ToolCall, ToolObservation
from .observations import already_searched, text_from_execution
from .tool_visibility import discovered_tool_schemas, model_tool_definitions, tool_definitions


class ToolExecutor(Protocol):
    async def __call__(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        current_user_id: UUID | None = None,
    ) -> MCPToolExecution: ...


def _rejected_observation(name: object) -> ToolObservation:
    return {
        "name": str(name),
        "arguments": {},
        "is_error": True,
        "data": {},
        "text": "La herramienta solicitada no está disponible.",
    }


def _permitted_tool_names(state: GraphState) -> set[str]:
    definitions = tool_definitions(state)
    advertised_names = (tool.name for tool in definitions)
    return {tool.name for tool in model_tool_definitions(state)} | set(
        addressable_financial_tool_names(advertised_names)
    )


async def run_pending_tools(
    state: GraphState,
    pending: list[ToolCall],
    executor: ToolExecutor,
) -> GraphState:
    """Run one iteration's calls and append what each of them observed."""
    permitted = _permitted_tool_names(state)
    observations: list[ToolObservation] = list(state.get("tool_observations", []))
    update: GraphState = {
        "tool_calls": [],
        "tool_loop_count": state.get("tool_loop_count", 0) + 1,
        # A presentation belongs only to calls in this consumed batch.
        "final_tool_execution": None,
    }
    for call in pending:
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or name not in permitted or not isinstance(arguments, dict):
            event("tool.rejected", name=str(name), reason="unavailable")
            observations.append(_rejected_observation(name))
            continue
        # Everything downstream reasons about the domain tool, not the
        # `call_tool` envelope the model wrapped it in.
        target, target_arguments = resolve_tool_call(name, arguments)
        if target == SEARCH_TOOL_NAME and already_searched(state, arguments):
            event("tool.skipped", name=target, reason="duplicate_search")
            continue
        preview(f"tool.{target}.arguments", target_arguments)
        try:
            execution = await _execute(state, name, arguments, target, executor)
            structured = execution.result.structured_content
            data = deepcopy(structured) if isinstance(structured, dict) else {}
            if target == SEARCH_TOOL_NAME:
                data = discovered_tool_schemas(data)
            observations.append(
                {
                    "name": target,
                    "arguments": deepcopy(dict(target_arguments or {})),
                    "is_error": bool(execution.result.is_error),
                    "data": data,
                    "text": text_from_execution(execution),
                }
            )
            # Ownership comes from MCP's `_meta.ui`, as interpreted by the
            # bridge, never from a local tool-name list. A rejected MCP-owned
            # surface still terminates on its safe text/data fallback.
            if execution.mcp_ui_owned or execution.a2ui is not None:
                update["final_tool_execution"] = execution
        except (MCPConfigurationError, UserContextError) as exc:
            event("tool.failed", name=target, reason=type(exc).__name__)
            observations.append(
                {
                    "name": target,
                    "arguments": deepcopy(dict(target_arguments or {})),
                    "is_error": True,
                    "data": {},
                    "text": "No pude consultar el servicio de datos en este momento.",
                }
            )
    update["tool_observations"] = observations
    return update


async def _execute(
    state: GraphState,
    name: str,
    arguments: Mapping[str, Any],
    target: str,
    executor: ToolExecutor,
) -> MCPToolExecution:
    async with stage("tool.call", name=target, discovered=target != name) as step:
        execution = await executor(
            name,
            arguments,
            current_user_id=require_current_user_id(state.get("current_user_id")),
        )
        structured = execution.result.structured_content
        step.set(
            is_error=bool(execution.result.is_error),
            rows=len(structured["rows"])
            if isinstance(structured, dict) and isinstance(structured.get("rows"), list)
            else None,
            a2ui=execution.a2ui is not None,
        )
    return execution
