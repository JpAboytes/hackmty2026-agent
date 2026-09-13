"""Executing the pending tool calls of one loop iteration, with provenance.

Every call that runs is recorded as an observation whatever its outcome, so a
later stage can tell "MCP said there is nothing" from "MCP was never asked".

One gate decides what may run: ``_permitted_tool_names``, the union of what the
server advertised as model-visible (under progressive discovery, the search
pair) and ``FINANCIAL_DOMAIN_TOOL_NAMES``. The domain half is what lets a tool
the model reached *through* ``call_tool`` execute once ``resolve_tool_call`` has
unwrapped it, since the domain tool itself is never advertised to the model.
Permission is not scoping: whose data a call may touch is decided by
``mcp_client.trusted_scope`` regardless of how the call got here.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Protocol
from uuid import UUID

from ..mcp_client import (
    FINANCIAL_DOMAIN_TOOL_NAMES,
    SEARCH_TOOL_NAME,
    MCPConfigurationError,
    MCPToolExecution,
    UserContextError,
    require_current_user_id,
    resolve_tool_call,
)
from ..observability import event, preview, stage
from ..state import GraphState
from .observations import already_searched, text_from_execution
from .tool_visibility import discovered_tool_schemas, model_tool_definitions

#: Tools whose execution is itself the answer: their MCP-produced A2UI is
#: relayed through the bridge instead of a Finance v2 presentation.
_PRESENTATION_TOOL_NAMES = frozenset({"visualize_allowed_data", "database_overview"})


class ToolExecutor(Protocol):
    async def __call__(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        current_user_id: UUID | None = None,
    ) -> MCPToolExecution: ...


def _rejected_observation(name: object) -> dict[str, Any]:
    return {
        "name": str(name),
        "arguments": {},
        "is_error": True,
        "data": {},
        "text": "La herramienta solicitada no está disponible.",
    }


def _permitted_tool_names(state: GraphState) -> set[str]:
    return {tool.name for tool in model_tool_definitions(state)} | FINANCIAL_DOMAIN_TOOL_NAMES


async def run_pending_tools(
    state: GraphState,
    pending: list[dict[str, Any]],
    executor: ToolExecutor,
) -> GraphState:
    """Run one iteration's calls and append what each of them observed."""
    permitted = _permitted_tool_names(state)
    observations = list(state.get("tool_observations", []))
    update: GraphState = {
        "tool_calls": [],
        "tool_loop_count": state.get("tool_loop_count", 0) + 1,
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
            if target in _PRESENTATION_TOOL_NAMES:
                update["final_tool_execution"] = execution
        except (MCPConfigurationError, UserContextError) as exc:
            event("tool.failed", name=name, reason=type(exc).__name__)
            observations.append(
                {
                    "name": name,
                    "arguments": deepcopy(arguments),
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
