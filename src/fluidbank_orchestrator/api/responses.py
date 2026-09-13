"""Building the one envelope the client receives, and logging what left.

Three sources produce a response - a graph presentation, a raw MCP execution,
or a bounded refusal - and all three converge on ``ChatResponse`` here so the
client's contract has exactly one shape. Logging records sizes, keys and
counts; financial values never reach a log line.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from mcp.types import TextContent

from ..agent.policy import safe_model_message
from ..mcp_client import MCPToolExecution
from ..observability import event, preview
from ..schemas.chat import ChatResponse
from ..services.financial_presentation import FinancialPresentation


def _policy_safe_message(message: str) -> str:
    """Apply the final prose guard regardless of which internal path produced it."""
    return safe_model_message(message) or message


def response_from_tool(execution: MCPToolExecution) -> ChatResponse:
    """Relay one MCP result, with its bridged A2UI when the bridge produced one."""
    message = next(
        (
            content.text.strip()
            for content in execution.result.content
            if isinstance(content, TextContent) and content.text.strip()
        ),
        "La operación se completó.",
    )
    structured = execution.result.structured_content
    data = deepcopy(structured) if isinstance(structured, dict) else {}
    return ChatResponse(message=_policy_safe_message(message), data=data, a2ui=execution.a2ui)


def response_from_graph(result: dict[str, Any]) -> ChatResponse:
    """Read the graph's terminal state in priority order.

    A trusted Finance v2 presentation wins; a retained MCP execution is relayed
    through the generic bridge path; otherwise the turn ends conversationally
    with no surface at all.
    """
    presentation = result.get("financial_presentation")
    if isinstance(presentation, FinancialPresentation):
        event("graph.output", source="financial_presentation", intent=presentation.intent)
        return ChatResponse(
            message=_policy_safe_message(presentation.message),
            data=deepcopy(presentation.data),
            a2ui=presentation.a2ui,
        )
    final_execution = result.get("final_tool_execution")
    if isinstance(final_execution, MCPToolExecution):
        event("graph.output", source="final_tool_execution")
        return response_from_tool(final_execution)
    event("graph.output", source="model_message", turns=result.get("tool_loop_count", 0))
    data: dict[str, Any] = dict(result.get("user_profile", {}))
    months = result.get("months")
    if isinstance(months, int):
        data["months"] = months
    return ChatResponse(message=_policy_safe_message(result["message"]), data=data, a2ui=None)


def invalid_action_response(reason: str) -> ChatResponse:
    event("client.response", route="invalid_action", reason=reason, a2ui=False)
    return ChatResponse(message="La acción de interfaz no es válida.", data={}, a2ui=None)


def unavailable_response() -> ChatResponse:
    event("client.response", route="unavailable", a2ui=False)
    return ChatResponse(
        message="No pude consultar el servicio de datos en este momento.",
        data={},
        a2ui=None,
    )


def log_client_response(route: str, response: ChatResponse) -> ChatResponse:
    """Describe the exact envelope leaving for Expo, without its financial values."""
    bundle = response.a2ui
    event(
        "client.response",
        route=route,
        message_chars=len(response.message),
        data_keys=",".join(sorted(response.data)) or "none",
        a2ui=bundle is not None,
        resource_uri=bundle.resource_uri if bundle is not None else None,
        a2ui_messages=len(bundle.messages) if bundle is not None else None,
        a2ui_bytes=len(bundle.model_dump_json()) if bundle is not None else None,
    )
    preview("client.response", response.model_dump(mode="json"))
    return response
