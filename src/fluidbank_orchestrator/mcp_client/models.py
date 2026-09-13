"""Detached, JSON-safe values that cross the MCP boundary.

These carry no transport state: an execution or a tool definition can be held
in graph state, copied, and logged without holding a session open.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastmcp.client.client import CallToolResult

from ..schemas.a2ui import A2UIBundle
from ..state import ToolDefinitionState, UserProfile


@dataclass(frozen=True, slots=True)
class MCPToolExecution:
    """One MCP result plus its optional, independently validated presentation."""

    result: CallToolResult
    a2ui: A2UIBundle | None
    mcp_ui_owned: bool = False
    presentation_error: bool = False


@dataclass(frozen=True, slots=True)
class UserContext:
    """One user's derived profile plus the scoped rows it was built from.

    The rows travel with the profile so a later domain read does not fetch the
    same scoped table a second time in the same turn.
    """

    profile: UserProfile
    rows: dict[str, list[dict[str, object]]]


@dataclass(frozen=True, slots=True)
class MCPToolDefinition:
    """Detached, JSON-safe tool definition loaded from the active MCP endpoint."""

    name: str
    description: str
    input_schema: dict[str, Any]
    #: Whether the server declares this tool visible to a model. The MCP Apps
    #: spec puts that filtering on the host, and for the model-facing tool set
    #: this orchestrator is the host.
    model_visible: bool = True

    def as_dict(self) -> ToolDefinitionState:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "model_visible": self.model_visible,
        }
