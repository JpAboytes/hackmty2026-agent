"""What a model is allowed to see of the tool surface.

Two independent narrowings live here, and neither is a security control:

* *Visibility* - the server declares which tools a model may be offered
  (``_meta.ui.visibility``); this orchestrator is the host that honours it.
* *Schema sanitization* - trusted identity fields are removed from the
  declarations a model reads, so it is never invited to fabricate a user id,
  and unions Gemini rejects are narrowed to what it accepts.

The real boundary is ``mcp_client.trusted_scope``: identity is overwritten
there whatever the model sends.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

from ..mcp_client import FINANCIAL_DOMAIN_TOOL_NAMES, USER_CONTEXT_TOOL_NAME, MCPToolDefinition
from ..state import GraphState

#: Tools whose schemas legitimately declare trusted fields the orchestrator
#: fills in. Showing them would invite a fabricated user id.
_IDENTITY_BEARING_TOOL_NAMES = FINANCIAL_DOMAIN_TOOL_NAMES | {
    USER_CONTEXT_TOOL_NAME,
    "visualize_allowed_data",
}

_TRUSTED_SCHEMA_FIELDS = ("scope", "trustedScope")


def gemini_safe_schema(node: Any) -> Any:
    """Narrow an MCP JSON Schema to the subset Gemini accepts as a declaration.

    Gemini rejects a union that mixes an array branch with scalar branches,
    which MCP emits for filter values accepting either one scalar or a list.
    One such union made every tool-bound turn fail with INVALID_ARGUMENT, so
    the array branch is dropped and the model offers scalars only. This narrows
    nothing but the declaration: MCP still validates the real call against its
    own unmodified schema.
    """
    if isinstance(node, dict):
        narrowed = {key: gemini_safe_schema(value) for key, value in node.items()}
        for union in ("anyOf", "oneOf"):
            branches = narrowed.get(union)
            if not isinstance(branches, list):
                continue
            kept = [
                branch
                for branch in branches
                if not (isinstance(branch, dict) and branch.get("type") == "array")
            ]
            scalars = [
                branch
                for branch in kept
                if isinstance(branch, dict) and branch.get("type") not in (None, "null")
            ]
            if len(kept) != len(branches) and scalars:
                narrowed[union] = kept
        return narrowed
    if isinstance(node, list):
        return [gemini_safe_schema(item) for item in node]
    return node


def strip_model_identity_fields(node: Any) -> Any:
    """Remove trusted identity properties from a detached model-facing schema."""
    if isinstance(node, dict):
        stripped = {key: strip_model_identity_fields(value) for key, value in node.items()}
        properties = stripped.get("properties")
        if isinstance(properties, dict):
            for field in _TRUSTED_SCHEMA_FIELDS:
                properties.pop(field, None)
        required = stripped.get("required")
        if isinstance(required, list):
            stripped["required"] = [
                value for value in required if value not in set(_TRUSTED_SCHEMA_FIELDS)
            ]
        return stripped
    if isinstance(node, list):
        return [strip_model_identity_fields(value) for value in node]
    return node


def model_tool_schema(tool: MCPToolDefinition) -> dict[str, Any]:
    """The declaration a model may read for one advertised tool."""
    schema = gemini_safe_schema(deepcopy(tool.input_schema))
    if tool.name in _IDENTITY_BEARING_TOOL_NAMES:
        schema = strip_model_identity_fields(schema)
    # `call_tool.arguments` is declared `object | null`; gemini-3.6-flash
    # populates that union correctly, so it is passed through unchanged.
    return cast("dict[str, Any]", schema)


def discovered_tool_schemas(data: dict[str, Any]) -> dict[str, Any]:
    """Remove trusted identity fields from schemas handed back by search.

    The server's schemas legitimately declare `scope`, because the orchestrator
    fills it in. Showing it to the model would invite a fabricated user id that
    `enforce_trusted_user_scope` then has to overwrite; better that the model
    never sees the field at all.
    """
    result = data.get("result")
    if not isinstance(result, list):
        return data
    return {
        **data,
        "result": [
            {**entry, "inputSchema": strip_model_identity_fields(entry["inputSchema"])}
            if isinstance(entry, dict) and isinstance(entry.get("inputSchema"), dict)
            else entry
            for entry in result
        ],
    }


def tool_definitions(state: GraphState) -> list[MCPToolDefinition]:
    """Every tool the endpoint advertises, model-facing or not."""
    definitions: list[MCPToolDefinition] = []
    for value in state.get("available_tools", []):
        name = value.get("name")
        description = value.get("description")
        schema = value.get("input_schema")
        if isinstance(name, str) and isinstance(description, str) and isinstance(schema, dict):
            definitions.append(
                MCPToolDefinition(name, description, schema, value.get("model_visible", True))
            )
    return definitions


def model_tool_definitions(state: GraphState) -> list[MCPToolDefinition]:
    """Only what the server declares a model may see.

    Some tools are advertised purely so the orchestrator can address them by
    name; the server marks those app-only and the host - this graph - is what
    keeps them out of the prompt.
    """
    return [tool for tool in tool_definitions(state) if tool.model_visible]
