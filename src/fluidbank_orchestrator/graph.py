"""LangGraph model/tool loop for conversational and A2UI requests."""

from __future__ import annotations

import json
import logging
import os
import unicodedata
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, cast

from google import genai
from google.genai import types
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .mcp_client import (
    MCPConfigurationError,
    MCPToolDefinition,
    MCPToolExecution,
    UserContextError,
    execute_remote_tool,
    fetch_user_context,
    list_remote_tools,
)
from .state import GraphState, UserProfile

logger = logging.getLogger(__name__)

_FALLBACK_PROFILE: UserProfile = {
    "literacy_level": "medium",
    "font_scale": "lg",
    "contrast": "high",
    "hit_target": "large",
    "overdraft_risk": 0.82,
    "recurring_expenses": 3200.0,
    "available_balance": 1200.0,
}
_MAX_TOOL_TURNS = 8
_NUMERIC_TYPES = (
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "DECIMAL",
    "NUMERIC",
    "REAL",
    "DOUBLE",
    "FLOAT",
    "MONEY",
)
_DATE_TYPES = ("DATE", "TIME", "TIMESTAMP")


class _Intent(BaseModel):
    message: str
    months: int | None = None


@dataclass(frozen=True, slots=True)
class ModelTurn:
    message: str
    tool_calls: tuple[dict[str, Any], ...] = ()
    months: int | None = None


class ToolAwareModel(Protocol):
    async def generate(
        self,
        *,
        query: str,
        profile: UserProfile,
        tools: Sequence[MCPToolDefinition],
        observations: Sequence[Mapping[str, Any]],
    ) -> ModelTurn: ...


_MODEL_PROMPT = """Eres un asistente bancario accesible y conciso. Responde en español.
Usa exclusivamente los datos proporcionados y las herramientas MCP disponibles.
Para solicitudes explícitas de gráfica o visualización, usa visualize_allowed_data.
Nunca inventes esquemas, tablas o columnas: descubre primero con list_allowed_tables y
describe_table. Usa area para tendencias ordenadas y comparaciones; usa heatmap para
actividad o intensidad por fecha. No fuerces gráficas para preguntas de valores o texto.
No generes ni copies JSON A2UI: el puente de la aplicación conserva el resultado MCP.
El ámbito de usuario lo aplica la aplicación: nunca elijas ni cambies scope, user_id,
customer_id, account_id, owner_id o persona_id.

Consulta: {query}
Perfil: {profile}
Observaciones MCP anteriores: {observations}
"""


def _gemini_safe_schema(node: Any) -> Any:
    """Narrow an MCP JSON Schema to the subset Gemini accepts as a declaration.

    Gemini rejects a union that mixes an array branch with scalar branches,
    which MCP emits for filter values accepting either one scalar or a list.
    One such union made every tool-bound turn fail with INVALID_ARGUMENT, so
    the array branch is dropped and the model offers scalars only. This narrows
    nothing but the declaration: MCP still validates the real call against its
    own unmodified schema.
    """
    if isinstance(node, dict):
        narrowed = {key: _gemini_safe_schema(value) for key, value in node.items()}
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
        return [_gemini_safe_schema(item) for item in node]
    return node


class GeminiToolAwareModel:
    """Gemini adapter that receives the exact runtime MCP tool schemas."""

    async def generate(
        self,
        *,
        query: str,
        profile: UserProfile,
        tools: Sequence[MCPToolDefinition],
        observations: Sequence[Mapping[str, Any]],
    ) -> ModelTurn:
        declarations = [
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=_gemini_safe_schema(deepcopy(tool.input_schema)),
            )
            for tool in tools
        ]
        prompt = _MODEL_PROMPT.format(
            query=query,
            profile=json.dumps(profile, ensure_ascii=False, sort_keys=True),
            observations=json.dumps(list(observations), ensure_ascii=False, sort_keys=True),
        )
        try:
            response = await genai.Client().aio.models.generate_content(
                model=os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"),
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(function_declarations=declarations)]
                    if declarations
                    else None,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    response_mime_type="application/json",
                    response_schema=_Intent,
                ),
            )
            allowed = {tool.name for tool in tools}
            calls: list[dict[str, Any]] = []
            for function_call in response.function_calls or []:
                if function_call.name not in allowed:
                    continue
                arguments = function_call.args
                if not isinstance(arguments, Mapping):
                    continue
                calls.append({"name": function_call.name, "arguments": dict(arguments)})
            if calls:
                return ModelTurn(message="", tool_calls=tuple(calls[:2]))
            parsed = response.parsed
            intent = parsed if isinstance(parsed, _Intent) else _Intent.model_validate(parsed)
            return ModelTurn(message=intent.message, months=intent.months)
        except Exception as exc:  # noqa: BLE001 - deterministic policies remain available
            logger.warning("Gemini model turn failed (%s)", type(exc).__name__)
        return ModelTurn(message="No pude generar una respuesta personalizada en este momento.")


ToolLoader = Callable[[], Awaitable[list[MCPToolDefinition]]]
ToolExecutor = Callable[[str, Mapping[str, Any] | None], Awaitable[MCPToolExecution]]


_OWNERSHIP_FILTER_COLUMNS = frozenset(
    {
        "user_id",
        "customer_id",
        "account_id",
        "from_account_id",
        "to_account_id",
        "owner_id",
        "persona_id",
    }
)


def _business_filters(value: object, *, users_table: bool = False) -> list[object]:
    filters = list(value) if isinstance(value, list) else []
    ownership_columns = _OWNERSHIP_FILTER_COLUMNS | ({"id"} if users_table else set())
    return [
        item
        for item in filters
        if not (isinstance(item, Mapping) and item.get("column") in ownership_columns)
    ]


def _scope_tool_arguments(
    name: str, arguments: Mapping[str, Any], current_user_id: str
) -> dict[str, Any]:
    """Overwrite model-controlled ownership inputs with canonical graph state."""
    scoped = deepcopy(dict(arguments))
    canonical_scope = {"user_id": current_user_id}
    if name == "select_rows":
        scoped["scope"] = canonical_scope
        scoped["filters"] = _business_filters(
            scoped.get("filters"),
            users_table=scoped.get("schema") == "public" and scoped.get("table") == "users",
        )
        return scoped
    if name == "visualize_allowed_data":
        raw_request = scoped.get("request")
        request = deepcopy(dict(raw_request)) if isinstance(raw_request, Mapping) else {}
        request["scope"] = canonical_scope
        source = request.get("source")
        request["filters"] = _business_filters(
            request.get("filters"),
            users_table=(
                isinstance(source, Mapping)
                and source.get("schema") == "public"
                and source.get("table") == "users"
            ),
        )
        scoped["request"] = request
    return scoped


def _normalized(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )


def _visualization_kind(query: str) -> str | None:
    text = _normalized(query)
    explicit = (
        "show me a chart",
        "chart",
        "graph",
        "visualiz",
        "trend",
        "over time",
        "daily activity",
        "calendar heatmap",
        "compare these series",
        "grafica",
        "tendencia",
        "a lo largo del tiempo",
        "actividad diaria",
        "mapa de calor",
        "comparar estas series",
    )
    if not any(term in text for term in explicit):
        return None
    heatmap = (
        "heatmap",
        "mapa de calor",
        "calendar",
        "calendario",
        "daily activity",
        "actividad diaria",
    )
    return "heatmap" if any(term in text for term in heatmap) else "area"


def _tool_definitions(state: GraphState) -> list[MCPToolDefinition]:
    definitions: list[MCPToolDefinition] = []
    for value in state.get("available_tools", []):
        name = value.get("name")
        description = value.get("description")
        schema = value.get("input_schema")
        if isinstance(name, str) and isinstance(description, str) and isinstance(schema, dict):
            definitions.append(MCPToolDefinition(name, description, schema))
    return definitions


def _observations(state: GraphState, name: str) -> list[dict[str, Any]]:
    return [item for item in state.get("tool_observations", []) if item.get("name") == name]


def _successful_data(observation: Mapping[str, Any]) -> dict[str, Any] | None:
    data = observation.get("data")
    if observation.get("is_error") is True or not isinstance(data, dict) or data.get("ok") is False:
        return None
    return data


def _table_score(query: str, table: str) -> int:
    text = _normalized(query)
    name = _normalized(table)
    score = sum(2 for part in name.replace("_", " ").split() if part in text)
    aliases = {
        "transactions": (
            "activity",
            "actividad",
            "transaction",
            "transaccion",
            "spending",
            "gasto",
        ),
        "subscriptions": ("subscription", "suscripcion", "charge", "cargo"),
        "accounts": ("balance", "saldo", "account", "cuenta"),
    }
    score += sum(5 for term in aliases.get(name, ()) if term in text)
    if name == "transactions" and any(term in text for term in ("activity", "actividad")):
        score += 5
    return score


def _chart_columns(description: Mapping[str, Any], *, kind: str) -> tuple[str, list[str]] | None:
    columns = description.get("columns")
    if not isinstance(columns, list):
        return None
    dates: list[str] = []
    numerics: list[str] = []
    for column in columns:
        if not isinstance(column, Mapping):
            continue
        name = column.get("name")
        data_type = column.get("data_type")
        if not isinstance(name, str) or not isinstance(data_type, str):
            continue
        if name in _OWNERSHIP_FILTER_COLUMNS or name == "id":
            continue
        upper = data_type.upper()
        if (kind == "heatmap" and upper.startswith("DATE")) or (
            kind == "area" and upper.startswith(_DATE_TYPES)
        ):
            dates.append(name)
        if upper.startswith(_NUMERIC_TYPES):
            numerics.append(name)
    if not dates or not numerics:
        return None
    return dates[0], numerics[:4]


def _required_visualization_turn(state: GraphState, kind: str) -> ModelTurn:
    tools = {tool.name for tool in _tool_definitions(state)}
    if "visualize_allowed_data" not in tools:
        return ModelTurn(
            message="No puedo crear la visualización porque el MCP activo no ofrece "
            "visualize_allowed_data."
        )
    if "list_allowed_tables" not in tools or "describe_table" not in tools:
        return ModelTurn(
            message="No puedo crear la visualización porque el MCP no ofrece descubrimiento "
            "seguro del esquema."
        )

    listed = next(
        (
            data
            for observation in reversed(_observations(state, "list_allowed_tables"))
            if (data := _successful_data(observation)) is not None
        ),
        None,
    )
    if listed is None:
        return ModelTurn(
            message="",
            tool_calls=({"name": "list_allowed_tables", "arguments": {}},),
        )
    objects = listed.get("objects")
    if not isinstance(objects, list) or not objects:
        return ModelTurn(
            message="No hay tablas permitidas disponibles para crear la visualización."
        )

    described: dict[tuple[str, str], dict[str, Any]] = {}
    for observation in _observations(state, "describe_table"):
        data = _successful_data(observation)
        if data is None:
            continue
        schema = data.get("schema")
        table = data.get("table")
        if isinstance(schema, str) and isinstance(table, str):
            described[(schema, table)] = data

    candidates: list[tuple[int, str, str]] = []
    for item in objects:
        if not isinstance(item, Mapping):
            continue
        schema = item.get("schema")
        table = item.get("table")
        if isinstance(schema, str) and isinstance(table, str):
            candidates.append((_table_score(state["user_query"], table), schema, table))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    for _score, schema, table in candidates:
        description = described.get((schema, table))
        if description is None:
            return ModelTurn(
                message="",
                tool_calls=(
                    {
                        "name": "describe_table",
                        "arguments": {"schema": schema, "table": table},
                    },
                ),
            )
        columns = _chart_columns(description, kind=kind)
        if columns is None:
            continue
        date_column, value_columns = columns
        if kind == "heatmap":
            visualization: dict[str, Any] = {
                "kind": "heatmap",
                "date_column": date_column,
                "value_column": value_columns[0],
                "initial_view": "month",
            }
        else:
            visualization = {
                "kind": "area",
                "x_column": date_column,
                "y_columns": value_columns,
            }
        return ModelTurn(
            message="",
            tool_calls=(
                {
                    "name": "visualize_allowed_data",
                    "arguments": {
                        "request": {
                            "source": {"schema": schema, "table": table},
                            "order": [{"column": date_column, "direction": "asc"}],
                            "limit": 100,
                            "visualization": visualization,
                        }
                    },
                },
            ),
        )
    return ModelTurn(
        message="No encontré una tabla permitida con columnas de fecha y valor numérico para "
        "esa visualización."
    )


def _text_from_execution(execution: MCPToolExecution) -> str:
    for content in execution.result.content:
        text = getattr(content, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return "La herramienta terminó sin una respuesta de texto."


def _chat_message_call(text: str) -> dict[str, Any]:
    """Build the one tool call every plain final answer must end on.

    The MCP chat_message tool wraps drafted text as a validated A2UI text
    surface - this is how the agent guarantees it never returns bare text.
    """
    return {"name": "chat_message", "arguments": {"request": {"text": text}}}


def build_graph(
    *,
    model: ToolAwareModel | None = None,
    tool_loader: ToolLoader = list_remote_tools,
    tool_executor: ToolExecutor = execute_remote_tool,
) -> Any:
    """Build an injectable graph with an explicit model -> tools -> model loop."""
    resolved_model = model or GeminiToolAwareModel()

    async def load_tools_node(_state: GraphState) -> GraphState:
        try:
            tools = await tool_loader()
            return {"available_tools": [tool.as_dict() for tool in tools]}
        except (MCPConfigurationError, UserContextError):
            return {"available_tools": []}

    async def fetch_context_node(state: GraphState) -> GraphState:
        try:
            profile = await fetch_user_context(state["current_user_id"])
        except (MCPConfigurationError, UserContextError):
            # The fallback profile carries placeholder figures. They keep the
            # graph runnable, but they are nobody's money, so the turn is
            # marked unresolved rather than letting them be reported as real.
            return {"user_profile": _FALLBACK_PROFILE.copy(), "context_available": False}
        return {"user_profile": profile, "context_available": True}

    async def agent_node(state: GraphState) -> GraphState:
        if state.get("context_available") is False:
            if _observations(state, "chat_message"):
                return {"tool_calls": []}
            return {
                "message": "",
                "tool_calls": [
                    _chat_message_call(
                        "No pude identificar tu cuenta, así que no puedo mostrarte "
                        "cifras. Vuelve a iniciar sesión e inténtalo de nuevo."
                    )
                ],
            }
        observations = state.get("tool_observations", [])
        visualization_kind = _visualization_kind(state["user_query"])
        latest_visualization = _observations(state, "visualize_allowed_data")
        if visualization_kind is not None and latest_visualization:
            return {
                "message": str(latest_visualization[-1].get("text") or "La visualización terminó."),
                "tool_calls": [],
            }
        if _observations(state, "chat_message"):
            # The final answer was already wrapped as A2UI by chat_message; stop
            # here instead of asking the model to draft another turn.
            return {"tool_calls": []}
        if state.get("tool_loop_count", 0) >= _MAX_TOOL_TURNS:
            limit_text = (
                "No pude completar la visualización dentro del límite seguro de pasos."
            )
            return {"message": "", "tool_calls": [_chat_message_call(limit_text)]}

        candidate = await resolved_model.generate(
            query=state["user_query"],
            profile=state["user_profile"],
            tools=_tool_definitions(state),
            observations=observations,
        )
        if visualization_kind is not None:
            candidate = _required_visualization_turn(state, visualization_kind)
        else:
            candidate = ModelTurn(
                message=candidate.message,
                tool_calls=tuple(
                    call
                    for call in candidate.tool_calls
                    if call.get("name") != "visualize_allowed_data"
                ),
                months=candidate.months,
            )
        if candidate.tool_calls:
            result: GraphState = {
                "message": "",
                "tool_calls": [dict(call) for call in candidate.tool_calls],
            }
        else:
            # Every final answer must reach the client as A2UI, never as bare
            # text: wrap it through the MCP chat_message tool instead of ending.
            message_text = candidate.message.strip() or (
                "No tengo una respuesta para mostrar en este momento."
            )
            result = {"message": "", "tool_calls": [_chat_message_call(message_text)]}
        if candidate.months is not None:
            result["months"] = candidate.months
        return result

    async def tools_node(state: GraphState) -> GraphState:
        available = {tool.name for tool in _tool_definitions(state)}
        observations = list(state.get("tool_observations", []))
        update: GraphState = {
            "tool_calls": [],
            "tool_loop_count": state.get("tool_loop_count", 0) + 1,
        }
        for call in state.get("tool_calls", []):
            name = call.get("name")
            arguments = call.get("arguments")
            if (
                not isinstance(name, str)
                or name not in available
                or not isinstance(arguments, dict)
            ):
                observations.append(
                    {
                        "name": str(name),
                        "arguments": {},
                        "is_error": True,
                        "data": {},
                        "text": "La herramienta solicitada no está disponible.",
                    }
                )
                continue
            scoped_arguments = _scope_tool_arguments(name, arguments, state["current_user_id"])
            try:
                execution = await tool_executor(name, scoped_arguments)
                structured = execution.result.structured_content
                observations.append(
                    {
                        "name": name,
                        "arguments": deepcopy(arguments),
                        "is_error": bool(execution.result.is_error),
                        "data": deepcopy(structured) if isinstance(structured, dict) else {},
                        "text": _text_from_execution(execution),
                    }
                )
                if name in {"visualize_allowed_data", "database_overview", "chat_message"}:
                    update["final_tool_execution"] = execution
            except (MCPConfigurationError, UserContextError):
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

    def route_after_agent(state: GraphState) -> str:
        return "tools" if state.get("tool_calls") else END

    workflow = StateGraph(GraphState)
    workflow.add_node("load_tools", cast("Any", load_tools_node))
    workflow.add_node("fetch_context", cast("Any", fetch_context_node))
    workflow.add_node("agent", cast("Any", agent_node))
    workflow.add_node("tools", cast("Any", tools_node))
    workflow.add_edge(START, "load_tools")
    workflow.add_edge("load_tools", "fetch_context")
    workflow.add_edge("fetch_context", "agent")
    workflow.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    return workflow.compile()


graph = build_graph()
