"""LangGraph model/tool loop for conversational and A2UI requests."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, cast
from uuid import UUID

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
    require_current_user_id,
)
from .observability import event, preview, stage
from .schemas.banking_view import FinancialIntent
from .services.financial_presentation import (
    build_financial_presentation,
    classify_financial_request,
    normalize_action_intent,
    select_presentation_intent,
)
from .state import GraphState, UserProfile

logger = logging.getLogger(__name__)

_FALLBACK_PROFILE: UserProfile = {
    "literacy_level": "medium",
    "font_scale": "lg",
    "contrast": "high",
    "hit_target": "large",
    "overdraft_risk": None,
    "recurring_expenses": 0.0,
    "available_balance": None,
    "owned_balances": {},
}
_MAX_TOOL_TURNS = 8


class _Intent(BaseModel):
    message: str
    months: int | None = None
    presentation_intent: FinancialIntent | None = None


@dataclass(frozen=True, slots=True)
class ModelTurn:
    message: str
    tool_calls: tuple[dict[str, Any], ...] = ()
    months: int | None = None
    presentation_intent: FinancialIntent | None = None


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
Después de consultar datos financieros, elige como máximo una semántica de presentación
de la lista permitida. Puedes usar visualize_allowed_data cuando una tendencia o actividad
se beneficie de una gráfica aunque el usuario no diga "gráfica"; no la uses para un saldo escalar.
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
                parameters_json_schema=_model_tool_schema(tool),
            )
            for tool in tools
        ]
        prompt = _MODEL_PROMPT.format(
            query=query,
            profile=json.dumps(profile, ensure_ascii=False, sort_keys=True),
            observations=json.dumps(list(observations), ensure_ascii=False, sort_keys=True),
        )
        model_name = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
        # The prompt carries every prior observation verbatim, so its size is the
        # single number that explains a slow model turn late in a tool loop.
        async with stage(
            "model.gemini",
            model=model_name,
            declared_tools=len(declarations),
            observations=len(observations),
            prompt_chars=len(prompt),
        ) as step:
            try:
                response = await genai.Client().aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(function_declarations=declarations)]
                        if declarations
                        else None,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                        response_mime_type="application/json",
                        response_schema=_Intent,
                    ),
                )
                usage = response.usage_metadata
                if usage is not None:
                    step.set(
                        prompt_tokens=usage.prompt_token_count,
                        output_tokens=usage.candidates_token_count,
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
                    step.set(decision="tool_calls", calls=",".join(call["name"] for call in calls))
                    preview("model.gemini.calls", calls)
                    return ModelTurn(message="", tool_calls=tuple(calls[:2]))
                parsed = response.parsed
                intent = parsed if isinstance(parsed, _Intent) else _Intent.model_validate(parsed)
                step.set(
                    decision="message",
                    message_chars=len(intent.message),
                    presentation_intent=intent.presentation_intent,
                    months=intent.months,
                )
                return ModelTurn(
                    message=intent.message,
                    months=intent.months,
                    presentation_intent=intent.presentation_intent,
                )
            except Exception as exc:  # noqa: BLE001 - deterministic policies remain available
                step.set(decision="failed")
                logger.warning("Gemini model turn failed (%s)", type(exc).__name__)
        return ModelTurn(message="No pude generar una respuesta personalizada en este momento.")


ToolLoader = Callable[[], Awaitable[list[MCPToolDefinition]]]


class ToolExecutor(Protocol):
    async def __call__(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        current_user_id: UUID | None = None,
    ) -> MCPToolExecution: ...


def _strip_model_identity_fields(node: Any) -> Any:
    """Remove trusted identity properties from a detached model-facing schema."""
    if isinstance(node, dict):
        stripped = {key: _strip_model_identity_fields(value) for key, value in node.items()}
        properties = stripped.get("properties")
        if isinstance(properties, dict):
            properties.pop("scope", None)
            properties.pop("trustedScope", None)
        required = stripped.get("required")
        if isinstance(required, list):
            stripped["required"] = [
                value for value in required if value not in {"scope", "trustedScope"}
            ]
        return stripped
    if isinstance(node, list):
        return [_strip_model_identity_fields(value) for value in node]
    return node


def _model_tool_schema(tool: MCPToolDefinition) -> dict[str, Any]:
    schema = _gemini_safe_schema(deepcopy(tool.input_schema))
    if tool.name in {"select_rows", "visualize_allowed_data"}:
        schema = _strip_model_identity_fields(schema)
    return cast("dict[str, Any]", schema)


def _tool_definitions(state: GraphState) -> list[MCPToolDefinition]:
    definitions: list[MCPToolDefinition] = []
    for value in state.get("available_tools", []):
        name = value.get("name")
        description = value.get("description")
        schema = value.get("input_schema")
        if isinstance(name, str) and isinstance(description, str) and isinstance(schema, dict):
            definitions.append(MCPToolDefinition(name, description, schema))
    return definitions


def _text_from_execution(execution: MCPToolExecution) -> str:
    for content in execution.result.content:
        text = getattr(content, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return "La herramienta terminó sin una respuesta de texto."


def _retained_observations(state: GraphState) -> list[dict[str, Any]]:
    """Every verified row set this turn holds, prefetched context included.

    Context rows come first so a later, narrower domain read of the same table
    wins on the deduplicated identifiers.
    """
    return [*state.get("context_observations", []), *state.get("tool_observations", [])]


def _context_observations(rows: Mapping[str, list[dict[str, object]]]) -> list[dict[str, Any]]:
    """Record the profile's scoped reads in the shape a domain read produces."""
    return [
        {
            "name": "select_rows",
            "arguments": {"schema": "public", "table": table},
            "is_error": False,
            "data": {"ok": True, "rows": deepcopy(table_rows)},
            "text": f"Filas de {table} leídas con el contexto del usuario.",
        }
        for table, table_rows in rows.items()
        if table_rows
    ]


def _has_table_observation(state: GraphState, table: str) -> bool:
    return any(
        observation.get("name") == "select_rows"
        and isinstance(observation.get("arguments"), Mapping)
        and observation["arguments"].get("table") == table
        for observation in _retained_observations(state)
    )


def _financial_data_turn(state: GraphState, intent: FinancialIntent) -> ModelTurn:
    """Plan only bounded domain reads; presentation is selected in a later node."""
    available = {tool.name for tool in _tool_definitions(state)}
    table_by_intent = {
        "financial-summary": "accounts",
        "transactions": "transactions",
        "spending-analysis": "transactions",
        "recurring-payments": "subscriptions",
    }
    table = table_by_intent.get(intent)
    if table is None or _has_table_observation(state, table):
        return ModelTurn(message="")
    if "select_rows" not in available:
        return ModelTurn(message="No está disponible la consulta financiera requerida.")

    columns_by_table = {
        "accounts": ["id", "account_type", "currency", "available_balance"],
        "transactions": ["id", "amount", "direction", "category", "merchant", "occurred_at"],
        "subscriptions": [
            "id",
            "name",
            "amount",
            "billing_cycle",
            "next_charge_date",
            "status",
        ],
    }
    arguments: dict[str, Any] = {
        "schema": "public",
        "table": table,
        "columns": columns_by_table[table],
        "limit": 100 if table == "transactions" else 50,
    }
    if table == "transactions":
        arguments["order_by"] = [{"column": "occurred_at", "direction": "desc"}]
        period_end = datetime.now(UTC)
        period_start = period_end - timedelta(days=30)
        arguments["filters"] = [
            {
                "column": "occurred_at",
                "operator": "gte",
                "value": period_start.isoformat(),
            },
            {
                "column": "occurred_at",
                "operator": "lte",
                "value": period_end.isoformat(),
            },
        ]
    return ModelTurn(
        message="",
        tool_calls=({"name": "select_rows", "arguments": arguments},),
    )


def build_graph(
    *,
    model: ToolAwareModel | None = None,
    tool_loader: ToolLoader = list_remote_tools,
    tool_executor: ToolExecutor = execute_remote_tool,
) -> Any:
    """Build an injectable graph with an explicit model -> tools -> model loop."""
    resolved_model = model or GeminiToolAwareModel()

    async def validate_identity_node(state: GraphState) -> GraphState:
        return {"current_user_id": require_current_user_id(state.get("current_user_id"))}

    async def load_tools_node(_state: GraphState) -> GraphState:
        async with stage("node.load_tools") as step:
            try:
                tools = await tool_loader()
            except (MCPConfigurationError, UserContextError) as exc:
                step.set(outcome="unavailable", reason=type(exc).__name__, tools=0)
                return {"available_tools": []}
            step.set(outcome="loaded", tools=len(tools))
            return {"available_tools": [tool.as_dict() for tool in tools]}

    async def fetch_context_node(state: GraphState) -> GraphState:
        async with stage("node.fetch_context") as step:
            try:
                current_user_id = require_current_user_id(state.get("current_user_id"))
                context = await fetch_user_context(current_user_id)
            except (MCPConfigurationError, UserContextError) as exc:
                # Missing context must never become invented financial data.
                step.set(outcome="fallback", reason=type(exc).__name__)
                return {"user_profile": _FALLBACK_PROFILE.copy(), "context_available": False}
            retained = _context_observations(context.rows)
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

    async def agent_node(state: GraphState) -> GraphState:
        async with stage(
            "node.agent",
            turn=state.get("tool_loop_count", 0),
            observations=len(state.get("tool_observations", [])),
        ) as step:
            return await _agent_turn(state, step)

    async def _agent_turn(state: GraphState, step: stage) -> GraphState:
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
            candidate = _financial_data_turn(state, financial_intent)
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
            return {"message": _text_from_execution(final_execution), "tool_calls": []}
        if state.get("tool_loop_count", 0) >= _MAX_TOOL_TURNS:
            step.set(decision="loop_limit", limit=_MAX_TOOL_TURNS)
            return {
                "message": "No pude completar la consulta dentro del límite seguro de pasos.",
                "tool_calls": [],
            }

        candidate = await resolved_model.generate(
            query=state["user_query"],
            profile=state["user_profile"],
            tools=_tool_definitions(state),
            observations=observations,
        )
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

    async def tools_node(state: GraphState) -> GraphState:
        pending = state.get("tool_calls", [])
        async with stage("node.tools", calls=len(pending)):
            return await _run_tools(state, pending)

    async def _run_tools(state: GraphState, pending: list[dict[str, Any]]) -> GraphState:
        available = {tool.name for tool in _tool_definitions(state)}
        observations = list(state.get("tool_observations", []))
        update: GraphState = {
            "tool_calls": [],
            "tool_loop_count": state.get("tool_loop_count", 0) + 1,
        }
        for call in pending:
            name = call.get("name")
            arguments = call.get("arguments")
            if (
                not isinstance(name, str)
                or name not in available
                or not isinstance(arguments, dict)
            ):
                event("tool.rejected", name=str(name), reason="unavailable")
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
            preview(f"tool.{name}.arguments", arguments)
            try:
                async with stage(
                    "tool.call",
                    name=name,
                    table=arguments.get("table"),
                ) as step:
                    execution = await tool_executor(
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
                observations.append(
                    {
                        "name": name,
                        "arguments": deepcopy(arguments),
                        "is_error": bool(execution.result.is_error),
                        "data": deepcopy(structured) if isinstance(structured, dict) else {},
                        "text": _text_from_execution(execution),
                    }
                )
                if name in {"visualize_allowed_data", "database_overview"}:
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

    async def select_presentation_node(state: GraphState) -> GraphState:
        async with stage("node.select_presentation") as step:
            requested = normalize_action_intent(state.get("financial_request_intent"))
            if requested is None:
                step.set(outcome="invalid")
                return {"message": "La presentación financiera solicitada no es válida."}
            selected = select_presentation_intent(
                requested,
                _retained_observations(state),
                query=state["user_query"],
                action_requested=state.get("action_requested", False),
            )
            step.set(requested=requested, selected=selected)
            return {"presentation_intent": selected}

    async def build_presentation_node(state: GraphState) -> GraphState:
        async with stage("node.build_presentation") as step:
            intent = normalize_action_intent(state.get("presentation_intent"))
            if intent is None:
                step.set(outcome="invalid")
                return {"message": "La presentación financiera solicitada no es válida."}
            presentation = build_financial_presentation(
                intent,
                _retained_observations(state),
                state["user_profile"],
            )
            step.set(
                intent=intent,
                a2ui_messages=len(presentation.a2ui.messages),
                data_keys=",".join(sorted(presentation.data)) or "none",
            )
            return {"message": presentation.message, "financial_presentation": presentation}

    def route_after_agent(state: GraphState) -> str:
        if state.get("tool_calls"):
            destination = "tools"
        elif normalize_action_intent(state.get("financial_request_intent")) is not None:
            destination = "select_presentation"
        else:
            destination = END
        event("graph.route", node="agent", next=destination)
        return destination

    workflow = StateGraph(GraphState)
    workflow.add_node("validate_identity", cast("Any", validate_identity_node))
    workflow.add_node("load_tools", cast("Any", load_tools_node))
    workflow.add_node("fetch_context", cast("Any", fetch_context_node))
    workflow.add_node("agent", cast("Any", agent_node))
    workflow.add_node("tools", cast("Any", tools_node))
    workflow.add_node("select_presentation", cast("Any", select_presentation_node))
    workflow.add_node("build_presentation", cast("Any", build_presentation_node))
    workflow.add_edge(START, "validate_identity")
    workflow.add_edge("validate_identity", "load_tools")
    workflow.add_edge("load_tools", "fetch_context")
    workflow.add_edge("fetch_context", "agent")
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "select_presentation": "select_presentation", END: END},
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("select_presentation", "build_presentation")
    workflow.add_edge("build_presentation", END)
    return workflow.compile()


graph = build_graph()
