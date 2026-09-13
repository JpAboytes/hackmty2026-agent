"""LangGraph model/tool loop for conversational and A2UI requests."""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
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
    FINANCIAL_DOMAIN_TOOL_NAMES,
    SEARCH_TOOL_NAME,
    MCPConfigurationError,
    MCPToolDefinition,
    MCPToolExecution,
    UserContextError,
    execute_remote_tool,
    fetch_user_context,
    list_remote_tools,
    require_current_user_id,
    resolve_tool_call,
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
de la lista permitida. Las tools financieras ya devuelven contratos semánticos y chart-ready;
no consultes ni interpretes el esquema PostgreSQL.
Descubrimiento de herramientas / Tool discovery:
1. Llama search_tools con una consulta en lenguaje natural que describa la intención
   financiera del usuario, por ejemplo "deudas pendientes" o "gasto por categoría".
2. Lee las definiciones devueltas y llama call_tool con {{"name": <herramienta>,
   "arguments": {{...}}}} usando el esquema que search_tools acaba de darte.
Las definiciones que devuelve search_tools son completas: no vuelvas a buscar la misma
intención si ya obtuviste una herramienta adecuada. Usa la menor cantidad de tools.
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


def _api_failure_reason(exc: Exception) -> str:
    """Bounded, non-sensitive label for a failed model call.

    Only the transport code and the API's own status enum are logged. The
    response body can echo prompt content, so it never reaches a log line.
    """
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    parts = [type(exc).__name__]
    if isinstance(code, int):
        parts.append(str(code))
    if isinstance(status, str) and status.isascii() and len(status) <= 64:
        parts.append(status)
    return "/".join(parts)


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
            client = genai.Client()
            calls = await self._tool_calls(client, model_name, prompt, declarations, tools, step)
            if calls:
                step.set(decision="tool_calls", calls=",".join(call["name"] for call in calls))
                preview("model.gemini.calls", calls)
                return ModelTurn(message="", tool_calls=tuple(calls[:2]))
            turn = await self._answer(client, model_name, prompt, step)
            if turn is not None:
                return turn
        return ModelTurn(message="No pude generar una respuesta personalizada en este momento.")

    async def _tool_calls(
        self,
        client: Any,
        model_name: str,
        prompt: str,
        declarations: list[types.FunctionDeclaration],
        tools: Sequence[MCPToolDefinition],
        step: Any,
    ) -> list[dict[str, Any]]:
        """Ask only which tools to call.

        The declarations and a structured `response_schema` cannot travel in the
        same request: past a modest combined size Gemini answers 400
        INVALID_ARGUMENT and the whole turn is lost. The financial domain tools
        crossed that line, so every unclassified query fell back to the generic
        apology with no data and no A2UI. Tool selection needs no response
        schema, and the answer phase needs no tools.
        """
        if not declarations:
            return []
        try:
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(function_declarations=declarations)],
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - the answer phase still runs
            reason = _api_failure_reason(exc)
            step.set(tool_phase="failed", tool_phase_reason=reason)
            logger.warning("Gemini tool selection failed (%s)", reason)
            return []
        self._record_usage(response, step, prefix="tool_phase")
        allowed = {tool.name for tool in tools}
        calls: list[dict[str, Any]] = []
        for function_call in response.function_calls or []:
            if function_call.name not in allowed:
                continue
            arguments = function_call.args
            if not isinstance(arguments, Mapping):
                continue
            calls.append({"name": function_call.name, "arguments": dict(arguments)})
        return calls

    async def _answer(
        self, client: Any, model_name: str, prompt: str, step: Any
    ) -> ModelTurn | None:
        """Ask for the bounded `_Intent` answer, with no tools declared."""
        try:
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    response_mime_type="application/json",
                    response_schema=_Intent,
                ),
            )
            self._record_usage(response, step)
            parsed = response.parsed
            intent = parsed if isinstance(parsed, _Intent) else _Intent.model_validate(parsed)
        except Exception as exc:  # noqa: BLE001 - deterministic policies remain available
            reason = _api_failure_reason(exc)
            step.set(decision="failed", reason=reason)
            logger.warning("Gemini model turn failed (%s)", reason)
            return None
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

    @staticmethod
    def _record_usage(response: Any, step: Any, prefix: str = "") -> None:
        usage = response.usage_metadata
        if usage is None:
            return
        label = f"{prefix}_" if prefix else ""
        step.set(
            **{
                f"{label}prompt_tokens": usage.prompt_token_count,
                f"{label}output_tokens": usage.candidates_token_count,
            }
        )


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
    if tool.name in FINANCIAL_DOMAIN_TOOL_NAMES | {"select_rows", "visualize_allowed_data"}:
        schema = _strip_model_identity_fields(schema)
    # `call_tool.arguments` is declared `object | null`; gemini-3.6-flash
    # populates that union correctly, so it is passed through unchanged.
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


def _normalized_query(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )


def _already_searched(state: GraphState, arguments: Mapping[str, Any]) -> bool:
    """Whether this exact discovery query already produced tool definitions.

    Re-searching the same intent burns a model turn and returns the same
    definitions. A search that found nothing is allowed to run again with a
    different phrasing, which is the only case worth retrying.
    """
    query = arguments.get("query")
    if not isinstance(query, str):
        return False
    wanted = _normalized_query(query)
    return any(
        observation.get("name") == SEARCH_TOOL_NAME
        and observation.get("is_error") is not True
        and observation.get("data", {}).get("result")
        and isinstance(observation.get("arguments"), Mapping)
        and _normalized_query(str(observation["arguments"].get("query", ""))) == wanted
        for observation in state.get("tool_observations", [])
    )


def _discovered_tool_schemas(data: dict[str, Any]) -> dict[str, Any]:
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
            {**entry, "inputSchema": _strip_model_identity_fields(entry["inputSchema"])}
            if isinstance(entry, dict) and isinstance(entry.get("inputSchema"), dict)
            else entry
            for entry in result
        ],
    }


def _has_tool_observation(state: GraphState, tool_name: str) -> bool:
    return any(
        observation.get("name") == tool_name for observation in state.get("tool_observations", [])
    )


def _financial_data_turn(state: GraphState, intent: FinancialIntent) -> ModelTurn:
    """Plan the smallest bilingual financial-domain read for the request."""
    # This classifier knows which capability an intent needs, so it skips
    # discovery entirely; what it must still check is that MCP answered at all.
    # The capability set is the scoped-execution boundary, never the (now tiny)
    # set of schemas the model was handed.
    available = FINANCIAL_DOMAIN_TOOL_NAMES if _tool_definitions(state) else frozenset()
    text = _normalized_query(state["user_query"])
    tool_by_intent = {
        "financial-summary": "get_financial_overview",
        "transactions": "get_transactions",
        "spending-analysis": "analyze_spending",
        "cash-flow": "get_cash_flow",
        "budgets": "get_budget_progress",
        "recurring-payments": "get_upcoming_payments",
        "credit-card": "get_debt_overview",
        "debts": "get_debt_overview",
        "transfers": "get_payment_activity",
        "card-security": "get_transaction_disputes",
        "savings-goals": "get_savings_progress",
        "banking-information": "get_accounts",
    }

    if intent == "banking-information":
        if any(term in text for term in ("estado de cuenta", "bank statement", "statement")):
            tool_name = "get_bank_statements"
        elif any(term in text for term in ("beneficiario", "beneficiary", "destinatario")):
            tool_name = "get_beneficiaries"
        else:
            tool_name = "get_accounts"
    elif intent == "financial-summary" and any(
        term in text for term in ("alerta", "alert", "aviso", "warning")
    ):
        tool_name = "get_financial_alerts"
    else:
        tool_name = tool_by_intent.get(intent)

    compare_requested = intent == "debts" and any(
        term in text for term in ("compara", "comparar", "escenario", "compare", "scenario")
    )
    if compare_requested:
        overview = next(
            (
                observation
                for observation in state.get("tool_observations", [])
                if observation.get("name") == "get_debt_overview"
                and observation.get("is_error") is not True
            ),
            None,
        )
        if overview is None:
            tool_name = "get_debt_overview"
        else:
            data = overview.get("data")
            debts = data.get("debts") if isinstance(data, Mapping) else None
            debt_id = debts[0].get("id") if isinstance(debts, list) and debts else None
            if not isinstance(debt_id, str):
                return ModelTurn(message="No encontré una deuda guardada para comparar.")
            tool_name = "compare_debt_scenarios"
            arguments = {
                "request": {
                    "debt_id": debt_id,
                    "order_by": "lowest_total_interest",
                }
            }
            if tool_name not in available or _has_tool_observation(state, tool_name):
                return ModelTurn(message="")
            return ModelTurn(message="", tool_calls=({"name": tool_name, "arguments": arguments},))

    if tool_name is None or _has_tool_observation(state, tool_name):
        return ModelTurn(message="")
    if tool_name not in available:
        return ModelTurn(message="No está disponible la consulta financiera requerida.")
    request: dict[str, Any] = {}
    if tool_name in {"get_financial_overview", "get_transactions", "analyze_spending"}:
        request["period"] = "current_month"
    elif tool_name == "get_cash_flow":
        request["period"] = (
            "last_6_months"
            if any(term in text for term in ("seis", "six", "6 meses", "6 months"))
            else "last_12_months"
        )
    elif tool_name == "get_payment_activity":
        request["period"] = "last_90_days"
    if tool_name == "get_transactions" and "uber" in text:
        request["merchant_query"] = "Uber"
    if tool_name == "get_budget_progress" and any(
        term in text for term in ("comida", "alimentos", "food")
    ):
        request["category"] = "food"
    if tool_name == "get_upcoming_payments":
        match = re.search(r"\b(\d{1,3})\s+(?:dias|days)\b", text)
        request["days_ahead"] = min(int(match.group(1)), 365) if match else 30
    return ModelTurn(
        message="",
        tool_calls=({"name": tool_name, "arguments": {"request": request}},),
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
        # Two sources of calls, two reasons they are allowed: the model may only
        # use what the server advertised (the discovery pair), and the
        # deterministic classifier may only use the scoped financial set.
        permitted = {tool.name for tool in _tool_definitions(state)} | FINANCIAL_DOMAIN_TOOL_NAMES
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
                or name not in permitted
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
            # Everything downstream reasons about the domain tool, not the
            # `call_tool` envelope the model wrapped it in.
            target, target_arguments = resolve_tool_call(name, arguments)
            if target == SEARCH_TOOL_NAME and _already_searched(state, arguments):
                event("tool.skipped", name=target, reason="duplicate_search")
                continue
            preview(f"tool.{target}.arguments", target_arguments)
            try:
                async with stage(
                    "tool.call",
                    name=target,
                    discovered=target != name,
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
                data = deepcopy(structured) if isinstance(structured, dict) else {}
                if target == SEARCH_TOOL_NAME:
                    data = _discovered_tool_schemas(data)
                observations.append(
                    {
                        "name": target,
                        "arguments": deepcopy(dict(target_arguments or {})),
                        "is_error": bool(execution.result.is_error),
                        "data": data,
                        "text": _text_from_execution(execution),
                    }
                )
                if target in {"visualize_allowed_data", "database_overview"}:
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
