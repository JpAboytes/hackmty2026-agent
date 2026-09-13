"""The only Gemini-specific code in the orchestrator.

It owns the prompt, the bounded answer schema, and the two-request split that
tool declarations and a structured response schema require. Everything it is
handed is already safe to show a model: the tool schemas arrive stripped of
trusted identity fields by ``tool_visibility``, and nothing it returns is
trusted as identity or as protocol values.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel

from ..mcp_client import MCPToolDefinition
from ..observability import preview, stage
from ..schemas.banking_view import FinancialIntent
from ..state import ToolCall, UserProfile
from .model import ModelTurn
from .tool_visibility import model_tool_schema

logger = logging.getLogger(__name__)

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


class _Intent(BaseModel):
    """The bounded answer the model may produce. It selects, it never invents."""

    message: str
    months: int | None = None
    presentation_intent: FinancialIntent | None = None


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
                parameters_json_schema=model_tool_schema(tool),
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
    ) -> list[ToolCall]:
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
        calls: list[ToolCall] = []
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
