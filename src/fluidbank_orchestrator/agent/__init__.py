"""The LangGraph agent: how a turn decides, retrieves, and presents.

Modules:

* ``model``           - the ``ToolAwareModel`` port and its ``ModelTurn`` result.
* ``gemini``          - the only Gemini-specific code: prompt, adapter, answer schema.
* ``tool_visibility`` - what a model may see of the tool surface.
* ``observations``    - the turn's ledger of verified MCP results.
* ``status``          - the coarse lifecycle phases a client may display.
* ``tool_loop``       - executing pending calls and recording provenance.
* ``nodes``           - each workflow step, plus the routing decision.

``fluidbank_orchestrator.graph`` composes these into the compiled workflow.
"""

from __future__ import annotations

from .gemini import GeminiToolAwareModel
from .model import ModelTurn, ToolAwareModel
from .nodes import ToolLoader
from .tool_loop import ToolExecutor

__all__ = [
    "GeminiToolAwareModel",
    "ModelTurn",
    "ToolAwareModel",
    "ToolExecutor",
    "ToolLoader",
]
