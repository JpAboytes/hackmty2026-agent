"""Coarse lifecycle status the client may show while a turn runs.

Only the eight identifiers below ever leave the graph. They say *which phase*
the turn is in and nothing about *what the model is thinking*: no prompt text,
no reasoning, no tool arguments, no rows. That boundary is the whole point of
this module, so the payload builder is the only way to produce an event and it
accepts nothing but a validated identifier.

Transport is LangGraph's custom stream channel
(``get_stream_writer`` -> ``astream(stream_mode="custom")``), so a node reports
progress without adding a field to ``GraphState`` and without the HTTP boundary
polling anything.
"""

from __future__ import annotations

from typing import Literal, get_args

from langgraph.config import get_stream_writer

AgentStatus = Literal[
    "interpreting",
    "discovering_tools",
    "selecting_tools",
    "executing_tools",
    "interpreting_results",
    "preparing_action",
    "building_ui",
    "validating_ui",
]

#: The complete vocabulary. The client maps these to copy; nothing else is sent.
AGENT_STATUSES: frozenset[str] = frozenset(get_args(AgentStatus))

#: The discriminator the client switches on in the stream.
AGENT_STATUS_EVENT = "agent_status"


def status_event(status: AgentStatus) -> dict[str, str]:
    """The one payload shape a lifecycle event may have."""
    if status not in AGENT_STATUSES:
        raise ValueError(f"unknown agent status: {status!r}")
    return {"type": AGENT_STATUS_EVENT, "status": status}


def emit_status(status: AgentStatus) -> None:
    """Report one coarse phase, or do nothing when nobody is streaming.

    A non-streaming ``ainvoke`` (the plain POST route, and most tests) has no
    writer in its runnable context. That is not an error: progress reporting is
    optional by design, so the turn proceeds exactly as before.
    """
    event = status_event(status)
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    if writer is None:
        return
    writer(event)


__all__ = [
    "AGENT_STATUSES",
    "AGENT_STATUS_EVENT",
    "AgentStatus",
    "emit_status",
    "status_event",
]
