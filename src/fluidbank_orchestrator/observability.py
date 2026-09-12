"""Turn-scoped stage logging for the orchestrator.

Every request is one *turn*. Each meaningful step inside it is one *stage*
that logs its own duration, so a single log stream answers both questions the
agent's behaviour raises: what did it decide at each step, and where did the
wall-clock time go. Stage durations are aggregated per turn and emitted as a
closing timeline, which is what makes latency work possible without a tracing
backend.

Nothing here logs financial values, row contents, tokens or credentials. Only
names, counts, sizes, durations and classified outcomes are recorded. Truncated
payloads are available for local debugging behind ``LOG_PAYLOADS``, never by
default.
"""

from __future__ import annotations

import json
import logging
import logging.config
import os
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass, field
from itertools import count
from time import perf_counter
from types import TracebackType
from typing import Any, Self

# Short on purpose: every stage line carries it, and the module loggers keep
# their own dotted names for anything that is not a stage.
_LOGGER = logging.getLogger("agent")

_DEFAULT_PAYLOAD_LIMIT = 800
_QUIET_LIBRARIES = ("httpx", "httpcore", "hpack", "mcp", "fastmcp", "google_genai", "urllib3")


@dataclass(slots=True)
class _Turn:
    """One request's identity plus its accumulated stage timings."""

    id: str
    started: float
    fields: dict[str, Any] = field(default_factory=dict)
    totals: OrderedDict[str, list[float]] = field(default_factory=OrderedDict)

    def record(self, name: str, duration_ms: float) -> None:
        self.totals.setdefault(name, []).append(duration_ms)

    def elapsed_ms(self) -> float:
        return (perf_counter() - self.started) * 1_000

    def timeline(self) -> str:
        parts = []
        for name, durations in self.totals.items():
            total = sum(durations)
            suffix = f"x{len(durations)}" if len(durations) > 1 else ""
            parts.append(f"{name}={total:.0f}ms{suffix}")
        return " ".join(parts)


_TURN: ContextVar[_Turn | None] = ContextVar("fluidbank_turn", default=None)
_TURN_COUNTER = count(1)


class _TurnFilter(logging.Filter):
    """Stamp every record with the active turn id so lines can be grouped."""

    def filter(self, record: logging.LogRecord) -> bool:
        turn = _TURN.get()
        record.turn = turn.id if turn is not None else "-"
        return True


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def payloads_enabled() -> bool:
    """Whether truncated request/response payloads may be logged at DEBUG."""
    return _enabled("LOG_PAYLOADS")


_configured = False


def configure_logging(*, force: bool = False) -> None:
    """Install one stderr handler on the root logger.

    Without this the module-level ``logger.info`` calls across the package are
    dropped: uvicorn configures only its own loggers and leaves the root logger
    at WARNING, so every stage line would be invisible in Cloud Run.
    """
    global _configured
    if _configured and not force:
        return
    level = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
    if level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        level = "INFO"
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {"turn": {"()": _TurnFilter}},
            "formatters": {
                "stage": {
                    "format": "%(asctime)s %(levelname)-7s [%(turn)s] %(name)s: %(message)s",
                    "datefmt": "%H:%M:%S",
                }
            },
            "handlers": {
                "stage": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                    "formatter": "stage",
                    "filters": ["turn"],
                }
            },
            "root": {"level": level, "handlers": ["stage"]},
            # Transport chatter would bury the agent's own stages; DEBUG opts in.
            "loggers": {
                name: {"level": "DEBUG" if level == "DEBUG" else "WARNING"}
                for name in _QUIET_LIBRARIES
            },
        }
    )
    _configured = True


def _render(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.0f}" if value.is_integer() else f"{value:.1f}"
    if isinstance(value, bool) or value is None:
        return str(value).casefold()
    text = str(value)
    return f'"{text}"' if " " in text else text


def _fields(fields: dict[str, Any]) -> str:
    return " ".join(f"{key}={_render(value)}" for key, value in fields.items() if value is not None)


def start_turn(**fields: Any) -> str:
    """Begin a new turn, discarding any timings inherited from a prior one."""
    turn = _Turn(id=f"t{next(_TURN_COUNTER):04d}", started=perf_counter(), fields=dict(fields))
    _TURN.set(turn)
    _LOGGER.info("turn start %s", _fields(turn.fields))
    return turn.id


def end_turn(**fields: Any) -> None:
    """Close the turn with one line carrying the per-stage timing breakdown."""
    turn = _TURN.get()
    if turn is None:
        return
    summary = _fields({**fields, "total_ms": turn.elapsed_ms()})
    _LOGGER.info("turn done %s | %s", summary, turn.timeline())
    _TURN.set(None)


def event(event_name: str, /, **fields: Any) -> None:
    """Log one instantaneous decision or outcome inside the current turn.

    The label is positional-only so that ``name=`` stays available as a field.
    """
    _LOGGER.info("%s %s", event_name, _fields(fields))


def preview(label: str, value: Any, /, *, limit: int | None = None) -> None:
    """Log a truncated payload at DEBUG, only when ``LOG_PAYLOADS`` is set."""
    if not payloads_enabled() or not _LOGGER.isEnabledFor(logging.DEBUG):
        return
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(value)
    cap = limit if limit is not None else _payload_limit()
    if len(text) > cap:
        text = f"{text[:cap]}...<+{len(text) - cap} chars>"
    _LOGGER.debug("%s payload=%s", label, text)


def _payload_limit() -> int:
    raw = os.environ.get("LOG_PAYLOAD_LIMIT", "").strip()
    try:
        limit = int(raw)
    except ValueError:
        return _DEFAULT_PAYLOAD_LIMIT
    return limit if 0 < limit <= 100_000 else _DEFAULT_PAYLOAD_LIMIT


class stage:
    """Time one step and report its outcome, as a sync or async context manager.

    ``with stage("graph.agent") as step: step.set(decision="tool_calls")`` logs
    the start at DEBUG, the completion at INFO with ``duration_ms``, and adds
    the duration to the turn's timeline. An escaping exception is reported by
    type only and re-raised untouched.
    """

    __slots__ = ("_fields", "_name", "_start", "_status")

    # Positional-only: ``name`` is a common field (a tool's name, for one) and
    # must land in **fields rather than colliding with the stage's own label.
    def __init__(self, stage_name: str, /, **fields: Any) -> None:
        self._name = stage_name
        self._fields: dict[str, Any] = dict(fields)
        self._start = 0.0
        self._status = "ok"

    def set(self, **fields: Any) -> None:
        """Attach fields discovered while the stage was running."""
        self._fields.update(fields)

    def _open(self) -> Self:
        self._start = perf_counter()
        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug("%s start %s", self._name, _fields(self._fields))
        return self

    def _close(self, exc: BaseException | None) -> None:
        duration_ms = (perf_counter() - self._start) * 1_000
        turn = _TURN.get()
        if turn is not None:
            turn.record(self._name, duration_ms)
        if exc is not None:
            self._status = f"error:{type(exc).__name__}"
        fields = _fields({**self._fields, "status": self._status, "duration_ms": duration_ms})
        _LOGGER.info("%s %s", self._name, fields)

    def __enter__(self) -> Self:
        return self._open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._close(exc)

    async def __aenter__(self) -> Self:
        return self._open()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._close(exc)
