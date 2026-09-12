"""Fixed demo persona emails seeded by hackmty2026-mcp/scripts/seed_demo_data.py.

Used only by scripts/run_local.py for quick manual testing. The real API
identifies users by the client-supplied email (see api.ChatRequest), never by
a fixed persona name.
"""

from __future__ import annotations

from typing import Literal

Persona = Literal["ana", "luis", "sofia"]

PERSONA_EMAILS: dict[Persona, str] = {
    "ana": "ana.demo@fluidbank.test",  # liquidity-crisis scenario
    "luis": "luis.demo@fluidbank.test",  # subscription-review scenario
    "sofia": "sofia.demo@fluidbank.test",  # purchase-simulation, accessible reading mode
}
