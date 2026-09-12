"""Fixed demo persona ids seeded by hackmty2026-mcp/scripts/seed_demo_data.py.

Kept in sync manually with that script's deterministic uuid5 output. The two
repositories are independently owned per PROJECT_SPEC.MD, so there is no
shared package to derive these ids from automatically.
"""

from __future__ import annotations

from typing import Literal

Persona = Literal["ana", "luis", "sofia"]

PERSONA_USER_IDS: dict[Persona, str] = {
    "ana": "68dc4d66-07b8-5893-95f1-07f06989a552",  # liquidity-crisis scenario
    "luis": "c1a3797d-b335-5a9d-98a1-402311f82c7a",  # subscription-review scenario
    "sofia": "4a1cca3a-485e-5706-8c92-43a3bf78bc3b",  # purchase-simulation, accessible reading mode
}
