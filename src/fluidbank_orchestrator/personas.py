"""Canonical identifiers for the fixed users in the demo seed."""

from __future__ import annotations

from typing import Literal

Persona = Literal["ana", "luis", "sofia"]

PERSONA_EMAILS: dict[Persona, str] = {
    "ana": "ana.demo@fluidbank.test",  # liquidity-crisis scenario
    "luis": "luis.demo@fluidbank.test",  # subscription-review scenario
    "sofia": "sofia.demo@fluidbank.test",  # purchase-simulation, accessible reading mode
}

PERSONA_USER_IDS: dict[Persona, str] = {
    "ana": "68dc4d66-07b8-5893-95f1-07f06989a552",
    "luis": "c1a3797d-b335-5a9d-98a1-402311f82c7a",
    "sofia": "4a1cca3a-485e-5706-8c92-43a3bf78bc3b",
}

DEMO_USER_IDS = frozenset(PERSONA_USER_IDS.values())
