"""Run the orchestrator locally against real Supabase-backed context."""

from __future__ import annotations

import asyncio
import json

from dotenv import load_dotenv

from fluidbank_orchestrator.graph import graph
from fluidbank_orchestrator.personas import PERSONA_USER_IDS


async def main() -> None:
    load_dotenv()
    result = await graph.ainvoke(
        {
            "user_query": "Tengo dinero para el fin de semana?",
            "current_user_id": PERSONA_USER_IDS["ana"],
        }
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
