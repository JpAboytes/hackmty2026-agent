"""Run the orchestrator locally against real Supabase-backed context."""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from graph import graph
from personas import PERSONA_USER_IDS


async def main() -> None:
    load_dotenv()
    result = await graph.ainvoke(
        {
            "user_query": "Tengo dinero para el fin de semana?",
            "user_id": PERSONA_USER_IDS["ana"],
        }
    )
    print(result["a2ui_response"].model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
