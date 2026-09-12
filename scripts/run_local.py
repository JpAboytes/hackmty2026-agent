"""Run the orchestrator locally against real Supabase-backed context."""

from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv

from fluidbank_orchestrator.auth import verify_supabase_access_token
from fluidbank_orchestrator.graph import graph


async def main() -> None:
    load_dotenv()
    access_token = os.environ.get("SUPABASE_ACCESS_TOKEN", "").strip()
    current_user_id = await verify_supabase_access_token(
        f"Bearer {access_token}" if access_token else None
    )
    result = await graph.ainvoke(
        {
            "user_query": "Tengo dinero para el fin de semana?",
            "current_user_id": current_user_id,
        }
    )
    public_result = {key: value for key, value in result.items() if key != "current_user_id"}
    print(json.dumps(public_result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    asyncio.run(main())
