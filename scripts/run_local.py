"""Run one graph turn locally, with the full stage timeline on stderr.

Identity still comes only from a real Supabase access token, exactly as it does
over HTTP; the only thing skipped here is the FastAPI layer. Everything after
authentication - tool loading, context, model turns, MCP calls, presentation -
runs as it does in production.

    export SUPABASE_ACCESS_TOKEN='<supabase-access-token>'
    ./.venv/bin/python scripts/run_local.py "¿Cuánto dinero tengo?"

Set LOG_LEVEL=DEBUG and LOG_PAYLOADS=1 to include truncated payloads.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from dotenv import load_dotenv

from fluidbank_orchestrator.auth import verify_supabase_access_token
from fluidbank_orchestrator.graph import graph
from fluidbank_orchestrator.observability import configure_logging, end_turn, start_turn

_DEFAULT_QUERY = "Tengo dinero para el fin de semana?"


async def main() -> None:
    load_dotenv()
    configure_logging()
    query = " ".join(sys.argv[1:]).strip() or _DEFAULT_QUERY
    access_token = os.environ.get("SUPABASE_ACCESS_TOKEN", "").strip()
    current_user_id = await verify_supabase_access_token(
        f"Bearer {access_token}" if access_token else None
    )

    start_turn(input="query", query_chars=len(query), user=str(current_user_id)[:8])
    status = "ok"
    try:
        result = await graph.ainvoke(
            {"user_query": query, "current_user_id": current_user_id},
            config={"configurable": {"thread_id": f"user:{current_user_id}"}},
        )
    except Exception as exc:
        status = f"error:{type(exc).__name__}"
        raise
    finally:
        end_turn(status=status)

    public_result = {key: value for key, value in result.items() if key != "current_user_id"}
    print(json.dumps(public_result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    asyncio.run(main())
