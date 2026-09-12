# Agent Orchestrator

LangGraph agent for the accessibility-first banking demo. It fetches financial and accessibility context through the read-only FastMCP server, drafts ordinary conversational replies with Gemini, and bridges MCP-produced A2UI v0.9.1 surfaces to Expo. It never invents components or lets the LLM inspect or modify A2UI messages. See `PROJECT_SPEC.MD` for the full architecture and contract.

```text
Expo
  → POST /api/v1/agent/chat
  → agent selects an MCP domain tool
  → MCP result metadata selects an A2UI resource
  → agent reads/caches the static resource and validates the embedded updateDataModel
  → Expo receives one self-contained ordered v0.9.1 message sequence
```

## Layout

```text
src/fluidbank_orchestrator/
  graph.py         LangGraph workflow: fetch_context -> intent
  state.py         Graph state (TypedDict)
  mcp_client.py     Generic tool execution, validated config, Horizon bearer auth
  schemas/
    a2ui.py         Strict official-envelope and supported-catalog models
    a2ui_action.py  Strict parser for the temporary action-over-chat transport
  services/
    a2ui_bridge.py  Resource resolution, validation, bounded static-template cache
  personas.py       Fixed demo emails used only by the local-run helper
  api.py            FastAPI entrypoint
scripts/
  run_local.py      Run the graph once from the CLI, without an HTTP server
langgraph.json      LangGraph CLI / Studio manifest
```

## Setup

```bash
uv sync --locked --extra dev
cp .env.example .env   # fill in GEMINI_API_KEY, MCP_SERVER_URL, HORIZON_API_KEY
```

## Run

HTTP API:

```bash
./.venv/bin/uvicorn fluidbank_orchestrator.api:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/api/v1/agent/chat \
  -H 'Content-Type: application/json' \
  -d '{"query": "Tengo dinero para el fin de semana?", "email": "ana.demo@fluidbank.test"}'
```

A request for a database overview or available database objects deterministically calls the existing MCP `database_overview` domain tool. That tool selects the entire `a2ui://database/overview` surface through `_meta.ui.resourceUri`; no component-selection tool exists. The agent reads the static `createSurface` and `updateComponents` resource through MCP, combines it with the embedded dynamic `updateDataModel`, and returns:

```json
{
  "message": "Database overview loaded.",
  "data": {},
  "a2ui": {
    "resource_uri": "a2ui://database/overview",
    "messages": [
      { "version": "v0.9.1", "createSurface": {} },
      { "version": "v0.9.1", "updateComponents": {} },
      { "version": "v0.9.1", "updateDataModel": {} }
    ]
  }
}
```

The bodies above are abbreviated; actual messages are complete JSON objects rather than strings. Static templates are cached by MCP server identity plus resource URI, but included in every response. The cache is bounded, stores no dynamic or financial data, returns detached copies, coalesces concurrent reads, and retains only successfully validated templates.

Expo's temporary action serialization is parsed as strict JSON and forwarded directly to `a2ui_action` with exactly `name`, `surfaceId`, `sourceComponentId`, `timestamp`, and `context`. It never goes through Gemini. The MCP action registry remains authoritative for allowed action/component pairs, and action results use the same A2UI bridge.

Only the official v0.9.1 Basic Catalog and Expo's current `Text`, `Button`, `Card`, and `Column` subset are accepted. Charts require a future shared custom catalog. `database_overview` is an integration proof over allowlisted database objects, not a consumer banking screen. Future financial tools that follow the same MCP metadata/resource/update contract need no component-specific bridge code.

One-off local run (no HTTP server):

```bash
./.venv/bin/python scripts/run_local.py
```

LangGraph dev server / Studio (install the CLI separately - it's not a project
dependency, since its resolver is heavy):

```bash
./.venv/bin/pip install "langgraph-cli[inmem]"
./.venv/bin/langgraph dev
```

## Validate changes

```bash
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run mypy
```

## Deploy to Cloud Run

The `Dockerfile` builds a slim, non-root image that runs
`uvicorn fluidbank_orchestrator.api:app` bound to Cloud Run's `$PORT`. Secrets
and endpoint config are never baked into the image - pass them at deploy
time. `GEMINI_API_KEY` and `HORIZON_API_KEY` should live in Secret Manager,
not as plain `--set-env-vars`.

```bash
gcloud secrets create gemini-api-key --data-file=- <<< "$GEMINI_API_KEY"
gcloud secrets create horizon-api-key --data-file=- <<< "$HORIZON_API_KEY"

gcloud run deploy fluidbank-orchestrator \
  --source . \
  --region REPLACE_WITH_REGION \
  --allow-unauthenticated \
  --set-env-vars MCP_SERVER_URL=https://REPLACE_WITH_MCP_DEPLOYMENT.fastmcp.app/mcp,MCP_AUTH_MODE=horizon \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest,HORIZON_API_KEY=horizon-api-key:latest
```

`--source .` builds from the `Dockerfile` via Cloud Build; `.gcloudignore` and
`.dockerignore` keep the build context to `pyproject.toml`, `README.md`, and
`src/`. Verify locally first:

```bash
docker build -t fluidbank-orchestrator:local .
docker run --rm -p 8080:8080 --env-file .env fluidbank-orchestrator:local
curl http://127.0.0.1:8080/health
```
