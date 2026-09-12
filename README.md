# Agent Orchestrator

LangGraph agent for the accessibility-first banking demo. It verifies Supabase sessions, fetches user-scoped financial data through FastMCP, interprets returned data, and chooses a bounded financial presentation. Trusted Python builders produce strict Finance v2 `BankingView` surfaces; Gemini never emits or edits A2UI JSON. The generic bridge remains available for MCP-produced A2UI v0.9.1 surfaces.

```text
Expo
  → POST /api/v1/agent/chat
  → verify Supabase bearer token and derive current_user_id
  → interpret intent and execute scoped MCP data reads
  → interpret all retained results and select a known financial semantic intent
  → trusted builder constructs and validates Finance v2 BankingView
  → Expo receives one self-contained ordered v0.9.1 sequence
```

## Layout

```text
  src/fluidbank_orchestrator/
  graph.py         LangGraph workflow: runtime tools -> agent -> tools -> agent
  state.py         Graph state (TypedDict)
  auth.py          Supabase bearer-token verification
  mcp_client.py     Generic tool execution, validated config, Horizon bearer auth
  schemas/
    a2ui.py         Strict official-envelope and supported-catalog models
    banking_view.py Strict Finance v2 intent/data models
    a2ui_action.py  Strict parser for the temporary action-over-chat transport
  services/
    a2ui_bridge.py  Resource resolution, validation, bounded static-template cache
    financial_presentation.py  Deterministic selection and trusted BankingView builder
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
  -H 'Authorization: Bearer <supabase-access-token>' \
  -H 'Content-Type: application/json' \
  -d '{"query": "¿Cuánto dinero tengo?", "user_id": "<supabase-user-uuid>"}'
```

Configure `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY` (or the legacy `SUPABASE_ANON_KEY`). The API resolves the bearer token through Supabase Auth and requires the compatibility body `user_id` to equal the authenticated subject. It rejects anonymous or unconfirmed users, namespaces the LangGraph thread as `user:<uuid>`, and injects/overwrites `scope.user_id` immediately before scoped MCP reads. Model-provided ownership filters are discarded. Identity never comes from action context, model arguments, email mappings, or demo constants.

A request for database metadata still calls the MCP `database_overview` domain tool. Financial requests enter the normal graph, retain all relevant tool observations, and select their presentation only after data retrieval. Scalar balance questions use `financial-summary`; spending can combine category, trend, and daily-activity views when the same verified data supports them. Users do not need to say “chart” or “visualize,” and charts are not forced into scalar answers.

The selected domain tool chooses the entire A2UI surface through `_meta.ui.resourceUri`. The agent calls FastMCP with `raise_on_error=False` so sanitized error `CallToolResult` objects are preserved instead of becoming generic transport failures. For successful presentations it retains the raw result, reads the static `createSurface` and `updateComponents` resource through MCP, combines it with the embedded dynamic `updateDataModel`, and returns:

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

The bodies above are abbreviated; actual messages are complete JSON objects rather than strings. Static templates are cached by MCP server identity plus resource URI, but included in every response. The cache is bounded, stores no dynamic or financial data, returns detached copies, coalesces concurrent reads, and retains only successfully validated templates. Query results, `updateDataModel` messages, chart rows, and graph state are not globally cached.

Supabase Auth verification at the API boundary and MCP scoping are both mandatory. They complement rather than replace database RLS and a least-privilege MCP database role.

The request body may contain either `query` or a structured `action` with exactly `name`, `surfaceId`, `sourceComponentId`, `timestamp`, and `context`. The legacy serialized action format remains temporarily supported. Before graph re-entry, every action is sent to MCP's `a2ui_action` allowlist with the trusted authenticated UUID in the separate `trustedScope` field. MCP normalizes `request_financial_view` to an enumerated intent; only a successful normalized result re-enters the same authenticated graph. Actions never go through Gemini, and client context never supplies identity.

The official v0.9.1 Basic Catalog, Finance v1, and Finance v2 are accepted. Basic supports `Text`, `Button`, `Card`, and `Column`; Finance v1 adds strict `Chart`; Finance v2 adds strict `BankingView` for the 13 shared intents. Unknown components, actions, styles, IDs, and properties are rejected. `database_overview` remains an integration proof rather than a consumer banking screen.

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
