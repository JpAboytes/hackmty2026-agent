# FluidBank Orchestrator

LangGraph agent for the accessibility-first banking demo. It fetches real financial and accessibility context through the read-only `hackmty2026-mcp` server, drafts a grounded conversational reply with Gemini, and relays whatever A2UI content that server returns to the mobile client - it never constructs or edits A2UI itself. See `PROJECT_SPEC.MD` for the full architecture and contract.

## Layout

```text
src/fluidbank_orchestrator/
  graph.py         LangGraph workflow: fetch_context -> intent
  state.py         Graph state (TypedDict)
  mcp_client.py     Remote MCP client: validated config, Horizon bearer auth
  personas.py       Fixed demo persona ids (seeded by hackmty2026-mcp)
  api.py            FastAPI entrypoint
scripts/
  run_local.py      Run the graph once from the CLI, without an HTTP server
langgraph.json      LangGraph CLI / Studio manifest
```

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
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
  -d '{"query": "Tengo dinero para el fin de semana?", "persona": "ana"}'
```

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
./.venv/bin/ruff check .
./.venv/bin/pytest
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
curl http://127.0.0.1:8080/healthz
```
