# Agent Orchestrator

LangGraph agent for the accessibility-first banking demo. It verifies Supabase sessions, fetches user-scoped financial data through FastMCP, interprets returned data, and chooses a bounded financial presentation. Trusted Python builders produce strict Finance v2 `BankingView` surfaces; Gemini never emits or edits A2UI JSON. The generic bridge remains available for MCP-produced A2UI v0.9.1 surfaces.

```text
Expo
  → POST /api/v1/agent/chat  (or /chat/stream for coarse progress lines)
  → verify Supabase bearer token and derive current_user_id
  → the model reasons, discovers capabilities, and selects 1..N MCP tools
  → interpret all retained results and select a known financial semantic intent
    (or ask MCP to prepare one of the four A2UI action forms)
  → trusted builder constructs and validates Finance v2 BankingView
  → Expo receives one self-contained ordered v0.9.1 sequence
```

The graph has five terminal shapes and no pre-graph classifier:

```text
START -> validate_identity -> load_tools -> fetch_context -> agent
agent -> tools -> agent                    (model asked for 1..N tool calls)
agent -> prepare_action -> END             (model selected an A2UI form)
agent -> select_presentation -> build_presentation -> END
agent -> END                               (bounded conversational answer)
```

## Layout

```text
src/fluidbank_orchestrator/
  graph.py          LangGraph topology; the compiled graph and CLI entrypoint
  state.py          Graph state (TypedDict)
  auth.py           Supabase bearer-token verification
  observability.py  Turn-scoped stage logging and per-turn timing timeline
  api/              HTTP boundary: app, CORS, dispatch, actions, responses
  agent/            The agent: model port, Gemini adapter, tool visibility,
                    observation ledger, lifecycle status, tool loop, nodes
  mcp_client/       MCP boundary: config, session, trusted scope, catalog,
                    execution, user context
  schemas/          Strict wire contracts: A2UI, Finance v2, actions, chat
  services/         A2UI bridge and the Finance v2 presentation builders
  a2ui_actions/     Canonical action/input JSON and the four form names
  a2ui_catalogs/    Checked-in Finance v1 catalog
scripts/
  run_local.py      Run the graph once from the CLI, without an HTTP server
docs/
  ARCHITECTURE.md   Module responsibilities, dependency direction, boundaries
  FLOWS.md          The request flows as they actually run today
langgraph.json      LangGraph CLI / Studio manifest
```

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) says where to change a given
behaviour; [`docs/FLOWS.md`](docs/FLOWS.md) walks each request path with the
modules and invariants involved.

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
  -d '{"query": "¿Cuánto dinero tengo?"}'
```

### Progress lines

`POST /api/v1/agent/chat/stream` runs the same turn and reports which phase it is
in as it goes. It answers `application/x-ndjson` with one JSON object per line:
zero or more status lines, then exactly one result carrying the identical
envelope `/chat` would have returned.

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/agent/chat/stream \
  -H 'Authorization: Bearer <supabase-access-token>' \
  -H 'Content-Type: application/json' \
  -d '{"query": "¿En qué gasté este mes?"}'
```

```text
{"type":"agent_status","status":"interpreting"}
{"type":"agent_status","status":"interpreting"}
{"type":"agent_status","status":"discovering_tools"}
{"type":"agent_status","status":"selecting_tools"}
{"type":"agent_status","status":"executing_tools"}
{"type":"agent_status","status":"interpreting_results"}
{"type":"agent_status","status":"building_ui"}
{"type":"agent_status","status":"validating_ui"}
{"type":"result","result":{"message":"…","data":{},"a2ui":{…}}}
```

The vocabulary is exactly those eight ids (`agent/status.py`) and nothing else
ever travels on the channel: `api/__init__.py:_status_line` allowlists the
payload, so a node cannot turn a progress stream into a reasoning stream. No
prompt text, reasoning, tool name, argument or row is emitted. Consecutive
duplicates are legal and mean one state. Clients map ids to copy themselves —
the words live in `HackMTY2026_Mobile/src/features/assistant/agent-status.ts`,
not here. A structured `action` body is dispatched through the same code path as
`/chat` and reported as one `result` line with no phases. See
[`docs/FLOWS.md`](docs/FLOWS.md) §11 for which node emits which id.

Configure `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY` (or the legacy `SUPABASE_ANON_KEY`). The API resolves the bearer token through Supabase Auth and accepts no user UUID in the request body. It rejects anonymous or unconfirmed users, namespaces the LangGraph thread as `user:<uuid>`, and injects/overwrites `scope.user_id` immediately before scoped MCP reads. Model-provided ownership filters are discarded. Identity never comes from action context, model arguments, email mappings, or demo constants.

## Tool discovery

The MCP server no longer advertises its domain catalog. The model-facing part of
`tools/list` carries `search_tools` and `call_tool`; three pinned app-only tools
remain available only to the trusted host. Everything else is found through
discovery, so Gemini is bound to two schemas per turn instead of seventeen.

```text
agent -> search_tools("deudas pendientes") -> agent -> call_tool(...) -> agent
```

A hosted MCP resolves `tools/call` against its advertised catalog, so a tool
hidden by discovery is not callable by name there — it has to be addressed
through `call_tool`. `_wire_call` makes that decision from the live catalog, so
pinning or unpinning a tool server-side needs no change here. The tools the
server does advertise purely so a host can address them (`get_user_context`,
`a2ui_action`, `a2ui_form`) are called directly and are kept out of the prompt
by their own `_meta.ui.visibility` declaration, which this orchestrator honours
as the host the MCP Apps spec expects.

The orchestrator takes the model-facing tool set straight from `tools/list` and
applies no local allowlist; re-deriving the catalog here would put every schema
back into the prompt and defeat the server's discovery. That is deliberately
separate from the security boundary: `SCOPED_TOOL_NAMES` still decides which
tools have their identity fields overwritten with the authenticated UUID, and
`enforce_trusted_user_scope` reaches through the `call_tool` envelope to scope
the tool inside it, so discovery cannot smuggle an unscoped call past it.

Every domain tool is reached the same way: the model searches, reads the
candidate definitions, and calls the ones it needs. `search_tools` returns up to
five ranked candidates and the prompt tells the model to evaluate them rather
than take the first — one call if one suffices, several when the question needs
several. `MAX_CALLS_PER_TURN` (6) only stops a runaway fan-out, and
`MAX_TOOL_TURNS` (8) bounds the iterations. Schemas returned by `search_tools`
are stripped of their trusted `scope` fields before the model sees them.

Search queries are asked for in **English**: the MCP catalog is English-only and
BM25 is lexical, so the prompt has the model translate the user's Spanish intent
(`"spending by category"`) instead of copying their words. MCP also folds
Spanish domain vocabulary onto English catalog terms on the query side, so a
Spanish query still retrieves — the prompt instruction is what makes the ranking
good rather than merely non-zero.

The proxy's two-level envelope is hard for a model to fill in reliably. Measured
against `gemini-3.6-flash`, roughly three calls in ten arrive malformed: the
envelope re-wrapped in itself, the `name` one level too deep, or the tool's
single `request` wrapper dropped or applied twice. Each has exactly one reading
that can validate, so `resolve_tool_call` and `enforce_trusted_user_scope`
normalize them instead of paying a rejected round trip and a retry turn. With
that normalization, ten of ten live model calls reached the domain service, up
from four of eight without it. A payload with no tool name anywhere has no such
reading and is left for the server to refuse.

Every query enters the one graph. It retains all relevant tool observations and selects its presentation only after retrieval: the model proposes one of the 13 intents, the finite vocabulary validates it, and `select_presentation_intent` upgrades a `transactions` request phrased as a spending question only once the rows to analyse are actually present. Scalar balance questions use `financial-summary`; spending can combine category, trend, and daily-activity views when the same verified data supports them. Users do not need to say “chart” or “visualize,” and charts are not forced into scalar answers. `database_overview` is still reachable as a domain tool; nothing routes to it by phrase.

When the request is an operation rather than a question, the model answers with `action_form` instead of an intent, and `prepare_action` asks MCP's `a2ui_form` for that surface. The vocabulary is exactly four names — `budget.create`, `budget.load`, `savings_goal.create`, `savings_goal.load` — declared to Gemini as an enum and re-validated by `a2ui_actions/forms.py:normalize_form_name` before MCP is called. Preparing a form saves nothing; only a later user Button event reaches a write handler. `.update` is absent on purpose: MCP derives an update form from the matching `.load`. The model may read data first (a budget suggested from real spending needs `analyze_spending`) or prepare the form directly when it already has the context.

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

One-off local run (no HTTP server). It skips only authentication; every other
stage runs as it does in production:

```bash
export SUPABASE_ACCESS_TOKEN='<supabase-access-token>'
./.venv/bin/python scripts/run_local.py "¿Cuánto dinero tengo?"
```

Expo's web build is a browser origin and is therefore subject to CORS, while the
native builds are not. With `AGENT_ALLOWED_ORIGINS` unset the API answers
preflight for any localhost port, which is what the Expo dev server binds to;
set it to a comma-separated list of exact origins to serve a deployed web build.
Credentials are disabled because the API authenticates with an explicit
`Authorization` header rather than a cookie.

Tool schemas are loaded once per endpoint and reused for `MCP_TOOLS_CACHE_SECONDS`
(default 300). Concurrent turns that miss together load the collection once, the
cache hands out detached copies, and `MCP_TOOLS_CACHE_SECONDS=0` disables it so a
redeployed MCP is picked up without restarting this service. Watch
`mcp.tools_cache` to see which turns paid for the load.

Under progressive discovery that cache holds two static synthetic tools, so it
no longer needs invalidating: search results are read live from the server on
every query, which makes a newly deployed domain tool discoverable immediately.
FastMCP 4.0.3 emits no `notifications/tools/list_changed` from the server side,
and no client-side invalidation was added for a cache that cannot go stale.

## Reading the agent's behaviour

Every request is one *turn* with a short id, and every step inside it is one
*stage* that logs its own decision and `duration_ms`. The turn closes with a
timeline that aggregates the stages, which is where latency work starts:

```text
17:03:28 INFO [t0001] agent: turn start input=query query_chars=21 user=68dc4d66
17:03:28 INFO [t0001] agent: node.load_tools outcome=loaded tools=5 status=ok duration_ms=20.6
17:03:28 INFO [t0001] agent: node.fetch_context outcome=resolved accounts=1 has_balance=true status=ok duration_ms=41.1
17:03:28 INFO [t0001] agent: mcp.call name=get_user_context is_error=false contents=1 status=ok duration_ms=30.3
17:03:28 INFO [t0001] agent: node.agent turn=0 observations=0 decision=financial_ready intent=financial-summary source=classifier status=ok duration_ms=0.0
17:03:28 INFO [t0001] agent: graph.route node=agent next=select_presentation
17:03:28 INFO [t0001] agent: client.response route=graph message_chars=64 data_keys=currency,owned_balance a2ui=true resource_uri=a2ui://financial/view a2ui_messages=3 a2ui_bytes=2104
17:03:28 INFO [t0001] agent: turn done status=ok total_ms=104.7 | node.load_tools=21ms node.fetch_context=41ms node.agent=0ms node.build_presentation=6ms
17:03:28 INFO [t0001] agent: node.load_tools outcome=loaded tools=2 status=ok duration_ms=20.6
17:03:28 INFO [t0001] agent: node.fetch_context outcome=resolved accounts=1 has_balance=true retained=accounts status=ok duration_ms=41.1
17:03:29 INFO [t0001] agent: model.gemini model=gemini-3.6-flash declared_tools=2 observations=0 prompt_chars=2184 decision=tool_calls calls=search_tools status=ok duration_ms=812.4
17:03:29 INFO [t0001] agent: node.agent turn=0 observations=0 decision=model_tool_calls calls=search_tools status=ok duration_ms=813.0
17:03:29 INFO [t0001] agent: graph.route node=agent next=tools
17:03:29 INFO [t0001] agent: tool.call name=search_tools discovered=false is_error=false a2ui=false status=ok duration_ms=98.7
17:03:30 INFO [t0001] agent: node.agent turn=1 observations=1 decision=model_tool_calls calls=call_tool status=ok duration_ms=744.9
17:03:30 INFO [t0001] agent: tool.call name=get_accounts discovered=true is_error=false a2ui=false status=ok duration_ms=120.4
17:03:31 INFO [t0001] agent: node.agent turn=2 observations=2 decision=model_presentation intent=financial-summary source=model status=ok duration_ms=690.2
17:03:31 INFO [t0001] agent: graph.route node=agent next=select_presentation
17:03:31 INFO [t0001] agent: node.select_presentation requested=financial-summary selected=financial-summary status=ok duration_ms=0.2
17:03:31 INFO [t0001] agent: node.build_presentation intent=financial-summary a2ui_messages=3 data_keys=currency,domain_result,owned_balance,presentation_intent status=ok duration_ms=6.1
17:03:31 INFO [t0001] agent: client.response route=graph message_chars=64 data_keys=currency,domain_result,owned_balance,presentation_intent a2ui=true resource_uri=a2ui://finance/view a2ui_messages=3 a2ui_bytes=2104
17:03:31 INFO [t0001] agent: turn done status=ok total_ms=2531.4 | node.load_tools=21ms node.fetch_context=41ms model.gemini=812ms+745ms+690ms tool.call=99ms+120ms node.build_presentation=6ms
```

`discovered=true` on a `tool.call` means the model addressed that tool through
the `call_tool` proxy rather than by name.

Stage names are stable, so `grep` answers specific questions:

| Question | Filter |
| --- | --- |
| Where did the time go? | `turn done` |
| Did the model run at all, and how big was its prompt? | `model.gemini` |
| Which MCP tools ran, with which table, returning how many rows? | `tool.call` |
| How much of the turn was connection setup? | `mcp.connect` |
| What exactly did the client receive? | `client.response` |
| Why did it take that branch? | `decision=` / `graph.route` |

`node.agent` reports exactly one `decision=` per turn: `model_tool_calls` (the
model chose tools and the loop runs again), `model_action_form` (it chose an
A2UI form, so `node.prepare_action` follows), `model_presentation` (with
`source=model` or `source=requested`), `model_message`, or one of the three
guards `no_context`, `tool_presentation` and `loop_limit`. There is no
`decision=financial_retrieval` or `decision=financial_ready`: the deterministic
router was removed, so every turn that touches data has a `model.gemini` line
before it.

`LOG_LEVEL=DEBUG` adds stage starts and library transport logs.
`LOG_PAYLOADS=1` additionally logs truncated tool arguments, MCP results, and
the exact response body sent to the client - local debugging only, since those
carry real balances. Without it no financial value is ever logged: stages
record names, counts, sizes, durations, and classified outcomes only.

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

If a checked-out `.venv` was created before this repository was moved, its
editable install still points at the old path and every test module fails to
import `fluidbank_orchestrator`. Repair it once with `uv sync --extra dev` (or
`uv pip install -e .`), or drive the interpreter directly with the source tree on
the path:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest
```

The suite is 159 tests and passes clean that way. Console scripts installed into
a relocated venv (`.venv/bin/langgraph`, and the MCP repo's `.venv/bin/fastmcp`)
keep an absolute shebang to the old interpreter and fail with
`FileNotFoundError` until the same `uv sync` rewrites them.

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

## A2UI input forms

The Expo renderer supports a strict v0.9.1 subset of TextField, DateTimeInput (date only), Slider, ChoicePicker and Button. Requests such as “Crea un presupuesto”, “Edita un presupuesto”, “Crea una meta de ahorro”, “Quiero hacer una transferencia”, “Transfiere $500 a Ana” and “Quiero pagar mi tarjeta” deterministically prepare the matching MCP `a2ui_form`; this preparation does not save anything. The transfer form loads selectable accounts and contacts from the authenticated user's current MCP data, and an explicit amount or valid recipient can prefill it. The card-payment form uses Finance v2 to show the masked payment card and current terms. “Muéstrame mi tarjeta de crédito” uses the dedicated `get_credit_cards` tool and builds the same verified `PaymentCard` view; legacy `get_accounts` observations remain readable. Save submits a structured `action` in the authenticated HTTP body, using the five A2UI fields and explicit resolved context. The orchestrator supplies trustedScope from the verified token; model-supplied ownership never wins. The LLM cannot call a2ui_action. The MCP returns data.actionResult and the client displays success/failure rather than interpreting HTTP 200 as successful persistence.

After a successful `transfer.execute`, the API re-enters the authenticated graph
with the `financial-summary` intent so the response carries the newly committed
account balance. A successful `credit_card.pay` similarly refreshes the
`credit-card` view. A refresh failure preserves the confirmed write result and
does not invite a duplicate retry.
The Expo renderer supports a strict Basic v0.9.1 subset of TextField, DateTimeInput (date only), Slider and Button. Requests such as “Crea un presupuesto”, “Edita un presupuesto”, “Crea una meta de ahorro” and “Edita mi meta de ahorro” make the model answer with an `action_form`, which `prepare_action` turns into the matching MCP `a2ui_form` call; this preparation does not save anything. No phrase table is involved — the model picks one of the four enumerated names and `normalize_form_name` re-validates it. Editing asks for the existing name and loads its owned data. Save submits a structured `action` in the authenticated HTTP body, using the five A2UI fields and explicit resolved context. The orchestrator supplies trustedScope from the verified token; model-supplied ownership never wins. The LLM cannot call a2ui_action. The MCP returns data.actionResult and the client displays success/failure rather than interpreting HTTP 200 as successful persistence.

Canonical input/action JSON is packaged under `a2ui_actions/`, synchronized from the MCP contract. The agent validates forms with the official A2UI 0.9.1 SDK. Saving requires the new MCP code, action SQL and dedicated write-role configuration; this source change does not deploy them.

Set the same random `MCP_ACTIONS_SECRET` (at least 32 characters) on agent and MCP. The orchestrator signs the event plus verified user ID before forwarding it over the existing Horizon connection. This server-only signature never enters the A2UI context or the model; the MCP rejects unsigned or modified write requests.
