# Architecture

How this repository is organised, what each module is responsible for, and
where to make a given change. Runtime behaviour is described in
[FLOWS.md](FLOWS.md).

This service is the **Agent Orchestrator** of the system described in
`PROJECT_SPEC.MD`. It owns the graph topology, prompt/policy logic, the finite
protocol vocabularies the model selects from, presentation selection, the
trusted Finance v2 builders, and the MCP client integration. It does **not**
own data access, the A2UI catalog definition, action validation, or the renderer
— those belong to the MCP/data repository and the mobile client. It also does
not classify requests: there is no phrase table and no pre-graph router, so
"what does this user need" is a model decision bounded by enumerated values.

## Package layout

```text
src/fluidbank_orchestrator/
  graph.py                    LangGraph topology only; the compiled `graph` and
                              the LangGraph CLI entrypoint (langgraph.json)
  state.py                    GraphState / UserProfile TypedDicts
  observability.py            turn-scoped stage logging and per-turn timeline
  auth.py                     Supabase bearer-token verification

  api/                        the HTTP boundary
    __init__.py               ASGI app, CORS, /health, /api/v1/agent/chat,
                              /api/v1/agent/chat/stream, dispatch order,
                              the NDJSON status allowlist
    actions.py                A2UI action transport and its trust checks
    responses.py              ChatResponse construction and response logging

  agent/                      the LangGraph agent
    model.py                  ToolAwareModel port + ModelTurn
    gemini.py                 the only Gemini-specific code: prompt, adapter,
                              bounded answer schema, profile projection
    status.py                 the eight lifecycle status ids and emit_status
    tool_visibility.py        what a model may see of the tool surface
    observations.py           the turn's ledger of verified MCP results
    tool_loop.py              executing pending calls, recording provenance
    nodes.py                  each workflow step, plus the routing decision

  mcp_client/                 the MCP boundary
    errors.py                 the failure vocabulary
    models.py                 detached values crossing the boundary
    config.py                 endpoint validation, FastMCP client construction
    session.py                one shared connection per turn
    tool_names.py             financial / discovery / scoped tool-name sets
    trusted_scope.py          the identity boundary (security-critical)
    catalog.py                loading and caching the advertised tool list
    execution.py              addressing, scoping and running one tool
    user_context.py           deriving the profile from scoped reads

  schemas/                    strict wire contracts
    a2ui.py                   official A2UI envelope + supported catalogs
    banking_view.py           Finance v2 intent/data models
    a2ui_action.py            parser for the action-over-chat transport
    chat.py                   the HTTP request/response contract

  services/
    a2ui_bridge.py            MCP-produced A2UI: resolve, validate, cache
    financial_presentation/   Finance v2 presentation
      intents.py              the finite intent vocabulary
      verified_rows.py        reading only what MCP actually returned
      views.py                one view payload builder per supported intent
      surface.py              the trusted A2UI message-sequence builder
      builder.py              observations -> one validated presentation

  a2ui_actions/               canonical action/input JSON, synchronized from MCP
    forms.py                  ACTION_FORM_NAMES: the four preparable form names
  a2ui_catalogs/              checked-in Finance v1 catalog
```

Two finite vocabularies bound everything the model is allowed to select:
`schemas/banking_view.py:FINANCIAL_INTENTS` (13 presentation intents, normalized
by `services/financial_presentation/intents.py:normalize_action_intent`) and
`a2ui_actions/forms.py:ACTION_FORM_NAMES` (4 preparable A2UI forms, normalized
by `normalize_form_name`). Both are re-validated after the model answers, so an
invented value becomes "no branch" rather than an unchecked call.

## Dependency direction

```text
api  ->  graph  ->  agent  ->  services.financial_presentation
 |         |          |                 |
 |         +----------+------------> mcp_client  ->  services.a2ui_bridge
 |                                       |                    |
 +-> auth                                +------------> schemas, state, observability
```

Rules that hold today and should keep holding:

* `schemas`, `state` and `observability` are leaves. They import nothing from
  the rest of the package.
* `mcp_client` never imports `agent`, `api` or `graph`.
* `agent` never imports `api`. The one reverse-direction dependency is
  `api/__init__.py` importing `agent.status`'s id set to allowlist the stream.
* `graph.py` imports only `agent`, `mcp_client` and `state`; it contains no
  behaviour of its own.
* `services.financial_presentation` depends on `mcp_client` for one thing only:
  the shared `FINANCIAL_DOMAIN_TOOL_NAMES` vocabulary.

## Where to change a specific concern

| I want to change… | Edit |
| --- | --- |
| the order of graph steps, or add/remove a node | `graph.py` |
| what a node does (context, agent policy, presentation) | `agent/nodes.py` |
| the prompt, model name, or Gemini request shape | `agent/gemini.py` |
| swapping Gemini for another model | implement `agent/model.py:ToolAwareModel`, pass it to `build_graph(model=…)` |
| the per-turn tool-call cap | `agent/gemini.py:MAX_CALLS_PER_TURN` |
| the number of model↔tools iterations | `agent/nodes.py:MAX_TOOL_TURNS` |
| what the model sees of the user profile | `agent/gemini.py:_MODEL_PROFILE_KEYS` |
| which A2UI forms the model may ask MCP to prepare | `a2ui_actions/forms.py:ACTION_FORM_NAMES` |
| which intents are legal protocol values | `schemas/banking_view.py:FINANCIAL_INTENTS` |
| whether observations support a different view than requested | `services/financial_presentation/intents.py:select_presentation_intent` |
| which lifecycle phases exist, or where one is reported | `agent/status.py`, then the `emit_status` call in `agent/nodes.py` |
| the words a user reads for a phase | **not here** — `HackMTY2026_Mobile/src/features/assistant/agent-status.ts` |
| what a model may see of a tool schema | `agent/tool_visibility.py` |
| the tool-loop guard rails, provenance recording | `agent/tool_loop.py` |
| how a Finance v2 view payload is built | `services/financial_presentation/views.py` |
| the A2UI component tree of the financial surface | `services/financial_presentation/surface.py` |
| which intents have a real view vs an empty one | `services/financial_presentation/builder.py` |
| identity scoping, ownership filters, action signing | `mcp_client/trusted_scope.py` |
| which tools are scoped / financial / discovery | `mcp_client/tool_names.py` |
| MCP URL, auth mode, credentials | `mcp_client/config.py` |
| the tool-schema cache and its TTL | `mcp_client/catalog.py` |
| how a tool is addressed on a proxied deployment | `mcp_client/execution.py:_wire_call` |
| the derived user profile (balances, overdraft risk) | `mcp_client/user_context.py` |
| the HTTP request/response shape | `schemas/chat.py` |
| CORS, endpoints, dispatch order | `api/__init__.py` |
| how an action result is trusted | `api/actions.py` |
| what may travel on the status stream | `api/__init__.py:_status_line` |
| what the client envelope looks like and what is logged | `api/responses.py` |
| accepted A2UI components/catalogs | `schemas/a2ui.py` |
| the Finance v2 data contract | `schemas/banking_view.py` |
| MCP-produced A2UI resolution/caching | `services/a2ui_bridge.py` |
| stage names, turn timeline, payload previews | `observability.py` |

## Architectural and security boundaries

**Identity.** The authenticated `current_user_id` is derived exactly once, in
`api/__init__.py:_handle_chat`, from the Supabase bearer token via
`auth.verify_supabase_access_token`. It is the only trusted identity source. A
body `user_id` that disagrees is a 403; action context and model arguments are
never consulted. `mcp_client/trusted_scope.py:enforce_trusted_user_scope`
overwrites every server-owned identity field immediately before a scoped call,
including inside a `call_tool` envelope, so discovery cannot smuggle an unscoped
call past it. `require_current_user_id` fails closed on anything that is not a
`UUID`, and it runs as the graph's first node.

**Model containment.** The model may do exactly two things: choose tool calls
from what the server advertises as model-visible, and return the bounded
`_Intent` answer (a message, an optional month count, an optional presentation
intent, an optional action form — the last two declared as enums over the finite
vocabularies). It never sees or emits A2UI JSON, component names, IDs, styles or
protocol values. Both protocol fields are normalized again — by
`normalize_action_intent` and `normalize_form_name` — before either can reach a
builder or MCP.

What it is *handed* is bounded too. `agent/gemini.py:model_profile` projects
`UserProfile` down to `literacy_level`, so `available_balance`,
`owned_balances`, `overdraft_risk` and `recurring_expenses` never enter a
prompt: the trusted builder renders every figure from MCP observations, and the
client owns appearance, so sending them bought no behaviour and put the user's
money in every request of every turn.

**Tool discovery vs. the security boundary.** These are separate on purpose.
Discovery (`DISCOVERY_TOOL_NAMES`, `search_tools -> call_tool`) decides *how a
model reaches* a capability, and the model-facing set comes straight from
`tools/list` with no local allowlist. `SCOPED_TOOL_NAMES` decides *whose data*
a call may touch. Adding a local catalog allowlist would defeat the server's
discovery; weakening `SCOPED_TOOL_NAMES` would trust model-supplied identity.

**Model visibility.** `_meta.ui.visibility` is read in
`mcp_client/catalog.py:_declared_model_visible` and honoured in
`agent/tool_visibility.py:model_tool_definitions`. The MCP Apps spec puts this
filtering on the host; for the model-facing tool set, this orchestrator is the
host. Tools the server advertises purely so the orchestrator can address them by
name (`get_user_context`, `a2ui_action`, `a2ui_form`) stay out of the prompt.

**Provenance.** A domain answer is only published from data that arrived through
a successful MCP call. `agent/observations.py` is the ledger,
`services/financial_presentation/verified_rows.py` is the only reader, and
`builder.py` turns a failed domain read into an explicit empty view rather than
an estimate.

**Determinism.** Determinism here means identity, validation, protocol and
approval — **not** request classification. Deciding what a user needs is the
model's job and is not deterministic; everything that could let that decision
cause harm is:

* *identity* is derived once from the token and re-imposed on every scoped call;
* *validation* is exhaustive and offline — every intent, form name, view payload
  and message sequence is checked against a checked-in contract, twice for a
  surface;
* *protocol values* come from closed vocabularies, so the set of reachable
  outcomes is fixed at build time even though the choice among them is not;
* *approval gates* mean no state change happens without an explicit user event
  that MCP re-validates against the template that declared the button.

Given the same tool observations, the same intent produces the same surface
byte for byte: `services/financial_presentation/` is pure. Nothing upstream of
the model classifies anything.

**Lifecycle reporting.** `agent/status.py` publishes eight coarse phase ids over
LangGraph's custom stream channel, and `api/__init__.py:_status_line` allowlists
the payload at the HTTP boundary, so a node cannot widen a progress channel into
a reasoning channel: only the discriminator and a known id are ever forwarded.
No prompt text, reasoning, tool name, argument or row can travel on it.
`emit_status` is a no-op when nobody is streaming, so the plain route and the
tests run the identical graph. The copy a user reads is client-owned and lives
only in the mobile repo.

**A2UI ownership.** Two paths, never mixed. Finance v2 surfaces are constructed
by the trusted Python builder in
`services/financial_presentation/surface.py` and validated against the shared
contract. MCP-produced A2UI travels through `services/a2ui_bridge.py`, which
branches on `_meta.ui` rather than on tool names. Both are validated with the
official SDK before anything is forwarded.

**Actions.** State-changing A2UI actions are bounded backend operations, never
LLM decisions. Every action is forwarded to MCP's allowlist with the verified
UUID in a separate `trustedScope` field and an HMAC `actionProof` derived
server-side. Only a `request_financial_view` result that MCP itself normalized,
and that echoes this authenticated user, re-enters the graph.

## Testing seams

`build_graph(model=…, tool_loader=…, tool_executor=…)` is the main injection
point: the tests drive the whole workflow with a scripted `ToolAwareModel` and a
fake executor. The HTTP boundary keeps `verify_supabase_access_token`,
`execute_remote_tool` and `graph` as module-level names in `api/__init__.py` so
they can be substituted without a running MCP server or Supabase project. Within
`mcp_client`, patch the module that owns a behaviour (`mcp_client.catalog`,
`mcp_client.session`, `mcp_client.user_context`), not the package facade.
