# Architecture

How this repository is organised, what each module is responsible for, and
where to make a given change. Runtime behaviour is described in
[FLOWS.md](FLOWS.md).

This service is the **Agent Orchestrator** of the system described in
`PROJECT_SPEC.MD`. It owns intent routing, prompt/policy logic, presentation
selection, the trusted Finance v2 builders, and the MCP client integration. It
does **not** own data access, the A2UI catalog definition, action validation, or
the renderer — those belong to the MCP/data repository and the mobile client.

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
                              authenticated dispatch order
    actions.py                A2UI action transport and its trust checks
    query_routing.py          deterministic pre-graph query classification
    responses.py              ChatResponse construction and response logging

  agent/                      the LangGraph agent
    model.py                  ToolAwareModel port + ModelTurn
    gemini.py                 the only Gemini-specific code: prompt, adapter,
                              bounded answer schema
    tool_visibility.py        what a model may see of the tool surface
    observations.py           the turn's ledger of verified MCP results
    retrieval.py              deterministic financial retrieval planning
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

  a2ui_actions/               canonical action/input JSON + form detection
  a2ui_catalogs/              checked-in Finance v1 catalog
```

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
* `agent` never imports `api`.
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
| which tool a financial intent reads from | `agent/retrieval.py` |
| which phrases map to which intent | `services/financial_presentation/intents.py` |
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
| which plain queries bypass the graph | `api/query_routing.py` |
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
`_Intent` answer (a message, an optional month count, an optional intent from
the finite vocabulary). It never sees or emits A2UI JSON, component names, IDs,
styles or protocol values. Any intent it returns is normalized again by
`normalize_action_intent` before it can reach a builder.

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

**Determinism.** A classified financial request never reaches the model:
`agent/nodes.py` routes it to `agent/retrieval.py`, which plans at most one tool
call from the intent and the normalized query text. Same request, same intent,
same observations, same plan.

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
