# Project Specification: Generative UI Personal Banking

## 1. Purpose and Vision

This project is an accessibility-first mobile banking experience for a Mexican banking hackathon. Instead of presenting a fixed set of banking screens, the application renders a dynamic user interface generated from the user's request, financial context, and accessibility profile.

Users can interact through text or voice. An agent orchestrator interprets the request, retrieves the required financial data through controlled tools, selects an appropriate UI template, and streams a structured UI payload to the mobile client. The client renders that payload through a strict component registry.

The product has no final name yet. Do not introduce or assume a product name in code, documentation, package names, repository names, UI copy, or examples.

## 2. Goals

- Provide a reliable mobile banking demo driven by text and voice requests.
- Adapt explanations and visual complexity to the user's financial literacy and accessibility profile.
- Render agent-generated interfaces from validated, typed UI contracts rather than hardcoded screen layouts.
- Make critical financial actions clear, safe, and easy to complete.
- Keep the system modular so each service can be developed, tested, and deployed from its own repository.
- Preserve a local demo mode when remote agent or data services are unavailable.

## 3. Non-Goals for the Initial Release

- Production banking integrations or real money movement.
- A production-grade web dashboard.
- Replacing bank authentication, authorization, fraud detection, or regulatory controls.
- Supporting arbitrary UI components supplied by an agent.
- Treating generated UI as permission to bypass backend validation or user confirmation.

## 4. Initial User Scenarios

The initial release should support these representative scenarios:

1. **Financial warning:** Explain an urgent financial issue and present a concise set of safe next actions.
2. **Subscription or expense review:** Summarize recurring expenses and provide actions to manage them.
3. **Purchase simulation:** Let the user explore installment terms and see the projected monthly impact before confirming a simulation.
4. **Accessible reading mode:** Present the same intent using larger targets, stronger contrast, simpler language, and reduced visual complexity when the profile requires it.

The implementation may use seeded or mocked financial data. It must clearly distinguish demo data from real financial data.

## 5. System Architecture

The system consists of independently maintained repositories and deployable components:

```text
[Text or Voice Input]
          |
          v
[Mobile Client] <---- HTTP/WebSocket ----> [Agent Orchestrator]
      ^                                      |
      |                                      v
      |                             [MCP Tool Server]
      |                                      |
      |                                      v
      |                              [Supabase Data]
      |
      +---------- validated A2UI payload ----------
```

### 5.1 Mobile Client

- Expo and React Native with TypeScript.
- TailwindCSS for styling (NEXTS JS).
- Voice capture and transcription may use Expo-compatible audio tooling and a configured transcription provider.
- A custom `A2UIRenderer` renders only component types registered in `ComponentRegistry`.
- The client validates incoming payloads before rendering them.
- The client exposes generated-component actions as `A2UI_DISPATCH` events.

### 5.2 Agent Orchestrator

- LangGraph implemented in the backend language selected for the orchestration repository.
- There is **no financial intent routing**: no phrase table maps a request to a tool or presentation and there is no deterministic retrieval planner. One graph handles every query. A deterministic, fail-closed policy node runs before the model and before user-context/tool loading; it only accepts banking-scoped requests and rejects prompt injection, requests for executable code or scripts, historical narration, and all other out-of-domain content. It never selects a business intent. For accepted requests, the model decides whether data is needed, which capabilities provide it, how many, and whether the turn ends in a presentation, an A2UI action form, or bounded plain prose.
- Discovers capabilities through the MCP server's `search_tools` and calls one or more of them through `call_tool`; how many is the model's decision, bounded by a per-turn call cap and an iteration limit rather than by a "use one tool" policy.
- Interprets returned data, then selects one of the finite financial presentation semantics in the shared contract.
- May instead select one of the four enumerated A2UI action forms and ask MCP to *prepare* it. Preparing a form performs no write; only a subsequent explicit user event does.
- Reports coarse lifecycle phases on an optional streaming route as an enumerated identifier and nothing else — no prompt text, reasoning, tool names, arguments or rows. The user-facing wording of those phases belongs to the mobile client.
- Constructs Finance v2 `BankingView` payloads through trusted typed builders. The LLM never emits component names, actions, IDs, styles, properties, or raw A2UI JSON.
- Continues to validate and relay MCP-produced A2UI for legacy/domain tools through the generic bridge.
- Must not directly mutate client state or generate executable client code.
- Must not generate executable code of any kind, disclose or follow requests to alter its prompts/policies, provide historical narratives, or answer outside the banking domain. These restrictions must be enforced in deterministic pre-model and final-output guards; prompt wording alone is not a security boundary. A rejected query must not load financial context, expose tool schemas to the model, call MCP, or invoke the LLM.
- Verifies the Supabase bearer token at the HTTP boundary, derives `current_user_id` from its authenticated subject, and preserves that identity through graph and MCP scope.
- Must enforce confirmation boundaries for actions that could affect financial decisions or user data. Read-only `request_financial_view` actions re-enter the same intent/data/presentation graph; state-changing actions remain bounded backend operations and never become LLM decisions.

### 5.3 MCP Tool Server

- FastMCP in Python. Runnable locally over stdio and deployable remotely (e.g. Horizon) over Streamable HTTP.
- Provides narrowly scoped, typed tools for retrieving demo financial data and computing derived scores.
- Defines and validates the bounded data/action surface and the A2UI catalog/schema contract. It may return MCP-owned A2UI resources for its domain tools, while the agent uses the same versioned contract for post-data Finance v2 presentation construction.
- Tools must validate inputs, return structured errors, and expose only the data required by the current request.
- The MCP server is the only component that directly accesses Supabase financial data. It defines/validates allowed contracts and MCP-owned A2UI; the agent's trusted typed builder constructs and validates Finance v2 after interpreting returned data.

### 5.4 Supabase

- PostgreSQL stores seeded demo users, transactions, subscriptions, and accessibility preferences.
- Realtime may be used for demo updates where it provides clear value.
- Database migrations and seed data are owned by the data repository.
- No production credentials or real customer data may be committed to any repository.

## 6. Repository Boundaries

Repositories are separate. Do not create a monorepo-style directory tree that assumes the repositories live together.

### 6.1 Mobile Repository

Owns the Expo application, native configuration, screen shell, voice/text input, A2UI renderer, component registry, local demo fallback, and mobile tests.

It consumes the versioned shared contract package. It must not import source files directly from the orchestrator or MCP repositories.

### 6.2 Agent Orchestrator Repository

Owns LangGraph state and topology, prompt and policy logic, the finite protocol vocabularies the model selects from, financial presentation selection, trusted Finance v2 builders, the lifecycle status vocabulary, the MCP client integration, and orchestrator tests.

It consumes the official and shared Finance schemas to validate every envelope it relays or constructs. The LLM may select a known semantic intent but cannot invent protocol values. It communicates with the MCP server through public tool/resource interfaces and returns one ordered, self-contained response because Mobile is not an MCP client.

### 6.3 MCP and Data Repository

Owns the FastMCP server, bounded tool definitions, data access, action validation, shared A2UI catalog/schema definitions, MCP-owned templates/resources, Supabase migrations, seed data, score computation, and data/tool tests.

It must not contain mobile UI code or assume a local checkout of the mobile repository.

### 6.4 Shared Contracts Repository

Owns the versioned A2UI contract, JSON Schema, TypeScript types, validation fixtures, compatibility rules, and generated artifacts required by consumers.

The contract repository is the only source of truth for the wire format. Consumer repositories may add local adapters, but they must not silently fork the schema.

### 6.5 Future Web Repository

A future Next.js dashboard for small-business management may be created as a separate repository. It is out of scope for the initial release and must not add dependencies to the mobile, orchestrator, MCP, or shared-contract repositories.

## 7. Accessibility and Personalization

Accessibility is split across two independent mechanisms. This section records
which is which, because the difference is load-bearing and an earlier draft of
this document described the two as one thing.

### 7.1 Appearance is client-owned and device-local

Everything the user actually sees adapt is decided on the device. `Configuración
→ Accesibilidad` (also reachable from the sign-in screen before there is an
account) edits ten Zod-validated preferences — `textScale`, `lineSpacing`,
`letterSpacing`, `boldText`, `appearance`, `highContrast`, `colorPalette`,
`reduceMotion`, `largeTargets`, `chartDataTable` — persisted in AsyncStorage per
profile per device under `accessibility:v1:*`. They are **never** written to
Supabase user metadata and **never** sent to the agent or the MCP server. See
`HackMTY2026_Mobile/src/features/accessibility/README.md` for the full table and
the rules new components must follow. There is no onboarding questionnaire; the
settings panel is the only way these values are set.

Because the renderer's A2UI adapters are the same components the rest of the app
uses, a generated surface inherits those preferences automatically — which is the
real meaning of "every renderer component applies them consistently".

### 7.2 The backend profile is advisory and mostly unused

The MCP server stores five values in `accessibility_preferences`, and
`mcp_client/user_context.py` reads all five into `UserProfile`:

- `literacy_level`: `low`, `medium`, or `high`.
- `font_scale`: `sm`, `md`, `lg`, or `xl`.
- `contrast`: `normal` or `high`.
- `color_vision_mode`: `none` or `deuteranopia`.
- `hit_target`: `normal` or `large`.

Only `literacy_level` has an effect today: `agent/gemini.py:model_profile`
projects the profile down to that one key before it reaches the prompt, where it
shapes the wording of the model's prose. The other four describe appearance,
which the client owns from §7.1, so the orchestrator deliberately does not act on
them and does not put them in a prompt. A missing preference column falls back to
its default rather than invalidating the whole user context.

### 7.3 What is specified but not implemented

- **Functional tags.** `alert-critical`, `high-contrast`, `large-targets`,
  `action-first` and `simple-view` have **zero** implementations in any of the
  three repositories: nothing emits them, nothing validates them, nothing
  renders differently because of them. They remain a naming reservation. Before
  any of them is used, the wire mechanism has to be specified — which message
  carries them and which validator accepts them — and versioned in the shared
  contract.
- **Backend-driven information density and layout selection.** Which view an
  intent produces is decided by the intent and the verified data, not by the
  accessibility profile. There is no low-literacy variant of a surface.

### 7.4 What A2UI does carry

Accessible names are on the wire and are not optional. Every component of the
trusted Finance v2 surface declares `accessibility.label` bound to a data-model
key (`/viewLabel`, `/actionLabel`), and MCP's own
`templates/financial_view.json` mirrors it, so a generated surface announces
itself to a screen reader as well as a hand-written screen does. This is the
documented A2UI accessibility attribute set; no `meta.accessibility` block and no
arbitrary labels were added to the protocol.

Template selection must be deterministic for the same normalized request, user
context, and data snapshot. Templates may be extended only through the shared
contract and corresponding renderer support.

## 8. A2UI Contract

The system implements Google's A2UI protocol as-is: a hard requirement, not a bespoke JSON format inspired by it. An earlier draft of this document specified a flat, single-shot payload (`a2ui/v1`, `meta`, `surface`, `Banner`, `MetricCard`, and `InteractiveSlider`); that draft is superseded and must not be implemented. The MCP repository is the source of truth for the produced wire format, under `src/supabase_mcp/a2ui_support/`.

### 8.1 Ownership

- **MCP server**: defines and validates allowed catalogs, data access, actions, and MCP-owned templates/resources.
- **Agent orchestrator**: calls MCP data tools, interprets results, chooses a finite presentation intent, and uses trusted typed builders for Finance v2. It also resolves and validates MCP-owned A2UI. The LLM never sees, generates, or modifies A2UI JSON.
- **Mobile client**: implements the A2UI renderer, the component catalog, per-surface state, data bindings, and dispatches user actions. It is not an MCP client and never dereferences an `a2ui://` URI.

### 8.2 Transport shape

A2UI content travels inside an MCP tool result as an `EmbeddedResource` with MIME type `application/a2ui+json`, alongside the tool's ordinary text and structured content:

```text
ToolResult
├── TextContent            → may reach the LLM
├── structuredContent      → may reach the LLM if the flow needs it
├── EmbeddedResource (A2UI, application/a2ui+json)  → relayed to Mobile untouched
└── _meta.ui                                        → template/surface metadata
```

Templates are also published directly through MCP's `resources/read`. A tool selects a complete surface through `_meta.ui.resourceUri`; it does not select an individual component. Components remain declarations inside the selected static surface template.

### 8.3 Envelope forwarded by the agent

The orchestrator's HTTP response to the mobile client separates conversational content from A2UI content so the LLM's output path and the A2UI path never mix:

```json
{
  "message": "Encontré 5 resultados.",
  "data": { "count": 5 },
  "a2ui": {
    "resource_uri": "a2ui://database/overview",
    "messages": [
      {
        "version": "v0.9.1",
        "createSurface": {}
      },
      {
        "version": "v0.9.1",
        "updateComponents": {}
      },
      {
        "version": "v0.9.1",
        "updateDataModel": {
          "surfaceId": "database-overview",
          "path": "/",
          "value": {}
        }
      }
    ]
  }
}
```

The abbreviated bodies above represent complete official envelopes. `messages` always contains JSON objects, never encoded JSON strings. For a normal new surface, the order is `createSurface` → `updateComponents` → `updateDataModel`. `a2ui` is `null` whenever the MCP server has nothing to render or the optional presentation cannot be validated; the successful MCP text fallback and structured `data` remain available.

### 8.4 Agent bridge and cache

The generic bridge uses public FastMCP result fields. It reads runtime `CallToolResult.meta["ui"]`, requires an `a2ui://` resource URI and `application/a2ui+json`, reads that URI through the connected MCP client's `resources/read`, and extracts only matching `EmbeddedResource` content from the tool result. It validates the template, dynamic updates, surface consistency, exact `v0.9.1` protocol, and complete sequence with the official SDK.

Validated static templates are cached by MCP server identity plus resource URI in a bounded process-local LRU cache. The cache never stores `updateDataModel` data, returns detached copies, coalesces concurrent reads for the same uncached key, and does not retain failed reads. Static messages are included on every response because Mobile treats each response as a self-contained replacement.

Validation allowlists the v0.9.1 Basic Catalog, Finance v1, and `https://fluidbank.app/a2ui/catalogs/finance/v2`. Basic remains `Text`, `Button`, `Card`, and `Column`; Finance v1 adds strict `Chart`; Finance v2 adds only strict `BankingView`. Unknown components, arbitrary properties, inline catalogs, and all other catalog IDs fail before forwarding.

The agent carries a checked-in copy of Finance v1 and strict Python models matching the shared Finance v2 `BankingView` contract so validation is offline. Parity tests compare Finance v1 with MCP and exercise accepted/rejected Finance v2 payloads. The generic bridge still branches on `_meta.ui`, not tool names, while the Finance v2 builder produces a complete validated message sequence after data interpretation. Cache entries never contain banking rows.

The one wire component binds `{kind: "area", accessibleSummary?, props}` or `{kind: "heatmap", accessibleSummary?, props}`. The agent enforces stable unique area series IDs, 240 area points, 500 heatmap cells, finite bounded values, actual calendar dates, unique labels/dates, and forbidden unknown properties. Future catalog components require a versioned SDK-valid schema, matching agent/client validators, a trusted Expo adapter, and cross-repository fixtures; they do not require tool-name special cases in this bridge.

### 8.5 Actions

The API accepts a structured action envelope and temporarily preserves the legacy action-over-chat parser. It requires exactly `name`, `surfaceId`, `sourceComponentId`, `timestamp`, and `context`. The API forwards every action to MCP's allowlist with trusted application identity carried separately as `trustedScope`. A successfully normalized `request_financial_view` re-enters the authenticated graph with the same `current_user_id`; rejected actions do not reach the graph. Actions never enter the LLM.

### 8.6 Accessibility boundary

Personal accessibility preferences (font scale, high contrast, dark mode, reduced motion, screen reader state, large touch targets) are a Mobile/renderer concern and are not sent to the MCP server. The MCP server emits semantic components (e.g. `variant: "h1"`, not `fontSize: 28`) and lets the renderer map them to accessible native widgets. The MCP server would only need to know about an accessibility preference if it changed *what content or structure* to send, not merely its style. No such preference exists today: `prefersDataTable`, `requiresTextAlternative` and `avoidTimeLimitedInteractions` are named here as the shape a future request-`_meta` field should take, and none of them is implemented or sent. If one is added it travels in request `_meta`, never inside the LLM prompt.

### 8.7 Current integration proof

`database_overview` is a development/integration proof for the bridge. It selects `a2ui://database/overview`, whose static template declares `Text`, `Button`, `Card`, and `Column`, and embeds its dynamic `updateDataModel`. It demonstrates protocol integration over allowlisted database metadata; it is not a consumer-facing banking surface.

### 8.8 General requirements

- Every component has a stable, unique id within its surface.
- Component types, catalog entries, and action names are enumerated by the MCP server's catalog/template implementation, not invented by the LLM.
- Unknown component types or invalid props must produce a visible, recoverable error state on the client; they must never be rendered dynamically.
- The client must treat payload content as data, not executable code.
- Actions must be validated again by the MCP server before any state-changing operation.
- Contract changes require a version update, migration notes, and fixture coverage, tracked in `hackmty2026-mcp`.


