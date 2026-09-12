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
- Interprets user intent and obtains the minimum required data by calling the MCP server's tools.
- Drafts conversational text grounded in that data and extracts structured parameters (e.g. a requested number of months) the MCP server needs to build the right surface.
- Never invents, translates, or asks the LLM to edit A2UI content. The MCP server is the only component that produces A2UI. The orchestrator validates MCP-produced messages, resolves the MCP-selected static resource, and packages the original validated message objects for the mobile transport.
- Must not directly mutate client state or generate executable client code.
- Must enforce confirmation boundaries for actions that could affect financial decisions or user data: user-triggered actions are dispatched deterministically to the MCP server's action tool, never decided or rewritten by the LLM.

### 5.3 MCP Tool Server

- FastMCP in Python. Runnable locally over stdio and deployable remotely (e.g. Horizon) over Streamable HTTP.
- Provides narrowly scoped, typed tools for retrieving demo financial data and computing derived scores.
- Owns the full A2UI protocol implementation: templates, resources (served via `resources/read`), `updateDataModel`, validation, `_meta.ui` metadata, user-triggered actions, and A2UI error responses. A2UI content is returned as an `EmbeddedResource` (MIME type `application/a2ui+json`) alongside any plain text or structured content in the same tool result.
- Tools must validate inputs, return structured errors, and expose only the data required by the current request.
- The MCP server is the only component that directly accesses Supabase data and the only component that constructs or validates A2UI content.

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

Owns LangGraph state, intent routing, prompt and policy logic, the MCP client integration, and orchestrator tests. It does not own template selection or A2UI payload generation - that belongs to the MCP repository.

It consumes the official A2UI schema to validate the envelope it relays; it must never invent, translate, or semantically edit A2UI content. It communicates with the MCP server through its public tool and resource interfaces, whether a local stdio process or its deployed HTTP endpoint. Because Mobile is not an MCP client, the orchestrator combines the MCP-produced static resource messages and embedded dynamic messages into one ordered, self-contained response.

### 6.3 MCP and Data Repository

Owns the FastMCP server, tool definitions, the full A2UI protocol implementation (templates, resources, `updateDataModel`, actions, validation, error responses), Supabase migrations, seed data, score computation, and data/tool tests.

It must not contain mobile UI code or assume a local checkout of the mobile repository.

### 6.4 Shared Contracts Repository

Owns the versioned A2UI contract, JSON Schema, TypeScript types, validation fixtures, compatibility rules, and generated artifacts required by consumers.

The contract repository is the only source of truth for the wire format. Consumer repositories may add local adapters, but they must not silently fork the schema.

### 6.5 Future Web Repository

A future Next.js dashboard for small-business management may be created as a separate repository. It is out of scope for the initial release and must not add dependencies to the mobile, orchestrator, MCP, or shared-contract repositories.

## 7. Accessibility and Personalization

The backend computes or retrieves these user-context values:

- `literacy_level`: `low`, `medium`, or `high`.
- `font_scale`: `sm`, `md`, `lg`, or `xl`.
- `contrast`: `normal` or `high`.
- `color_vision_mode`: `none` or `deuteranopia`.
- `hit_target`: `normal` or `large`.

These values influence language, information density, component properties, and layout selection. Accessibility settings are not cosmetic metadata: every renderer component must apply them consistently.

Functional tags may include:

- `alert-critical`: Critical financial warning.
- `high-contrast`: High-contrast borders and text.
- `large-targets`: Controls with a minimum height of 64 pixels.
- `action-first`: Prioritize primary actions over secondary details.
- `simple-view`: Show only the most important metrics.

Tag names are part of the shared contract. New tags must be documented and versioned there.


Template selection must be deterministic for the same normalized request, user context, and data snapshot. Templates may be extended only through the shared contract and corresponding renderer support.

## 8. A2UI Contract

The system implements Google's A2UI protocol as-is: a hard requirement, not a bespoke JSON format inspired by it. An earlier draft of this document specified a flat, single-shot payload (`a2ui/v1`, `meta`, `surface`, `Banner`, `MetricCard`, and `InteractiveSlider`); that draft is superseded and must not be implemented. The MCP repository is the source of truth for the produced wire format, under `src/supabase_mcp/a2ui_support/`.

### 8.1 Ownership

- **MCP server**: builds and validates every A2UI artifact - templates, resources, surfaces, the component catalog, `updateDataModel` patches, actions, and A2UI error responses. This is the only component allowed to construct or edit A2UI content.
- **Agent orchestrator**: calls MCP domain tools, reads `_meta.ui`, resolves the selected MCP resource, validates static and dynamic messages, and returns detached copies in source order. The LLM never sees, generates, or modifies A2UI JSON.
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

The supported catalog is exactly the v0.9.1 Basic Catalog. The currently implemented Mobile subset is `Text`, `Button`, `Card`, and `Column`. Charts remain local-only; network-delivered charts require a future, explicitly versioned custom catalog shared by MCP, agent, and Mobile. Future financial domain tools that publish the same metadata/resource/embedded-update contract require no component-specific changes to this bridge.

### 8.5 Actions

User-triggered actions (button presses, confirmations) call the MCP server's `a2ui_action` tool deterministically from the mobile/agent layer. During the temporary chat-transport compatibility period, the agent strictly parses Mobile's serialized action JSON, requires exactly `name`, `surfaceId`, `sourceComponentId`, `timestamp`, and `context`, and forwards those values unchanged. Actions never enter the LLM. The MCP action registry enforces action/component allowlists, and any A2UI-enabled result returns through the same generic bridge.

### 8.6 Accessibility boundary

Personal accessibility preferences (font scale, high contrast, dark mode, reduced motion, screen reader state, large touch targets) are a Mobile/renderer concern and are not sent to the MCP server. The MCP server emits semantic components (e.g. `variant: "h1"`, not `fontSize: 28`) and lets the renderer map them to accessible native widgets. The MCP server only needs to know about an accessibility preference when it changes *what content or structure* to send, not merely its style - for example `prefersDataTable`, `requiresTextAlternative`, or `avoidTimeLimitedInteractions` passed via request `_meta`, never inside the LLM prompt.

### 8.7 Current integration proof

`database_overview` is a development/integration proof for the bridge. It selects `a2ui://database/overview`, whose static template declares `Text`, `Button`, `Card`, and `Column`, and embeds its dynamic `updateDataModel`. It demonstrates protocol integration over allowlisted database metadata; it is not a consumer-facing banking surface.

### 8.8 General requirements

- Every component has a stable, unique id within its surface.
- Component types, catalog entries, and action names are enumerated by the MCP server's catalog/template implementation, not invented by the LLM.
- Unknown component types or invalid props must produce a visible, recoverable error state on the client; they must never be rendered dynamically.
- The client must treat payload content as data, not executable code.
- Actions must be validated again by the MCP server before any state-changing operation.
- Contract changes require a version update, migration notes, and fixture coverage, tracked in `hackmty2026-mcp`.


