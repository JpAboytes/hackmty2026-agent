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
- Interprets user intent, obtains the minimum required data, selects a template, and produces an A2UI payload.
- Must not directly mutate client state or generate executable client code.
- Must enforce confirmation boundaries for actions that could affect financial decisions or user data.

### 5.3 MCP Tool Server

- FastMCP in Python.
- Provides narrowly scoped, typed tools for retrieving demo financial data and computing derived scores.
- Tools must validate inputs, return structured errors, and expose only the data required by the current request.
- The MCP server is the only component that directly accesses Supabase data for the initial architecture.

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

Owns LangGraph state, intent routing, prompt and policy logic, template selection, payload generation, and orchestrator tests.

It consumes the versioned shared contract definition and communicates with the MCP server through its public tool interface.

### 6.3 MCP and Data Repository

Owns the FastMCP server, tool definitions, Supabase migrations, seed data, score computation, and data/tool tests.

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

Every response from the orchestrator to the mobile client must validate against the versioned A2UI schema before it is rendered.

Example payload:

```json
{
  "version": "a2ui/v1",
  "template_id": "projection",
  "applied_tags": ["high-contrast", "large-targets"],
  "meta": {
    "narrative": "Your current budget supports a six-month purchase without affecting your rent.",
    "accessibility": {
      "font_scale": "lg",
      "contrast": "high",
      "hit_target": "large"
    }
  },
  "surface": {
    "layout": "vertical_stack",
    "components": [
      {
        "id": "alert-header",
        "type": "Banner",
        "tags": ["high-contrast"],
        "props": {
          "variant": "info",
          "title": "Purchase installment simulation"
        }
      },
      {
        "id": "installment-term",
        "type": "InteractiveSlider",
        "tags": ["large-targets"],
        "props": {
          "label": "Number of months",
          "min": 3,
          "max": 18,
          "step": 3,
          "default_value": 6
        }
      },
      {
        "id": "confirm-simulation",
        "type": "Button",
        "tags": ["large-targets"],
        "props": {
          "label": "Confirm simulation",
          "action": {
            "type": "A2UI_DISPATCH",
            "intent": "CONFIRM_SIMULATION",
            "payload": {
              "months": 6
            }
          }
        }
      }
    ]
  }
}
```

Contract requirements:

- Every component has a stable, unique `id` within the surface.
- `type`, layout values, tags, accessibility values, and action intents are enumerated by the shared contract.
- Unknown component types or invalid props must produce a visible, recoverable error state; they must never be rendered dynamically.
- The client must treat payload content as data, not executable code.
- Actions must be validated again by the receiving backend before any state-changing operation.
- Contract changes require a version update, migration notes, and fixture coverage.


