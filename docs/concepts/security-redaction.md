---
description: >-
  APAB's security model: local-first defaults, workspace-only filesystem
  access, three redaction levels, and audit-logged LLM egress.
---

# Security and redaction

Antenna designs are often sensitive. APAB's defaults assume the
conservative case: nothing leaves your machine unless you opt in, and
what gets recorded is under your control.

## Local-first defaults

The default provider is Ollama on `localhost`: prompts, tool results,
and design parameters stay local. Remote providers (OpenAI, Anthropic,
Gemini, OpenAI-compatible) are separate extras you install and
configure explicitly; nothing falls back to a hosted model silently.

## Workspace-only filesystem access

Tools that write files are confined to the project workspace. Path
arguments are validated against traversal (`..` components are
rejected, resolved paths must stay inside the workspace root), so a
misbehaving model cannot direct a tool to write outside the project.

## Redaction modes

`llm.redaction_mode` controls what the audit log and (by inheritance)
trace attributes record:

| Mode | audit.json arguments | Result summaries | Span attributes |
|---|---|---|---|
| `none` (local default) | full values | truncated text | argument JSON + summaries |
| `metadata_only` | key names only | truncated text | key names + text lengths |
| `strict` | `[REDACTED]` | `[REDACTED]` | content hashes only |

Observability can diverge from the audit log via
`observability.capture_mode`: for example, full audit detail locally
while exported spans stay at `strict`. A sha256-derived argument hash
is always recorded, so identical calls remain correlatable even when
values are withheld.

## Egress logging

Every LLM response is logged (at the configured redaction level) before
the orchestrator acts on it, and every tool call lands in `audit.json`.
With tracing enabled, the OTLP exporter is explicit configuration, an
endpoint you set rather than a default, and span content obeys the
capture mode above.

## Practical guidance

- Working on sensitive designs with a remote provider: set
  `redaction_mode: strict` and keep `capture_mode` unset so spans
  inherit it.
- Sharing run bundles: a bundle generated under `strict` contains tool
  names, hashes, durations, and statuses. That is enough to debug the
  workflow, and it exposes no geometry values.
- Export control: pattern and link-budget outputs may fall under ITAR/
  EAR depending on use; the README's disclaimer applies to anything the
  tools produce.
