---
description: >-
  APAB's plugin system: the LLMProvider protocol, per-call usage
  tracking, and entry-point groups for providers, EM adapters, and
  compute backends.
---

# Providers and plugins

APAB extends through Python entry points — install a package that
registers the right group and APAB discovers it, no fork needed.

## The LLMProvider protocol

A provider is any class with:

```python
@property
def name(self) -> str: ...
def supports_tool_calling(self) -> bool: ...
def supports_streaming(self) -> bool: ...
def chat(self, messages, tools=None, **kwargs) -> dict: ...
```

`chat` returns a normalized response:

```python
{"role": "assistant", "content": "...", "tool_calls": [{"name": ..., "arguments": {...}}]}
```

Five providers ship built-in: `ollama` (default, fully local),
`openai`, `anthropic`, `gemini`, and `openai_compatible` (vLLM,
LM Studio, Together, and any other OpenAI-style endpoint via
`base_url`).

## Per-call usage tracking

Each built-in provider exposes `last_usage` — a `ProviderUsage` with
prompt/completion tokens, latency, and a cost estimate for the most
recent call. It is deliberately not part of the protocol, so
third-party providers without it keep working; consumers read it with
`getattr(provider, "last_usage", None)`. The orchestrator accumulates
these into session totals that land in the manifest and on the session
span.

## Registering a provider

```toml
# your package's pyproject.toml
[project.entry-points."apab.llm_providers"]
my_provider = "my_pkg.provider:MyProvider"
```

```yaml
# apab.yaml
llm:
  provider: my_provider
  model: whatever-your-provider-expects
```

The Ollama-specific fallback parser is worth knowing about: some local
models emit tool calls as JSON in text instead of structured
`tool_calls`; `OllamaProvider` detects and recovers those, which is
part of why small local models work at all.

## Other entry-point groups

| Group | Purpose | Shipped implementations |
|---|---|---|
| `apab.llm_providers` | LLM backends | ollama, openai, anthropic, gemini, openai_compatible |
| `apab.em_adapters` | Commercial EM tool import (HFSS/CST/FEKO) | none yet — `emtool_import_results` defines the seam |
| `apab.compute_backends` | Execution backends | local |

The `ComputeBackend` protocol anticipates cloud executors (AWS/GCP)
without changing tool code: tools ask the backend to run, and the
backend decides where.
