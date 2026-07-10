---
description: >-
  APAB is a local-first agentic workflow for phased-array antenna design in
  Python: an LLM agent drives typed MCP engineering tools and records every
  run as an auditable, traceable bundle.
---

# APAB — Agentic Phased Array Builder

APAB connects an LLM agent to typed engineering tools for phased-array
antenna design: full-wave unit-cell simulation with mutual coupling,
array patterns, and system-level link metrics. It runs local-first on
[Ollama](https://ollama.ai) by default, and every run produces a
reproducible bundle with an audit log, provenance manifest, and
OpenTelemetry trace.

![apab run: the agent computes an 8x8 pattern, evaluates the link, and writes an audited run bundle](assets/demo.svg)

## Install

```bash
pip install "apab[ollama]"            # array + system tools, local LLM
pip install "apab[ollama,edgefem]"    # + full-wave unit-cell solver
pip install "apab[observability]"     # + OpenTelemetry tracing
```

## What APAB is

- **17 MCP tools** for unit-cell simulation (EdgeFEM), array patterns,
  system trades, import/export, and plotting
- **An agent orchestrator** that plans and executes designs from
  natural-language requests
- **A recorded workflow**: tool calls, token usage, config hashes, and
  design metrics land in a run bundle you can inspect, score, and replay
- **Framework-neutral**: the same tools drive APAB's own agent, a
  [Strands agent](tutorials/strands-adapter.md), or a deterministic
  [LangGraph pipeline](tutorials/langgraph-pipeline.md)

## What APAB is not

APAB does not replace full-wave EM verification or an RF engineer's
judgment. It sequences physics tools, enforces the constraints you give
it, and shows its work — treat its outputs as engineering analysis to
review, not signed-off designs.

## Start here

- [Quickstart](quickstart.md) — a working agent session in five minutes
- [Tutorials](tutorials/ollama-mcp-agent.md) — Ollama, tracing, Strands,
  LangGraph, and the 28 GHz case study
- [Concepts](concepts/architecture.md) — architecture, run bundles,
  security model, plugins
- [API reference](reference/schemas.md) — generated from the source

Source, issues, and releases live on
[GitHub](https://github.com/jman4162/agentic-phased-array-builder);
releases ship to [PyPI](https://pypi.org/project/apab/) via Trusted
Publishing.
