---
description: >-
  Install APAB, scaffold a project, and run your first agent-driven
  phased-array analysis in five minutes — no C++ solver build required.
---

# Quickstart

This gets you from zero to an agent-driven array analysis without
EdgeFEM (the C++ full-wave solver). Array patterns and system metrics
work with pure-Python dependencies.

## 1. Install

```bash
pip install "apab[ollama]"
```

Install [Ollama](https://ollama.ai), then pull the default model:

```bash
ollama pull qwen2.5-coder:14b
```

Any Ollama model with tool-calling support works; set it in `apab.yaml`
under `llm.model`.

## 2. Scaffold a project

```bash
apab init --name my_array --quickstart
cd my_array
```

`--quickstart` writes a ready-to-run `apab.yaml` for an 8x8 array with
no solver dependency. Use `--quickstart-fullwave` instead if you have
EdgeFEM installed.

## 3. Check your environment

```bash
apab doctor
```

This verifies Python dependencies, EdgeFEM availability, and that the
Ollama server is reachable with your configured model.

## 4. Run the agent

Interactive session:

```bash
apab design
```

```text
you> Compute the pattern for this array at 28 GHz and evaluate a
     500 m comms link with 200 MHz bandwidth.
```

Or non-interactive, driven by the config:

```bash
apab run --config apab.yaml
```

The agent picks tools (`pattern_compute`, `system_evaluate`, ...),
executes them, and reports metrics. Every session writes a run bundle:

```text
workspace/runs/<run_id>/
├── audit.json       every tool call, with arguments per your redaction mode
├── manifest.json    config hash, dependency versions, status, token usage
└── artifacts/       patterns, plots, system results, report
```

## 5. Next steps

- Turn on [OpenTelemetry tracing](tutorials/tracing-agent-runs.md) to
  see the full span tree for each run
- Read [run bundles](concepts/run-bundles.md) to understand what gets
  recorded and why
- Try the [LangGraph pipeline](tutorials/langgraph-pipeline.md) when you
  want the same analysis with no LLM in the loop
