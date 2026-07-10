---
description: >-
  Use APAB's MCP engineering tools from a Strands Agents agent over
  stdio, with Strands' own OpenTelemetry tracing.
---

# Strands Agents adapter

APAB's tool layer speaks MCP, so any MCP-capable agent framework can
drive it. The Strands adapter launches APAB's MCP server as a stdio
subprocess and hands it to Strands' `MCPClient` — the integration rides
the public MCP protocol, not APAB internals.

## Install

```bash
pip install "apab[strands]" "strands-agents[ollama,otel]"
```

## Minimal agent

```python
from strands import Agent
from strands.models.ollama import OllamaModel

from apab.adapters.strands import apab_mcp_client, apab_system_prompt

client = apab_mcp_client()          # launches `apab mcp serve --transport stdio`
model = OllamaModel(host="http://localhost:11434", model_id="qwen2.5-coder:14b")

with client:
    agent = Agent(
        model=model,
        tools=client.list_tools_sync(),      # all 17 APAB tools
        system_prompt=apab_system_prompt(),  # APAB's own agent prompt
    )
    agent(
        "Compute the array pattern for an 8x8 phased array at 28 GHz "
        "with half-wavelength spacing, then evaluate the system metrics."
    )
```

`apab_mcp_client(config_path="apab.yaml")` forwards a project config to
the server, so workspace and solver settings apply.

## Tracing from the Strands side

Strands emits its own OpenTelemetry spans for the agent loop, model
calls, and every APAB tool invocation:

```python
from strands.telemetry import StrandsTelemetry

StrandsTelemetry().setup_console_exporter()
# or, with the Jaeger lab running:
# OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 -> setup_otlp_exporter()
```

APAB tools appear as tool spans inside Strands' trace — useful when
APAB is one toolbox among several in a larger agent.

## When to use which frontend

- **APAB's own orchestrator**: local-first sessions with run bundles,
  audit logs, and APAB-side tracing.
- **Strands adapter**: you already build on Strands, or you want APAB's
  tools alongside other toolkits under one agent and one trace.
- **[LangGraph pipeline](langgraph-pipeline.md)**: reproducible runs
  with no LLM in the loop.

The runnable version of this tutorial is
[`examples/07_strands_agent.py`](https://github.com/jman4162/agentic-phased-array-builder/blob/main/examples/07_strands_agent.py).
