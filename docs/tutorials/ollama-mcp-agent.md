---
description: >-
  Build a local-first engineering agent with Ollama and MCP tools: drive
  APAB's orchestrator programmatically and inspect the tool-calling loop.
---

# Local agent with Ollama and MCP

APAB's agent is an LLM tool-calling loop over an MCP server. This
tutorial drives it programmatically so you can see each moving part;
`apab design` and `apab run` wrap the same orchestrator.

## The pieces

1. **Provider**: anything implementing the `LLMProvider` protocol:
   `chat(messages, tools) -> {content, tool_calls}`. The default
   `OllamaProvider` talks to a local Ollama server, so no data leaves
   your machine.
2. **MCP tools**: 17 typed tools registered on a FastMCP server
   (`pattern_compute`, `system_evaluate`, `edgefem_run_unit_cell`, ...).
   The orchestrator reads their JSON schemas and passes them to the
   provider on every turn.
3. **Orchestrator**: `run_to_completion` loops: ask the model, execute
   any tool calls, feed results back, stop when the model answers in
   text or the turn budget runs out.

## A programmatic session

```python
from apab.agent.orchestrator import AgentOrchestrator
from apab.core.schemas import ProjectConfig, ProjectMeta

config = ProjectConfig(
    project=ProjectMeta(name="demo", workspace="./workspace"),
)
# llm defaults: provider="ollama", model="qwen2.5-coder:14b"

orch = AgentOrchestrator(config)
result = orch.run_to_completion(
    "Compute the pattern for an 8x8 array at 28 GHz with half-wave "
    "spacing and report directivity and sidelobe level."
)
print(result)
print(orch.session_usage)   # tokens, cost estimate, LLM call count
```

## Watching the loop

`run_to_completion` accepts an `on_event` callback that fires as the
loop progresses. The CLI uses it for its live rendering, and you can
use it for your own:

```python
def on_event(name: str, payload: dict) -> None:
    if name == "tool_call":
        print(f"-> {payload['name']}({list(payload['arguments'])})")
    elif name == "tool_result":
        print(f"<- {payload['tool']}: {payload['result'][:80]}")

orch.run_to_completion("...", on_event=on_event)
```

Events: `session_start`, `turn_start`, `tool_call`, `tool_result`,
`assistant_message`, `max_turns`.

## Scripting without an LLM

For tests and demos, pass a scripted provider:
`examples/04_agent_session.py` shows a `DemoProvider` that returns a
fixed tool-call sequence. The whole loop, including real tool execution
and the run bundle, works with no model attached.

## Where results land

Every session writes `workspace/runs/<run_id>/` with `audit.json`,
`manifest.json`, and artifacts. See [run bundles](../concepts/run-bundles.md)
for the anatomy and [tracing agent runs](tracing-agent-runs.md) to add
OpenTelemetry spans on top.
