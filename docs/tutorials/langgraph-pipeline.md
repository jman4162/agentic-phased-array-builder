---
description: >-
  Run a deterministic phased-array engineering workflow as a LangGraph
  graph with SQLite checkpointing, constraint gates, and no LLM.
---

# LangGraph pipeline

The agent picks tools turn by turn; the LangGraph pipeline runs a fixed
engineering sequence instead, with no LLM involved. Use it when the
workflow is known in advance and the run must be repeatable: same
config in, same bundle out, resumable if it dies partway.

```text
validate_config
-> unit_cell        (only with EdgeFEM installed and configured)
-> pattern
-> system_eval
-> constraint_check
-> plots
-> report
```

Nodes are plain Python callables that dispatch APAB's MCP tools
in-process. Results land in a normal run bundle whose manifest status
reflects errors and constraint violations.

## Install and run

```bash
pip install "apab[langgraph]"
```

```python
from apab.adapters.langgraph_pipeline import Constraints, Scenario, run_pipeline
from apab.core.schemas import ProjectConfig

config = ProjectConfig.model_validate({
    "project": {"name": "pipeline_demo", "workspace": "./workspace"},
    "array": {"size": [8, 8], "spacing_m": [0.0054, 0.0054], "taper": "taylor"},
})

state = run_pipeline(
    config,
    scenario=Scenario(bandwidth_hz=200e6, range_m=500.0),
    constraints=Constraints(min_directivity_dbi=20.0,
                            max_sidelobe_level_db=-15.0),
)
print(state["pattern"]["directivity_dbi"], state["violations"])
```

## Streaming progress and checkpoints

`build_pipeline` returns the compiled graph for finer control:

```python
from apab.adapters.langgraph_pipeline import build_pipeline

graph, run_ctx, initial_state = build_pipeline(config)
thread = {"configurable": {"thread_id": run_ctx.run_id}}

for update in graph.stream(initial_state, config=thread, stream_mode="updates"):
    print(update)          # one dict per completed node
```

With checkpointing on (the default), state persists to
`<run_dir>/checkpoint.sqlite` keyed by the run ID. Re-invoking the same
thread resumes from the saved state instead of recomputing, which
matters when a long sweep dies at node five of seven.

## Constraint gates

`constraint_check` compares pattern metrics against your thresholds and
records violations in the state and the manifest
(`status: constraint_violation`). The pipeline still completes and
writes its report, so a failed gate leaves you the evidence, not a
half-empty bundle.

## Tracing

Each node runs inside an `apab.node.<name>` span when
[observability](tracing-agent-runs.md) is enabled, so pipeline runs
show up in Jaeger alongside agent runs.

The runnable version is
[`examples/08_langgraph_golden_pipeline.py`](https://github.com/jman4162/agentic-phased-array-builder/blob/main/examples/08_langgraph_golden_pipeline.py).
One dependency note: langgraph brings `langchain-core` transitively;
APAB uses no LangChain model wrappers.
