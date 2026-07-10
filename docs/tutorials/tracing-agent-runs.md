---
description: >-
  Trace LLM tool calls with OpenTelemetry: enable APAB's tracing, read
  trace.jsonl, and view agent runs in Jaeger.
---

# Tracing agent runs

APAB records each agent session as an OpenTelemetry trace: one span per
session, turn, LLM call, and tool call, with token counts, latency,
cost estimates, and redaction-aware tool arguments.

## Enable tracing

```bash
pip install "apab[observability]"
```

```yaml
# apab.yaml
observability:
  enabled: true
```

That alone writes `trace.jsonl` into every run bundle. For ad-hoc runs,
`APAB_OBSERVABILITY=1` forces tracing without touching the config.

## Read the trace without any tooling

`trace.jsonl` is one JSON object per span:

```python
import json, pathlib

spans = [json.loads(line) for line in
         pathlib.Path("workspace/runs/<run_id>/trace.jsonl").read_text().splitlines()]

tools = [s for s in spans if s["name"].startswith("apab.tool.")]
for s in tools:
    a = s["attributes"]
    ms = (s["end_time_unix_nano"] - s["start_time_unix_nano"]) / 1e6
    print(f"{a['apab.tool.name']:20s} {a['apab.tool.status']:5s} {ms:8.1f} ms")
```

Questions this answers directly: which tool dominates wall time, which
calls failed on which turn, how many tokens each turn consumed, and
whether the agent repeated a call with the same argument hash.

## View traces in Jaeger

One container gives you a UI (the repo ships this as
[`lab/docker-compose.yml`](https://github.com/jman4162/agentic-phased-array-builder/tree/main/lab)):

```bash
cd lab && docker compose up -d

OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 APAB_OBSERVABILITY=1 \
    apab run --config apab.yaml
```

Open <http://localhost:16686>, pick the `apab` service, and expand a
trace: `apab.session` at the root, one `apab.turn` per LLM turn, and
`apab.llm.chat` plus `apab.tool.<name>` spans underneath.

## Cross-referencing the run bundle

The trace ID appears in three places, so records join cleanly:

- `manifest.json` carries the `trace_id` for the whole run
- `audit.json` carries `trace_id` and `span_id` on every tool-call entry
- `trace.jsonl` holds the spans themselves

## Redaction

Span attributes respect `observability.capture_mode` (inherits
`llm.redaction_mode` when unset): `none` records argument JSON,
`metadata_only` records key names, `strict` records only content
hashes. The hash is always present, so identical calls correlate across
runs at every level.

Full attribute tables live in the
[observability reference](../observability.md).
