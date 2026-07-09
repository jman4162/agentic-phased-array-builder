# Observability

APAB can record every agent session as an OpenTelemetry trace: one span
per session, turn, LLM call, and tool call. Traces land in three places:

1. `trace.jsonl` in the run bundle (on by default when tracing is enabled)
2. stdout, via the console exporter
3. any OTLP endpoint (Jaeger, Grafana Tempo, vendor backends)

Tracing is off by default and adds no required dependencies. The
`audit.json` and `manifest.json` files are written on every run whether
or not tracing is enabled; when it is enabled, both carry the trace ID
so you can cross-reference them with the span data.

## Setup

```bash
pip install "apab[observability]"
```

Enable it in `apab.yaml`:

```yaml
observability:
  enabled: true
  service_name: apab
  console_exporter: false      # print spans to stdout
  otlp_endpoint: null          # e.g. http://localhost:4318
  trace_jsonl: true            # write <run_dir>/trace.jsonl
  capture_mode: null           # null inherits llm.redaction_mode
  set_global: false            # install as the OTel global provider
```

Environment overrides:

| Variable | Effect |
| --- | --- |
| `APAB_OBSERVABILITY=1` | Force-enable tracing regardless of config |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint when `otlp_endpoint` is unset |

If `enabled: true` but the `opentelemetry` packages are missing, APAB
logs a warning and runs untraced. Core modules never import
OpenTelemetry directly, so `import apab` works without the extra.

## Span hierarchy

`AgentOrchestrator.run_to_completion` is the instrumented entry point.
Driving `start_session`/`step` manually still produces `apab.llm.chat`
and `apab.tool.*` spans, but without a session root.

```
apab.session                 one per run_to_completion
└── apab.turn                one per LLM turn
    ├── apab.llm.chat        the provider call
    └── apab.tool.<name>     one per tool call
```

### Attributes

| Span | Attribute | Meaning |
| --- | --- | --- |
| `apab.session` | `apab.run_id` | Run bundle ID |
| | `gen_ai.system` | Provider name (`ollama`, `openai`, ...) |
| | `gen_ai.request.model` | Configured model |
| | `apab.max_turns` | Turn budget |
| | `apab.config_hash` | Hash of the project config |
| | `apab.status` | `success`, `error`, or `max_turns` |
| | `gen_ai.usage.input_tokens` / `output_tokens` | Session totals |
| | `apab.cost_estimate_usd` | Session cost estimate |
| `apab.turn` | `apab.turn.index` | 0-based turn number |
| `apab.llm.chat` | `gen_ai.usage.input_tokens` / `output_tokens` | Per-call tokens |
| | `apab.latency_s` | Provider call latency |
| | `apab.cost_estimate_usd` | Per-call cost estimate |
| | `apab.tool_call_count` | Tool calls in the response |
| | `apab.response.has_content` | Whether text content was returned |
| `apab.tool.<name>` | `apab.tool.name` | Tool name |
| | `apab.tool.args_hash` | sha256[:16] of the arguments |
| | `apab.tool.args_json` / `arg_keys` | Arguments, per capture mode |
| | `apab.tool.status` | `ok` or `error` |
| | `apab.tool.result_summary` | Truncated result, per capture mode |

The `gen_ai.*` names follow the OpenTelemetry GenAI semantic
conventions, which are still experimental. APAB writes them as literal
strings; if the conventions change, only the attribute names change,
not the span structure.

### Capture modes and redaction

`observability.capture_mode` controls what tool arguments and results
appear in span attributes, with the same three levels as
`llm.redaction_mode` (which it inherits when unset):

- `none` captures the full argument JSON and truncated result summaries
- `metadata_only` captures argument key names and result lengths
- `strict` captures content hashes only

The argument hash is always recorded, so identical calls can be
correlated across runs at every capture level.

## trace.jsonl format

One JSON object per span, in export order:

```json
{
  "name": "apab.tool.pattern_compute",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id": "00f067aa0ba902b7",
  "parent_span_id": "53995c3f42cd8ad8",
  "start_time_unix_nano": 1767995000000000000,
  "end_time_unix_nano": 1767995001500000000,
  "status": "UNSET",
  "attributes": {"apab.tool.name": "pattern_compute", "...": "..."},
  "events": []
}
```

The file needs no OpenTelemetry tooling to read:

```python
import json, pathlib
spans = [json.loads(l) for l in pathlib.Path("trace.jsonl").read_text().splitlines()]
tools = [s for s in spans if s["name"].startswith("apab.tool.")]
```

## Viewing traces in Jaeger

A single container gives you a trace UI (see `lab/` for a compose file):

```bash
docker run --rm -d -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 APAB_OBSERVABILITY=1 \
    apab run --config apab.yaml
```

Then open http://localhost:16686 and select the `apab` service.

## Run bundle layout

```
workspace/runs/<run_id>/
├── audit.json       tool-call audit log (+ trace_id/span_id per entry)
├── manifest.json    provenance: config hash, dependency versions,
│                    status, token usage, trace_id
├── trace.jsonl      one JSON object per span
└── artifacts/       patterns, plots, system results, report
```
