---
description: >-
  Anatomy of an APAB run bundle: audit.json, manifest.json, trace.jsonl,
  artifacts, and the provenance hashing that makes runs reproducible.
---

# Run bundles

A run bundle is the durable record of one agent session or pipeline
run. Chat logs tell you what the model said; the bundle tells you what
actually happened: which tools ran, with what inputs, producing which
artifacts, under which exact software versions.

```text
workspace/runs/<run_id>/
├── audit.json            tool-call audit log
├── manifest.json         provenance manifest
├── trace.jsonl           OpenTelemetry spans (when tracing is enabled)
├── checkpoint.sqlite     LangGraph pipeline state (pipeline runs only)
└── artifacts/
    ├── coupling/  patterns/  system/  emtool/  plots/  report/
```

Run IDs are timestamps plus a random suffix
(`20260709T190027_836ececc`), so bundles sort chronologically and never
collide.

## audit.json

One entry per tool call:

```json
{
  "timestamp": "2026-07-09T19:00:28.412Z",
  "tool": "pattern_compute",
  "arguments": {"nx": 8, "ny": 8, "freq_hz": 2.8e10},
  "result_summary": "{'directivity_dbi': 12.99, ...",
  "trace_id": "352932bb443149e4...",
  "span_id": "00f067aa0ba902b7"
}
```

`arguments` and `result_summary` respect the configured
[redaction mode](security-redaction.md); the trace IDs appear when
observability is enabled and join the entry to its span.

## manifest.json

The reproducibility record, written on every `run_to_completion` and
every pipeline run:

- `config_hash`, plus `geometry_hash` and `sweep_hash` when those
  sections exist: sha256-derived, so any input change is visible
- `dependency_versions`: apab, edgefem, the phased-array packages,
  numpy, scipy, pydantic, mcp
- `status`: `success`, `error`, `max_turns`, or
  `constraint_violation` (pipeline)
- `usage`: prompt/completion tokens, cost estimate, LLM call count
- `provider_name`, `model_name`, `artifacts`, `trace_id`

## trace.jsonl

One JSON object per OpenTelemetry span, written by a dependency-light
exporter so the file reads with `json.loads` alone. Span hierarchy and
attributes are documented in the
[observability reference](../observability.md).

## Why hashes instead of copies

The manifest stores hashes of config/geometry/sweep rather than
duplicating them: enough to prove which inputs produced which outputs
and to key the result cache, without bloating every bundle. The config
itself lives in your project's `apab.yaml`, under version control where
it belongs.

## Scoring bundles

The [eval harness](../reference/evals.md) treats bundles as the ground
truth: expected tool sequences match against `audit.json`, budgets and
status against `manifest.json`, and metric thresholds against tool
results. If you can score it, you can regression-test it.
