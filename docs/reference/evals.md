---
description: >-
  Golden-task evaluation format: YAML task definitions and how runs are
  scored from their bundles.
---

# Evals

The eval harness (`evals/run_evals.py`, repo-level, not shipped in the
wheel) runs YAML-defined golden tasks through the agent and scores each
run from its [bundle](../concepts/run-bundles.md) — no LLM judge, just
recorded facts.

## Task format

```yaml
# evals/golden/pattern_28ghz_basic.yaml
name: pattern_28ghz_basic
description: Single pattern computation with a directivity floor.
prompt: >
  Compute the array pattern for an 8x8 phased array at 28 GHz with
  5.4 mm element spacing and report the directivity in dBi.
expected_tools:
  - pattern_compute
max_llm_calls: 6
metrics:
  directivity_dbi:
    min: 10.0
```

## Scoring semantics

| Check | Source | Rule |
|---|---|---|
| `tool_sequence` | `audit.json` | `expected_tools` must appear in order as a subsequence of the calls made; extra calls in between are allowed |
| `status` | `manifest.json` | must be `success` |
| `llm_calls` | `manifest.json` usage | at most `max_llm_calls` provider calls |
| `metric:<name>` | tool result summaries in `audit.json` | numeric value extracted by name; latest occurrence wins; compared against `min`/`max` bounds |

A task passes only if every check passes. Results are written to
`evals/results/<timestamp>.json` with per-check detail.

## Running

```bash
python evals/run_evals.py --config apab.yaml
python evals/run_evals.py --config apab.yaml --tasks evals/golden --out evals/results
```

Exit code 0 means all tasks passed; 1 means at least one failed. The
scorer is pure functions over bundle contents
(`check_tool_sequence`, `extract_metric`, `score_run`), unit-tested on
canned fixtures so CI exercises the scoring logic without a model.

## What to use it for

- Comparing models: run the same tasks against two Ollama models and
  diff the per-check results
- Prompt changes: catch regressions in tool selection after editing the
  system prompt
- Tracking small-model capability over time via the scheduled CI run
