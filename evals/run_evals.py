#!/usr/bin/env python3
"""Run golden-task evaluations against a configured LLM provider.

Each YAML task in evals/golden/ gives the agent a prompt and scores
the resulting run bundle:

- tool_sequence: expected_tools appear in the audit log, in order
  (ordered-subset match; extra calls in between are allowed)
- status: manifest.json reports success
- llm_calls: at most max_llm_calls provider calls
- metrics: numeric thresholds against values found in the audit log's
  tool result summaries (e.g. directivity_dbi >= 10)

Usage:
    python evals/run_evals.py --config apab.yaml
    python evals/run_evals.py --config apab.yaml --tasks evals/golden --out evals/results

Scoring is separate from execution so it can be unit-tested on canned
run bundles without an LLM.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_task(path: Path) -> dict[str, Any]:
    task: dict[str, Any] = yaml.safe_load(path.read_text())
    task.setdefault("expected_tools", [])
    task.setdefault("metrics", {})
    return task


# ── scoring (pure functions over run-bundle contents) ──────────────


def check_tool_sequence(
    expected: list[str], called: list[str],
) -> tuple[bool, str]:
    """Ordered-subset match of expected tools within the called list."""
    it = iter(called)
    missing = [tool for tool in expected if tool not in it]
    if missing:
        return False, f"missing or out of order: {missing} (called: {called})"
    return True, f"matched {expected}"


_METRIC_RE_CACHE: dict[str, re.Pattern[str]] = {}


def extract_metric(audit: list[dict[str, Any]], metric: str) -> float | None:
    """Find a numeric metric in tool result summaries, latest wins."""
    pattern = _METRIC_RE_CACHE.setdefault(
        metric,
        re.compile(rf"['\"]?{re.escape(metric)}['\"]?\s*[:=]\s*(-?\d+(?:\.\d+)?)"),
    )
    value = None
    for entry in audit:
        summary = str(entry.get("result_summary", ""))
        m = pattern.search(summary)
        if m:
            value = float(m.group(1))
    return value


def score_run(
    task: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """Score one task from its run bundle. Returns per-check results."""
    audit_path = run_dir / "audit.json"
    manifest_path = run_dir / "manifest.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else []
    manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )

    checks: dict[str, dict[str, Any]] = {}

    called = [e["tool"] for e in audit]
    ok, detail = check_tool_sequence(task["expected_tools"], called)
    checks["tool_sequence"] = {"passed": ok, "detail": detail}

    status = manifest.get("status", "missing")
    checks["status"] = {
        "passed": status == "success",
        "detail": f"manifest status = {status}",
    }

    max_calls = task.get("max_llm_calls")
    if max_calls is not None:
        llm_calls = manifest.get("usage", {}).get("llm_calls", 0)
        checks["llm_calls"] = {
            "passed": llm_calls <= max_calls,
            "detail": f"{llm_calls} calls (limit {max_calls})",
        }

    for metric, bounds in task["metrics"].items():
        value = extract_metric(audit, metric)
        if value is None:
            checks[f"metric:{metric}"] = {
                "passed": False,
                "detail": "not found in tool results",
            }
            continue
        passed = True
        if "min" in bounds and value < bounds["min"]:
            passed = False
        if "max" in bounds and value > bounds["max"]:
            passed = False
        checks[f"metric:{metric}"] = {
            "passed": passed,
            "detail": f"{metric} = {value} (bounds: {bounds})",
        }

    return {
        "task": task["name"],
        "passed": all(c["passed"] for c in checks.values()),
        "checks": checks,
        "run_dir": str(run_dir),
    }


# ── execution ───────────────────────────────────────────────────────


def run_task(task: dict[str, Any], config: Any) -> dict[str, Any]:
    from apab.agent.orchestrator import AgentOrchestrator

    orch = AgentOrchestrator(config)
    try:
        orch.run_to_completion(
            task["prompt"], max_turns=task.get("max_llm_calls", 10),
        )
    except Exception as exc:
        # Score whatever the bundle recorded before the failure.
        print(f"  [{task['name']}] run failed: {exc}", file=sys.stderr)
    return score_run(task, orch.run_context.run_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="apab.yaml")
    parser.add_argument("--tasks", default="evals/golden")
    parser.add_argument("--out", default="evals/results")
    args = parser.parse_args()

    from apab.core.config import load_config

    config = load_config(Path(args.config))

    task_files = sorted(Path(args.tasks).glob("*.yaml"))
    if not task_files:
        print(f"No task files in {args.tasks}", file=sys.stderr)
        return 2

    results = []
    for path in task_files:
        task = load_task(path)
        print(f"Running {task['name']} ...")
        result = run_task(task, config)
        results.append(result)
        marker = "PASS" if result["passed"] else "FAIL"
        print(f"  {marker}")
        for name, check in result["checks"].items():
            sub = "ok" if check["passed"] else "FAIL"
            print(f"    {name}: {sub} — {check['detail']}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"eval_{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2))

    n_pass = sum(r["passed"] for r in results)
    print(f"\n{n_pass}/{len(results)} tasks passed. Results: {out_path}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
