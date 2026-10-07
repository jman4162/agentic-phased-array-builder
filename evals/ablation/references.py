#!/usr/bin/env python3
"""Compute reference answers for the ablation tasks.

Calls each task's ``reference`` and ``ignored_reference`` tool directly (no
LLM) and writes ``references.json`` with the values and the package versions
they were computed with. ``v04`` references run in a subprocess because a
process can hold only one tool surface.

Usage:
    python evals/ablation/references.py            # writes evals/ablation/references.json
    python evals/ablation/references.py --check    # fail if any task does not discriminate
    python evals/ablation/references.py --tasks-file evals/ablation/tasks_xband.yaml
                                                   # writes references_xband.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TASKS_PATH = HERE / "tasks.yaml"
OUT_PATH = HERE / "references.json"


def refs_path_for(tasks_path: Path) -> Path:
    """References file paired with a tasks file: tasks_<x>.yaml -> references_<x>.json."""
    if tasks_path.name == TASKS_PATH.name:
        return OUT_PATH
    return tasks_path.with_name(f"references_{tasks_path.stem.removeprefix('tasks_')}.json")


def load_tasks(path: Path = TASKS_PATH) -> list[dict[str, Any]]:
    """Load tasks, filling the shared prompt fragments (top-level string keys)."""
    doc = yaml.safe_load(path.read_text())
    fragments = {k: v for k, v in doc.items() if isinstance(v, str)}
    tasks = doc["tasks"]
    for task in tasks:
        task["prompt"] = " ".join(task["prompt"].format(**fragments).split())
    return tasks


def _call(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    from apab.mcp import tools_system  # noqa: F401  (registers tools)
    from apab.mcp.server import get_mcp

    server = get_mcp()
    fn = server._tool_manager.get_tool(tool).fn  # direct call, same as dispatcher
    result: dict[str, Any] = asyncio.run(fn(**args))
    return result


def evaluate_refs(
    surface: str, refs: list[tuple[str, str, dict[str, Any]]], metric_of: dict[str, str]
) -> dict[str, Any]:
    """Evaluate (key, tool, args) refs on one surface in this process."""
    sys.path.insert(0, str(ROOT))
    from evals.ablation.surfaces import use_surface

    use_surface(surface)
    out = {}
    for key, tool, args in refs:
        result = _call(tool, args)
        metric = metric_of[key]
        out[key] = {
            "value": result.get(metric),
            "status": result.get("status"),
            "error": result.get("error"),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--tasks-file", type=Path, default=TASKS_PATH)
    parser.add_argument("--_worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker:  # subprocess entry: evaluate refs from a JSON payload on stdin
        payload = json.loads(sys.stdin.read())
        refs = [tuple(r) for r in payload["refs"]]
        print(json.dumps(evaluate_refs(args._worker, refs, payload["metric_of"])))  # type: ignore[arg-type]
        return 0

    tasks = load_tasks(args.tasks_file)
    by_surface: dict[str, list[tuple[str, str, dict[str, Any]]]] = {"v04": [], "v05": []}
    metric_of: dict[str, str] = {}
    for task in tasks:
        for kind in ("reference", "ignored_reference"):
            ref = task.get(kind)
            if not ref:
                continue
            key = f"{task['name']}:{kind}"
            by_surface[ref.get("surface", "v05")].append((key, ref["tool"], ref["args"]))
            metric_of[key] = task["answer"]["metric"]

    values: dict[str, Any] = {}
    for surface, refs in by_surface.items():
        if not refs:
            continue
        proc = subprocess.run(
            [sys.executable, __file__, "--_worker", surface],
            input=json.dumps({"refs": refs, "metric_of": metric_of}),
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        )
        values.update(json.loads(proc.stdout.strip().splitlines()[-1]))

    problems = []
    report: dict[str, Any] = {}
    for task in tasks:
        name, tol = task["name"], task["answer"]["tol"]
        ref = values.get(f"{name}:reference")
        ign = values.get(f"{name}:ignored_reference")
        entry = {"reference": ref, "ignored_reference": ign}
        report[name] = entry
        if ref is not None and not isinstance(ref["value"], (int, float)):
            problems.append(f"{name}: reference has no numeric {task['answer']['metric']}: {ref}")
        elif ref is not None and not math.isfinite(ref["value"]):
            problems.append(f"{name}: reference is not finite: {ref}")
        if (
            ref
            and ign
            and isinstance(ign["value"], (int, float))
            and isinstance(ref["value"], (int, float))
        ):
            if abs(ref["value"] - ign["value"]) <= max(tol, 1e-12):
                problems.append(
                    f"{name}: reference and ignored_reference agree within tol "
                    f"({ref['value']} vs {ign['value']})"
                )

    doc = {
        "versions": {
            p: version(p) for p in ("apab", "phased-array-systems", "phased-array-modeling")
        },
        "tasks": report,
    }
    refs_path_for(args.tasks_file).write_text(json.dumps(doc, indent=2) + "\n")
    for task in tasks:
        e = report[task["name"]]
        r = e["reference"]["value"] if e["reference"] else None
        i = e["ignored_reference"]["value"] if e["ignored_reference"] else None
        print(f"{task['category']:8s} {task['name']:28s} ref={r!s:>24s}  ignored={i!s:>24s}")
    for p in problems:
        print("PROBLEM:", p, file=sys.stderr)
    return 1 if (args.check and problems) else 0


if __name__ == "__main__":
    sys.exit(main())
