#!/usr/bin/env python3
"""Run the tool-surface ablation: same tasks, two tool surfaces, N models.

Each (model, surface) pair runs in its own subprocess, because a process can
hold only one tool surface. Every task attempt appends one JSON line to the
results file, so an interrupted sweep keeps what it finished and ``--resume``
skips completed attempts.

Providers:
    ollama:<model>        local Ollama, e.g. ollama:qwen3.5:9b
    openrouter:<model>    OpenRouter, e.g. openrouter:qwen/qwen-2.5-coder-32b-instruct
                          (key read from $OPENROUTER_API_KEY, never written anywhere)

Usage:
    python evals/ablation/run_ablation.py --models ollama:qwen3.5:9b --repeats 1 \
        --tasks comms_scan_45_eirp radar_rain_no_rate --out evals/ablation/results/pilot.jsonl
    python evals/ablation/run_ablation.py --models ollama:qwen3.5:9b --surfaces current \
        --tasks-file evals/ablation/tasks_xband.yaml --out evals/ablation/results/xband.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from evals.ablation.references import TASKS_PATH, load_tasks, refs_path_for  # noqa: E402
from evals.ablation.score import classify, tool_call_stats  # noqa: E402
from evals.ablation.surfaces import SURFACES  # noqa: E402

OPENROUTER_URL = "https://openrouter.ai/api/v1"
ANSWER_INSTRUCTION = (
    "Use the available tools to compute the result. Finish your reply with a "
    "final line of the form `ANSWER: <number>` giving {what} in {unit}. If the "
    "request cannot be carried out as stated with the available tools, finish "
    "with `ANSWER: NONE` and explain why."
)


def make_provider(spec: str) -> Any:
    kind, _, model = spec.partition(":")
    if kind == "ollama":
        from apab.providers.ollama import OllamaProvider

        return OllamaProvider(model=model)
    if kind == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY is not set")
        from apab.providers.openai_compat import OpenAICompatibleProvider

        return OpenAICompatibleProvider(base_url=OPENROUTER_URL, model=model, api_key=key)
    raise SystemExit(f"unknown provider kind in {spec!r}; use ollama:<m> or openrouter:<m>")


def build_prompt(task: dict[str, Any]) -> str:
    ans = task["answer"]
    what = ans.get("describe", ans["metric"].replace("_", " "))
    return task["prompt"] + "\n\n" + ANSWER_INSTRUCTION.format(what=what, unit=ans["unit"])


def run_one(
    task: dict[str, Any],
    model: str,
    surface: str,
    refs: dict[str, Any],
    workspace: Path,
    max_turns: int,
) -> dict[str, Any]:
    from apab.agent.orchestrator import AgentOrchestrator
    from apab.core.schemas import LLMSpec, ProjectConfig, ProjectMeta

    config = ProjectConfig(
        project=ProjectMeta(name="ablation", workspace=str(workspace)),
        llm=LLMSpec(provider=model.split(":", 1)[0], model=model.split(":", 1)[1]),
    )
    orch = AgentOrchestrator(config, provider=make_provider(model))
    t0 = time.monotonic()
    text, error = "", None
    try:
        text = orch.run_to_completion(build_prompt(task), max_turns=max_turns)
    except Exception as exc:  # score whatever happened; record the failure
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - t0

    run_dir = orch.run_context.run_dir if orch.run_context else None
    audit = (
        json.loads((run_dir / "audit.json").read_text())
        if run_dir and (run_dir / "audit.json").exists()
        else []
    )
    manifest = (
        json.loads((run_dir / "manifest.json").read_text())
        if run_dir and (run_dir / "manifest.json").exists()
        else {}
    )

    entry = refs["tasks"][task["name"]]
    ref = (entry.get("reference") or {}).get("value")
    ign = (entry.get("ignored_reference") or {}).get("value")
    scored = classify(task, text, ref, ign)
    usage = manifest.get("usage", {})
    return {
        "model": model,
        "surface": surface,
        "task": task["name"],
        "category": task["category"],
        **scored,
        "reference": ref,
        "ignored_reference": ign,
        **tool_call_stats(audit),
        "tools_called": [e.get("tool") for e in audit],
        "llm_calls": usage.get("llm_calls"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "seconds": round(elapsed, 2),
        "error": error,
        "final_text": text[-2000:],
        "run_dir": str(run_dir) if run_dir else None,
    }


def worker(
    model: str,
    surface: str,
    task_names: list[str],
    repeats: int,
    out: Path,
    done: set[tuple[str, str, str, int]],
    max_turns: int,
    tasks_file: Path = TASKS_PATH,
) -> None:
    from evals.ablation.surfaces import use_surface

    out = out.resolve()
    use_surface(surface)
    refs = json.loads(refs_path_for(tasks_file).read_text())
    tasks = [t for t in load_tasks(tasks_file) if not task_names or t["name"] in task_names]
    # Keep every run bundle (audit.json, manifest.json) next to the results.
    slug = "".join(c if c.isalnum() or c in "-." else "_" for c in model)
    workspace = (out.parent / f"{out.stem}_runs" / f"{slug}__{surface}").resolve()
    # Tools write relative output paths against the process cwd (plots,
    # project_init); run from inside the workspace so they land there.
    workspace.mkdir(parents=True, exist_ok=True)
    os.chdir(workspace)
    for rep in range(repeats):
        for task in tasks:
            key = (model, surface, task["name"], rep)
            if key in done:
                continue
            row = run_one(task, model, surface, refs, workspace, max_turns)
            row["repeat"] = rep
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(
                f"{model} {surface} {task['name']} r{rep}: {row['outcome']}"
                f" ({row['tool_calls']} calls, {row['failed_tool_calls']} failed,"
                f" {row['seconds']}s)",
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--surfaces", nargs="+", default=list(SURFACES), choices=SURFACES)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--tasks-file", type=Path, default=TASKS_PATH)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--out", type=Path, default=HERE / "results" / "ablation.jsonl")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--_worker", nargs=2, metavar=("MODEL", "SURFACE"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str, str, int]] = set()
    if args.resume and args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["model"], r["surface"], r["task"], r["repeat"]))

    if args._worker:
        model, surface = args._worker
        worker(
            model,
            surface,
            args.tasks,
            args.repeats,
            args.out,
            done,
            args.max_turns,
            args.tasks_file.resolve(),
        )
        return 0

    refs_path = refs_path_for(args.tasks_file)
    if not refs_path.exists():
        raise SystemExit(f"{refs_path.name} missing: run evals/ablation/references.py first")
    for model in args.models:
        for surface in args.surfaces:
            cmd = [
                sys.executable,
                __file__,
                "--_worker",
                model,
                surface,
                "--models",
                model,
                "--repeats",
                str(args.repeats),
                "--max-turns",
                str(args.max_turns),
                "--out",
                str(args.out),
                "--tasks-file",
                str(args.tasks_file.resolve()),
            ]
            if args.tasks:
                cmd += ["--tasks", *args.tasks]
            if args.resume:
                cmd.append("--resume")
            print(f"== {model} / {surface}", flush=True)
            subprocess.run(cmd, cwd=ROOT, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
