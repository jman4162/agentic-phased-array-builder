#!/usr/bin/env python3
"""Summarize ablation results as markdown tables.

Rates per (model, surface): correct, confidently wrong (a wrong number
delivered as the answer, including silently wrong), silently wrong alone,
and safe failure (refused or no answer), each with a Wilson 95% interval.
Surface pairs (v05/v04, current/v04, current/v05) are compared per model
with an exact McNemar test on correctness, pairing the same (task, repeat).

Usage:
    python evals/ablation/analyze.py evals/ablation/results/local_sweep.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

CONFIDENTLY_WRONG = {"wrong", "silently_wrong"}
SAFE_FAILURE = {"refused", "no_answer"}

# Surface pairs compared; a pair is skipped when either arm has no rows.
PAIRS = (
    ("v05", "v04"),
    ("v051", "v04"),
    ("v051", "v05"),
    ("current", "v04"),
    ("current", "v05"),
    ("current", "v051"),
)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def fmt_rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} [{lo:.0%}, {hi:.0%}]" if n else "–"


def load(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for p in paths:
        rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return rows


def summarize(rows: list[dict[str, Any]]) -> str:
    out = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["surface"])].append(r)

    out.append("## Outcomes by model and surface\n")
    out.append(
        "| Model | Surface | Correct | Confidently wrong | (silently wrong) | Safe failure |"
    )
    out.append("|---|---|---|---|---|---|")
    for (model, surface), rs in sorted(groups.items()):
        n = len(rs)
        c = sum(r["outcome"] == "correct" for r in rs)
        cw = sum(r["outcome"] in CONFIDENTLY_WRONG for r in rs)
        sw = sum(r["outcome"] == "silently_wrong" for r in rs)
        sf = sum(r["outcome"] in SAFE_FAILURE for r in rs)
        out.append(
            f"| {model} | {surface} | {fmt_rate(c, n)} | {fmt_rate(cw, n)} "
            f"| {fmt_rate(sw, n)} | {fmt_rate(sf, n)} |"
        )

    out.append("\n## Correct rate by category\n")
    cats = sorted({r["category"] for r in rows})
    out.append("| Model | Surface | " + " | ".join(cats) + " |")
    out.append("|---|---|" + "---|" * len(cats))
    for (model, surface), rs in sorted(groups.items()):
        cells = []
        for cat in cats:
            sub = [r for r in rs if r["category"] == cat]
            k = sum(r["outcome"] == "correct" for r in sub)
            cells.append(f"{k}/{len(sub)}" if sub else "–")
        out.append(f"| {model} | {surface} | " + " | ".join(cells) + " |")

    out.append("\n## Paired comparisons (same task and repeat)\n")
    out.append("| Model | A vs B | Pairs | A only correct | B only correct | McNemar p |")
    out.append("|---|---|---|---|---|---|")
    for model in sorted({r["model"] for r in rows}):
        by_key: dict[tuple[str, int], dict[str, bool]] = defaultdict(dict)
        for r in rows:
            if r["model"] == model:
                by_key[(r["task"], r["repeat"])][r["surface"]] = r["outcome"] == "correct"
        for a, b_ in PAIRS:
            pairs = [v for v in by_key.values() if a in v and b_ in v]
            if not pairs:
                continue
            b = sum(v[a] and not v[b_] for v in pairs)
            c = sum(v[b_] and not v[a] for v in pairs)
            out.append(
                f"| {model} | {a} vs {b_} | {len(pairs)} | {b} | {c} "
                f"| {mcnemar_exact(b, c):.3g} |"
            )

    out.append("\n## Task-level sign test (repeats pooled per task)\n")
    out.append(
        "Repeats of one task are correlated, so the McNemar p-values above are "
        "optimistic. Here each task contributes one paired difference in its "
        "number of correct repeats; ties are dropped.\n"
    )
    out.append("| Model | A vs B | Tasks | A better | B better | Tied | Sign-test p |")
    out.append("|---|---|---|---|---|---|---|")
    for model in sorted({r["model"] for r in rows}):
        per_task: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in rows:
            if r["model"] == model:
                per_task[r["task"]][r["surface"]] += r["outcome"] == "correct"
        for a, b_ in PAIRS:
            diffs = [v[a] - v[b_] for v in per_task.values() if a in v and b_ in v]
            if not diffs:
                continue
            up = sum(d > 0 for d in diffs)
            down = sum(d < 0 for d in diffs)
            out.append(
                f"| {model} | {a} vs {b_} | {len(diffs)} | {up} | {down} "
                f"| {len(diffs) - up - down} | {mcnemar_exact(up, down):.3g} |"
            )

    out.append("\n## Cost drivers per run\n")
    out.append(
        "| Model | Surface | Tool calls | Failed calls | LLM calls | Prompt tok "
        "| Completion tok |"
    )
    out.append("|---|---|---|---|---|---|---|")
    for (model, surface), rs in sorted(groups.items()):
        def mean(key: str) -> float:
            vals = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
            return sum(vals) / len(vals) if vals else math.nan

        out.append(
            f"| {model} | {surface} | {mean('tool_calls'):.1f} | {mean('failed_tool_calls'):.2f} "
            f"| {mean('llm_calls'):.1f} | {mean('prompt_tokens'):,.0f} "
            f"| {mean('completion_tokens'):,.0f} |"
        )
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--md", type=Path, default=None, help="Also write the tables here")
    args = parser.parse_args()
    text = summarize(load(args.results))
    print(text)
    if args.md:
        args.md.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
