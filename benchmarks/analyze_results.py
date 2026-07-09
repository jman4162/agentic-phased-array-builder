#!/usr/bin/env python3
"""Analyze benchmark results and generate figures/tables for the paper.

Usage
-----
    python -m benchmarks.analyze_results benchmarks/results/benchmark_*.json \
        --output-dir papers/llm_mcp_paper/figures
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Consistent styling for paper figures.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

PROVIDER_COLORS = {
    "ollama": "#2196F3",
    "openai": "#10A37F",
    "anthropic": "#D97706",
    "gemini": "#4285F4",
}

PROVIDER_SHORT_NAMES = {
    "ollama/qwen2.5-coder:14b": "Ollama\nQwen-14B",
    "openai/gpt-4.1-mini": "OpenAI\nGPT-4.1-mini",
    "openai/gpt-4.1": "OpenAI\nGPT-4.1",
    "anthropic/claude-sonnet-4-20250514": "Anthropic\nSonnet 4",
    "gemini/gemini-2.5-pro": "Google\nGemini 2.5 Pro",
}


def load_results(paths: list[Path]) -> list[dict[str, Any]]:
    """Load and merge results from one or more benchmark JSON files."""
    all_runs: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text())
        all_runs.extend(data.get("runs", []))
    return all_runs


def group_by_provider(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group runs by provider/model key."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        key = f"{run['provider']}/{run['model']}"
        grouped[key].append(run)
    return grouped


# ── Figure generators ─────────────────────────────────────────────────

def fig_cost_vs_quality(
    grouped: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Fig 4: Cost vs. tool-call correctness scatter plot."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for key, runs in grouped.items():
        provider = runs[0]["provider"]
        color = PROVIDER_COLORS.get(provider, "#666666")
        short = PROVIDER_SHORT_NAMES.get(key, key)

        costs = [r["cost_estimate_usd"] for r in runs]
        correctness = [r["tool_call_correctness"] * 100 for r in runs]

        ax.scatter(
            costs, correctness,
            c=color, label=short.replace("\n", " "),
            s=40, alpha=0.8, edgecolors="white", linewidth=0.5,
        )

    ax.set_xlabel("Cost per run (USD)")
    ax.set_ylabel("Tool-call correctness (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_cost_vs_quality.pdf")
    fig.savefig(output_dir / "fig_cost_vs_quality.png")
    plt.close(fig)
    logger.info("Saved fig_cost_vs_quality")


def fig_token_usage(
    grouped: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Fig 5: Stacked bar chart of token usage by provider."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    keys = list(grouped.keys())
    x = np.arange(len(keys))
    width = 0.6

    prompt_means = []
    completion_means = []
    labels = []

    for key in keys:
        runs = grouped[key]
        prompt_means.append(np.mean([r["prompt_tokens"] for r in runs]))
        completion_means.append(np.mean([r["completion_tokens"] for r in runs]))
        labels.append(PROVIDER_SHORT_NAMES.get(key, key))

    prompt_arr = np.array(prompt_means)
    completion_arr = np.array(completion_means)

    ax.bar(x, prompt_arr / 1000, width, label="Input tokens", color="#2196F3", alpha=0.8)
    ax.bar(
        x, completion_arr / 1000, width, bottom=prompt_arr / 1000,
        label="Output tokens", color="#FF9800", alpha=0.8,
    )

    ax.set_ylabel("Tokens (thousands)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_token_usage.pdf")
    fig.savefig(output_dir / "fig_token_usage.png")
    plt.close(fig)
    logger.info("Saved fig_token_usage")


def fig_latency_comparison(
    grouped: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Supplementary: Box plot of latency by provider."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    data = []
    labels = []
    for key in grouped:
        runs = grouped[key]
        data.append([r["total_latency_s"] for r in runs])
        labels.append(PROVIDER_SHORT_NAMES.get(key, key))

    bp = ax.boxplot(data, patch_artist=True)
    colors = [
        PROVIDER_COLORS.get(grouped[k][0]["provider"], "#666666")
        for k in grouped
    ]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Total latency (s)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_latency.pdf")
    fig.savefig(output_dir / "fig_latency.png")
    plt.close(fig)
    logger.info("Saved fig_latency")


# ── Table generators ──────────────────────────────────────────────────

def table_main_results(
    grouped: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Table I: Main benchmark results as LaTeX tabular."""
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        (
            r"Provider & Turns & Correct & Seq.\ OK "
            r"& Latency (s) & Tokens & Cost (\$) \\"
        ),
        r"\midrule",
    ]

    for key, runs in grouped.items():
        n = len(runs)
        short = PROVIDER_SHORT_NAMES.get(key, key).replace("\n", " ")
        avg_turns = np.mean([r["total_turns"] for r in runs])
        avg_correct = np.mean([r["tool_call_correctness"] for r in runs]) * 100
        seq_ok = sum(1 for r in runs if r["sequence_correct"]) / n * 100
        avg_latency = np.mean([r["total_latency_s"] for r in runs])
        avg_tokens = np.mean(
            [r["prompt_tokens"] + r["completion_tokens"] for r in runs]
        )
        avg_cost = np.mean([r["cost_estimate_usd"] for r in runs])

        lines.append(
            f"{short} & {avg_turns:.1f} & {avg_correct:.0f}\\% "
            f"& {seq_ok:.0f}\\% & {avg_latency:.1f} "
            f"& {avg_tokens:.0f} & {avg_cost:.4f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    (output_dir / "table_main_results.tex").write_text("\n".join(lines))
    logger.info("Saved table_main_results.tex")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("result_files", nargs="+", type=Path, help="Benchmark JSON files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("papers/llm_mcp_paper/figures"),
        help="Output directory for figures and tables.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_results(args.result_files)
    grouped = group_by_provider(runs)

    logger.info("Loaded %d runs across %d providers", len(runs), len(grouped))

    fig_cost_vs_quality(grouped, args.output_dir)
    fig_token_usage(grouped, args.output_dir)
    fig_latency_comparison(grouped, args.output_dir)
    table_main_results(grouped, args.output_dir)

    print(f"\nGenerated {3} figures and {1} table in {args.output_dir}")


if __name__ == "__main__":
    main()
