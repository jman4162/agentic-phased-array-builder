#!/usr/bin/env python3
"""CLI runner for the APAB provider benchmark.

Usage
-----
    # Run with local Ollama only (free, no API keys required):
    python -m benchmarks.run_benchmark --providers ollama

    # Run all providers (requires API keys in environment):
    python -m benchmarks.run_benchmark --providers all

    # Quick test with 1 run per provider:
    python -m benchmarks.run_benchmark --providers ollama --n-runs 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from benchmarks.provider_benchmark import BenchmarkHarness

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Provider configurations: (provider_name, model)
PROVIDER_CONFIGS = {
    "ollama": ("ollama", "qwen2.5-coder:14b"),
    "openai-mini": ("openai", "gpt-4.1-mini"),
    "openai": ("openai", "gpt-4.1"),
    "anthropic": ("anthropic", "claude-sonnet-4-20250514"),
    "gemini": ("gemini", "gemini-2.5-pro"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run APAB provider benchmark")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["ollama"],
        help=(
            "Providers to benchmark. Use provider keys "
            f"({', '.join(PROVIDER_CONFIGS)}) or 'all'."
        ),
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of runs per provider (default: 5).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum LLM turns per run (default: 20).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Directory for result files.",
    )
    args = parser.parse_args()

    # Resolve providers.
    if "all" in args.providers:
        selected = list(PROVIDER_CONFIGS.values())
    else:
        selected = []
        for key in args.providers:
            if key not in PROVIDER_CONFIGS:
                logger.error("Unknown provider key: %s", key)
                sys.exit(1)
            selected.append(PROVIDER_CONFIGS[key])

    logger.info(
        "Benchmark config: providers=%s, n_runs=%d, max_turns=%d",
        [f"{p}/{m}" for p, m in selected],
        args.n_runs,
        args.max_turns,
    )

    harness = BenchmarkHarness(max_turns=args.max_turns)
    suite = harness.run_suite(selected, n_runs=args.n_runs)

    # Save results.
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"benchmark_{timestamp}.json"
    suite.save(result_path)
    logger.info("Results saved to %s", result_path)

    # Print summary.
    _print_summary(suite)


def _print_summary(suite) -> None:
    """Print a human-readable summary table."""
    from collections import defaultdict

    by_provider: dict[str, list] = defaultdict(list)
    for run in suite.runs:
        key = f"{run.provider}/{run.model}"
        by_provider[key].append(run)

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(
        f"{'Provider':<30} {'Runs':>4} {'Complete':>8} "
        f"{'Correct':>8} {'Turns':>8} {'Latency':>10} {'Cost':>10}"
    )
    print("-" * 80)

    for key, runs in by_provider.items():
        n = len(runs)
        completed = sum(1 for r in runs if r.completed)
        avg_correct = sum(r.tool_call_correctness for r in runs) / n
        avg_turns = sum(r.total_turns for r in runs) / n
        avg_latency = sum(r.total_latency_s for r in runs) / n
        total_cost = sum(r.cost_estimate_usd for r in runs)

        print(
            f"{key:<30} {n:>4} {completed:>8} "
            f"{avg_correct:>7.1%} {avg_turns:>8.1f} "
            f"{avg_latency:>9.1f}s ${total_cost:>8.4f}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
