"""Core benchmark harness for multi-provider evaluation.

This module defines the data structures and harness for running a
standardised phased-array design task against multiple LLM providers
and collecting per-run metrics (tool-call correctness, latency, token
usage, cost).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Reference task ────────────────────────────────────────────────────

STANDARD_TASK = (
    "Design a 28 GHz 8×8 phased-array antenna for 5G NR fixed wireless. "
    "Use RO4003C substrate (εr=3.55, h=0.254mm, tanδ=0.0027). "
    "The unit-cell period is 5.0mm × 5.0mm (half-wavelength at 28 GHz). "
    "Run a unit-cell frequency sweep from 26–30 GHz with EdgeFEM using 11 "
    "frequency points (n_freq=11). Use element spacing dx=dy=0.005m. "
    "Then compute the array radiation pattern with Taylor taper, "
    "and evaluate the system link budget for 200m range with 400 MHz bandwidth. "
    "Report the directivity, EIRP, and link margin."
)

REFERENCE_TOOL_SEQUENCE = [
    "edgefem_run_unit_cell",
    "pattern_compute",
    "system_evaluate",
]


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class BenchmarkTask:
    """Definition of a benchmark task with reference outputs."""

    name: str
    prompt: str
    reference_tool_sequence: list[str]
    reference_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ToolCallRecord:
    """A single tool call captured during a benchmark run."""

    turn: int
    tool_name: str
    arguments: dict[str, Any]
    result_summary: str = ""
    latency_s: float = 0.0


@dataclass
class BenchmarkRun:
    """Results from a single provider run."""

    provider: str
    model: str
    run_index: int
    task_name: str
    completed: bool = False
    final_response: str = ""
    total_turns: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    total_latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_estimate_usd: float = 0.0
    tool_call_correctness: float = 0.0
    sequence_correct: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class BenchmarkSuite:
    """Aggregated results across all providers and runs."""

    task: BenchmarkTask
    runs: list[BenchmarkRun] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": asdict(self.task),
            "runs": [asdict(r) for r in self.runs],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))


# ── Scoring ───────────────────────────────────────────────────────────

def score_tool_sequence(
    actual: list[str],
    reference: list[str],
) -> tuple[float, bool]:
    """Score tool-call correctness and sequence order.

    Returns (correctness_fraction, sequence_matches).
    correctness_fraction is the fraction of reference tools that appear
    in the actual call list (order-independent).
    sequence_matches is True if the reference tools appear in the actual
    list in the correct relative order.
    """
    if not reference:
        return 1.0, True

    # Correctness: fraction of reference tools called.
    hits = sum(1 for t in reference if t in actual)
    correctness = hits / len(reference)

    # Sequence: check relative ordering.
    ref_positions = []
    for t in reference:
        if t in actual:
            ref_positions.append(actual.index(t))
        else:
            ref_positions.append(-1)

    valid = [p for p in ref_positions if p >= 0]
    sequence_ok = valid == sorted(valid) and len(valid) == len(reference)

    return correctness, sequence_ok


def score_parameters(
    actual_args: dict[str, Any],
    reference_args: dict[str, Any],
    tolerance: float = 0.10,
) -> float:
    """Score parameter quality against reference values.

    Numeric parameters are compared with a relative tolerance.
    String parameters require exact match.
    Returns fraction of parameters matching.
    """
    if not reference_args:
        return 1.0

    matches = 0
    for key, ref_val in reference_args.items():
        act_val = actual_args.get(key)
        if act_val is None:
            continue
        if isinstance(ref_val, (int, float)) and isinstance(act_val, (int, float)):
            if ref_val == 0:
                if act_val == 0:
                    matches += 1
            elif abs(act_val - ref_val) / abs(ref_val) <= tolerance:
                matches += 1
        elif str(act_val) == str(ref_val):
            matches += 1

    return matches / len(reference_args)


# ── Harness ───────────────────────────────────────────────────────────

class BenchmarkHarness:
    """Runs a benchmark task against a provider and collects metrics."""

    def __init__(
        self,
        task: BenchmarkTask | None = None,
        max_turns: int = 20,
    ) -> None:
        self.task = task or BenchmarkTask(
            name="standard_28ghz",
            prompt=STANDARD_TASK,
            reference_tool_sequence=REFERENCE_TOOL_SEQUENCE,
        )
        self.max_turns = max_turns

    def run_single(
        self,
        provider_name: str,
        model: str,
        run_index: int = 0,
        config_overrides: dict[str, Any] | None = None,
    ) -> BenchmarkRun:
        """Execute one benchmark run and return the results.

        This creates an AgentOrchestrator with the specified provider
        and runs the standard task through the agentic loop.
        """
        from apab.agent.orchestrator import AgentOrchestrator
        from apab.agent.provider_registry import get_provider
        from apab.core.schemas import ProjectConfig

        run = BenchmarkRun(
            provider=provider_name,
            model=model,
            run_index=run_index,
            task_name=self.task.name,
        )

        try:
            provider = get_provider(provider_name, model=model)

            # Minimal config for orchestrator.
            config_dict = {
                "project": {"name": f"benchmark_{run_index}", "workspace": "workspace"},
                "llm": {
                    "provider": provider_name,
                    "model": model,
                    "redaction_mode": "none",
                },
            }
            if config_overrides:
                config_dict.update(config_overrides)

            config = ProjectConfig.model_validate(config_dict)
            orchestrator = AgentOrchestrator(config=config, provider=provider)

            t0 = time.monotonic()
            final_response = orchestrator.run_to_completion(
                self.task.prompt,
                max_turns=self.max_turns,
            )
            run.total_latency_s = time.monotonic() - t0
            run.final_response = final_response
            run.completed = True

            # Extract tool calls from audit log.
            turn = 0
            for entry in orchestrator.dispatcher.audit_log:
                tc = ToolCallRecord(
                    turn=turn,
                    tool_name=entry.get("tool", ""),
                    arguments=entry.get("arguments", {}),
                    result_summary=str(entry.get("result_summary", ""))[:200],
                )
                run.tool_calls.append(tc)
                turn += 1

            run.total_turns = len(orchestrator.messages)

            # Extract usage if provider tracks it.
            if hasattr(provider, "last_usage") and provider.last_usage:
                usage = provider.last_usage
                run.prompt_tokens = usage.prompt_tokens
                run.completion_tokens = usage.completion_tokens
                run.cost_estimate_usd = usage.cost_estimate_usd

        except Exception as e:
            run.errors.append(str(e))
            logger.exception("Benchmark run failed: %s/%s #%d", provider_name, model, run_index)

        # Score.
        actual_tools = [tc.tool_name for tc in run.tool_calls]
        run.tool_call_correctness, run.sequence_correct = score_tool_sequence(
            actual_tools, self.task.reference_tool_sequence
        )

        return run

    def run_suite(
        self,
        providers: list[tuple[str, str]],
        n_runs: int = 5,
    ) -> BenchmarkSuite:
        """Run the benchmark across multiple providers.

        Parameters
        ----------
        providers:
            List of (provider_name, model) tuples.
        n_runs:
            Number of runs per provider.
        """
        suite = BenchmarkSuite(task=self.task)

        for provider_name, model in providers:
            for i in range(n_runs):
                logger.info(
                    "Running benchmark: %s/%s run %d/%d",
                    provider_name, model, i + 1, n_runs,
                )
                run = self.run_single(provider_name, model, run_index=i)
                suite.runs.append(run)

        return suite
