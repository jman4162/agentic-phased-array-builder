"""Autonomous optimization loop for phased-array design."""

from __future__ import annotations

import json
import logging
from typing import Any

from apab.optimize.protocol import Constraint, ResearchProtocol
from apab.optimize.tracker import ResultsTracker

logger = logging.getLogger(__name__)


def _check_constraints(
    metrics: dict[str, float],
    constraints: list[Constraint],
) -> tuple[bool, str]:
    """Check if metrics satisfy all constraints.

    Returns ``(ok, reason)`` where *reason* describes the first
    violation, or an empty string if all pass.
    """
    ops = {
        "<": lambda a, b: a < b,
        ">": lambda a, b: a > b,
        "<=": lambda a, b: a <= b,
        ">=": lambda a, b: a >= b,
    }
    for c in constraints:
        val = metrics.get(c.metric)
        if val is None:
            return False, f"missing metric '{c.metric}'"
        if not ops[c.op](val, c.value):
            return False, (
                f"{c.metric}={val:.2f} violates "
                f"{c.metric} {c.op} {c.value}"
            )
    return True, ""


def _is_improvement(
    new_val: float,
    best_val: float,
    direction: str,
) -> bool:
    if direction == "maximize":
        return new_val > best_val
    return new_val < best_val


def _extract_metrics_from_tool_results(
    tool_results: list[dict[str, Any]],
) -> dict[str, float]:
    """Pull numeric metrics out of tool result JSON strings."""
    metrics: dict[str, float] = {}
    for r in tool_results:
        try:
            data = json.loads(r["result"]) if isinstance(r["result"], str) else r["result"]
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, float)) and k != "status":
                    metrics[k] = float(v)
    return metrics


class OptimizeRunner:
    """Drives the autonomous optimization loop.

    Parameters
    ----------
    orchestrator:
        A configured :class:`AgentOrchestrator`.
    protocol:
        Parsed research protocol defining objectives and constraints.
    tracker:
        Results tracker for recording experiments.
    max_experiments:
        Maximum number of experiments before stopping.
    """

    def __init__(
        self,
        orchestrator: Any,
        protocol: ResearchProtocol,
        tracker: ResultsTracker,
        max_experiments: int = 50,
    ) -> None:
        self.orch = orchestrator
        self.protocol = protocol
        self.tracker = tracker
        self.max_experiments = max_experiments

    def build_prompt(self, is_baseline: bool = False) -> str:
        """Build the agent prompt for the current experiment."""
        from apab.agent.prompts import build_optimize_prompt

        return build_optimize_prompt(
            protocol=self.protocol,
            tracker=self.tracker,
            is_baseline=is_baseline,
        )

    def run_experiment(self, prompt: str) -> tuple[dict[str, float], str]:
        """Run a single experiment: agent proposes + tool executes.

        Returns ``(metrics, description)`` where *description* is the
        agent's reasoning for the design change.
        """
        self.orch.start_session(prompt)

        description = ""
        all_tool_results: list[dict[str, Any]] = []

        for _turn in range(10):
            response = self.orch.step()
            tool_calls = response.get("tool_calls")

            if not tool_calls:
                description = response.get("content") or ""
                break

            results = self.orch.execute_tool_calls(response)
            all_tool_results.extend(results)

        metrics = _extract_metrics_from_tool_results(all_tool_results)

        # Truncate description to first line for the TSV
        first_line = description.split("\n")[0][:120] if description else ""
        return metrics, first_line

    def evaluate(
        self, metrics: dict[str, float],
    ) -> tuple[str, str]:
        """Evaluate metrics against protocol.

        Returns ``(status, reason)`` where *status* is one of
        ``"keep"``, ``"discard"``, or ``"baseline"``.
        """
        # Check constraints first
        ok, reason = _check_constraints(
            metrics, self.protocol.constraints,
        )
        if not ok:
            return "discard", reason

        best = self.tracker.best
        if best is None:
            return "baseline", ""

        metric_key = self.protocol.metric
        new_val = metrics.get(metric_key)
        best_val = best.metrics.get(metric_key)

        if new_val is None:
            return "discard", f"metric '{metric_key}' not in results"
        if best_val is None:
            return "keep", "no previous best for metric"

        if _is_improvement(new_val, best_val, self.protocol.direction):
            diff = new_val - best_val
            return "keep", f"{diff:+.2f}"
        return "discard", "no improvement"
