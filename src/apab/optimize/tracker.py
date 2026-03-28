"""Results tracker for the optimization loop (TSV-based)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentResult:
    """A single experiment result."""

    experiment_id: int
    metrics: dict[str, float]
    status: str  # "baseline", "keep", "discard"
    description: str


class ResultsTracker:
    """Read/write optimization results to a TSV file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._results: list[ExperimentResult] = []
        self._best: ExperimentResult | None = None
        self._metric_keys: list[str] = []
        if path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                exp_id = int(row["experiment"])
                status = row["status"]
                desc = row["description"]
                metrics = {
                    k: float(v) for k, v in row.items()
                    if k not in ("experiment", "status", "description")
                }
                result = ExperimentResult(exp_id, metrics, status, desc)
                self._results.append(result)
                if not self._metric_keys:
                    self._metric_keys = list(metrics.keys())

        # Find current best (last "keep" or "baseline")
        for r in reversed(self._results):
            if r.status in ("keep", "baseline"):
                self._best = r
                break

    def record(
        self,
        metrics: dict[str, float],
        status: str,
        description: str,
    ) -> ExperimentResult:
        """Record a new experiment result."""
        exp_id = len(self._results) + 1
        result = ExperimentResult(exp_id, metrics, status, description)
        self._results.append(result)

        if not self._metric_keys:
            self._metric_keys = list(metrics.keys())

        if status in ("keep", "baseline"):
            self._best = result

        self._append_row(result)
        return result

    def _append_row(self, result: ExperimentResult) -> None:
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            if is_new:
                writer.writerow(
                    ["experiment"] + self._metric_keys
                    + ["status", "description"]
                )
            writer.writerow(
                [result.experiment_id]
                + [result.metrics.get(k, 0.0) for k in self._metric_keys]
                + [result.status, result.description]
            )

    @property
    def best(self) -> ExperimentResult | None:
        return self._best

    @property
    def results(self) -> list[ExperimentResult]:
        return list(self._results)

    def recent(self, n: int = 5) -> list[ExperimentResult]:
        """Return the last *n* results."""
        return self._results[-n:]

    def format_history(
        self, n: int = 5, metric: str = "eirp_dbw",
    ) -> str:
        """Format recent history as text for the agent prompt."""
        lines = []
        for r in self.recent(n):
            val = r.metrics.get(metric, 0.0)
            lines.append(
                f"#{r.experiment_id:03d}: {r.description} "
                f"→ {val:.1f} ({r.status.upper()})"
            )
        return "\n".join(reversed(lines))
