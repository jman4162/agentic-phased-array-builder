"""Tests for the autonomous optimization loop."""

from __future__ import annotations

from unittest.mock import MagicMock

from apab.optimize.protocol import Constraint, load_protocol
from apab.optimize.runner import (
    OptimizeRunner,
    _check_constraints,
    _extract_metrics_from_tool_results,
    _is_improvement,
)
from apab.optimize.tracker import ResultsTracker


class TestProtocol:
    def test_load_protocol(self, tmp_path):
        proto_file = tmp_path / "research.md"
        proto_file.write_text(
            "# Test Protocol\n\n"
            "## Objective\nMaximize EIRP under cost constraints.\n\n"
            "## Metric\nPrimary: eirp_dbw (maximize)\n"
            "Constraint: cost_usd < 10000, snr_db > 15\n\n"
            "## Strategy\nTry different tapers first.\n"
        )
        proto = load_protocol(proto_file)
        assert proto.metric == "eirp_dbw"
        assert proto.direction == "maximize"
        assert len(proto.constraints) == 2
        assert proto.constraints[0].metric == "cost_usd"
        assert proto.constraints[0].op == "<"
        assert proto.constraints[0].value == 10000
        assert proto.constraints[1].metric == "snr_db"
        assert proto.constraints[1].op == ">"
        assert proto.constraints[1].value == 15
        assert "tapers" in proto.strategy

    def test_load_protocol_minimal(self, tmp_path):
        proto_file = tmp_path / "minimal.md"
        proto_file.write_text("# Minimal\nJust some text.\n")
        proto = load_protocol(proto_file)
        assert proto.metric == "eirp_dbw"  # default
        assert proto.direction == "maximize"  # default
        assert proto.constraints == []


class TestConstraints:
    def test_passes_when_satisfied(self):
        metrics = {"cost_usd": 5000, "snr_db": 20}
        constraints = [
            Constraint("cost_usd", "<", 10000),
            Constraint("snr_db", ">", 15),
        ]
        ok, reason = _check_constraints(metrics, constraints)
        assert ok is True

    def test_fails_on_violation(self):
        metrics = {"cost_usd": 12000, "snr_db": 20}
        constraints = [Constraint("cost_usd", "<", 10000)]
        ok, reason = _check_constraints(metrics, constraints)
        assert ok is False
        assert "cost_usd" in reason

    def test_fails_on_missing_metric(self):
        metrics = {"snr_db": 20}
        constraints = [Constraint("cost_usd", "<", 10000)]
        ok, reason = _check_constraints(metrics, constraints)
        assert ok is False
        assert "missing" in reason

    def test_empty_constraints_pass(self):
        ok, _ = _check_constraints({"x": 1}, [])
        assert ok is True


class TestImprovement:
    def test_maximize_higher_is_better(self):
        assert _is_improvement(35.0, 30.0, "maximize") is True
        assert _is_improvement(25.0, 30.0, "maximize") is False

    def test_minimize_lower_is_better(self):
        assert _is_improvement(5.0, 10.0, "minimize") is True
        assert _is_improvement(15.0, 10.0, "minimize") is False


class TestExtractMetrics:
    def test_extracts_from_json_string(self):
        results = [
            {"tool": "system_evaluate", "result": '{"eirp_dbw": 30.1, "cost_usd": 6400}'}
        ]
        metrics = _extract_metrics_from_tool_results(results)
        assert metrics["eirp_dbw"] == 30.1
        assert metrics["cost_usd"] == 6400

    def test_skips_non_numeric(self):
        results = [
            {"tool": "test", "result": '{"status": "ok", "value": 42}'}
        ]
        metrics = _extract_metrics_from_tool_results(results)
        assert "value" in metrics
        assert "status" not in metrics

    def test_handles_invalid_json(self):
        results = [{"tool": "test", "result": "not json"}]
        metrics = _extract_metrics_from_tool_results(results)
        assert metrics == {}


class TestTracker:
    def test_record_and_retrieve(self, tmp_path):
        tracker = ResultsTracker(tmp_path / "results.tsv")

        tracker.record(
            {"eirp_dbw": 30.1, "cost_usd": 6400},
            "baseline",
            "8x8 uniform",
        )
        assert tracker.best is not None
        assert tracker.best.experiment_id == 1
        assert tracker.best.metrics["eirp_dbw"] == 30.1

        tracker.record(
            {"eirp_dbw": 32.4, "cost_usd": 6400},
            "keep",
            "8x8 taylor",
        )
        assert tracker.best.experiment_id == 2
        assert len(tracker.results) == 2

    def test_discard_does_not_update_best(self, tmp_path):
        tracker = ResultsTracker(tmp_path / "results.tsv")
        tracker.record({"eirp_dbw": 30.0}, "baseline", "base")
        tracker.record({"eirp_dbw": 28.0}, "discard", "worse")
        assert tracker.best.experiment_id == 1

    def test_format_history(self, tmp_path):
        tracker = ResultsTracker(tmp_path / "results.tsv")
        tracker.record({"eirp_dbw": 30.0}, "baseline", "base")
        tracker.record({"eirp_dbw": 32.0}, "keep", "taylor")
        history = tracker.format_history(n=5, metric="eirp_dbw")
        assert "base" in history
        assert "taylor" in history
        assert "KEEP" in history

    def test_persistence(self, tmp_path):
        path = tmp_path / "results.tsv"
        t1 = ResultsTracker(path)
        t1.record({"eirp_dbw": 30.0}, "baseline", "base")
        t1.record({"eirp_dbw": 35.0}, "keep", "improved")

        # Reload from disk
        t2 = ResultsTracker(path)
        assert len(t2.results) == 2
        assert t2.best.metrics["eirp_dbw"] == 35.0


class TestRunnerEvaluate:
    def test_baseline_when_no_best(self):
        proto = MagicMock()
        proto.constraints = []
        proto.metric = "eirp_dbw"
        proto.direction = "maximize"
        tracker = MagicMock()
        tracker.best = None

        runner = OptimizeRunner(MagicMock(), proto, tracker)
        status, _ = runner.evaluate({"eirp_dbw": 30.0})
        assert status == "baseline"

    def test_keep_on_improvement(self):
        proto = MagicMock()
        proto.constraints = []
        proto.metric = "eirp_dbw"
        proto.direction = "maximize"

        best = MagicMock()
        best.metrics = {"eirp_dbw": 30.0}
        tracker = MagicMock()
        tracker.best = best

        runner = OptimizeRunner(MagicMock(), proto, tracker)
        status, _ = runner.evaluate({"eirp_dbw": 35.0})
        assert status == "keep"

    def test_discard_on_regression(self):
        proto = MagicMock()
        proto.constraints = []
        proto.metric = "eirp_dbw"
        proto.direction = "maximize"

        best = MagicMock()
        best.metrics = {"eirp_dbw": 30.0}
        tracker = MagicMock()
        tracker.best = best

        runner = OptimizeRunner(MagicMock(), proto, tracker)
        status, _ = runner.evaluate({"eirp_dbw": 25.0})
        assert status == "discard"

    def test_discard_on_constraint_violation(self):
        proto = MagicMock()
        proto.constraints = [Constraint("cost_usd", "<", 10000)]
        proto.metric = "eirp_dbw"
        proto.direction = "maximize"

        best = MagicMock()
        best.metrics = {"eirp_dbw": 30.0}
        tracker = MagicMock()
        tracker.best = best

        runner = OptimizeRunner(MagicMock(), proto, tracker)
        status, reason = runner.evaluate({"eirp_dbw": 40.0, "cost_usd": 15000})
        assert status == "discard"
        assert "cost_usd" in reason
