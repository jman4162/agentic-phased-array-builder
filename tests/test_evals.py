"""Tests for the eval-harness scoring functions (no LLM involved)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "run_evals", _REPO_ROOT / "evals" / "run_evals.py",
)
run_evals = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_evals)


def _make_bundle(tmp_path, audit, manifest):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "audit.json").write_text(json.dumps(audit))
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return run_dir


GOOD_AUDIT = [
    {
        "tool": "pattern_compute",
        "result_summary": "{'directivity_dbi': 21.5, 'sidelobe_level_db': -18.2, ...",
    },
    {
        "tool": "system_evaluate",
        "result_summary": "{'link_margin_db': 6.3, ...",
    },
]

GOOD_MANIFEST = {"status": "success", "usage": {"llm_calls": 3}}


class TestToolSequence:
    def test_exact_match(self):
        ok, _ = run_evals.check_tool_sequence(["a", "b"], ["a", "b"])
        assert ok

    def test_subset_with_extras(self):
        ok, _ = run_evals.check_tool_sequence(
            ["pattern_compute", "system_evaluate"],
            ["project_validate", "pattern_compute", "plot_quicklook", "system_evaluate"],
        )
        assert ok

    def test_out_of_order_fails(self):
        ok, detail = run_evals.check_tool_sequence(["b", "a"], ["a", "b"])
        assert not ok
        assert "missing or out of order" in detail

    def test_missing_fails(self):
        ok, _ = run_evals.check_tool_sequence(["a", "c"], ["a", "b"])
        assert not ok

    def test_empty_expected_passes(self):
        ok, _ = run_evals.check_tool_sequence([], ["anything"])
        assert ok


class TestExtractMetric:
    def test_finds_metric_in_summary(self):
        assert run_evals.extract_metric(GOOD_AUDIT, "directivity_dbi") == 21.5

    def test_latest_value_wins(self):
        audit = [
            {"tool": "t", "result_summary": "{'directivity_dbi': 10.0}"},
            {"tool": "t", "result_summary": "{'directivity_dbi': 22.0}"},
        ]
        assert run_evals.extract_metric(audit, "directivity_dbi") == 22.0

    def test_negative_values(self):
        assert run_evals.extract_metric(GOOD_AUDIT, "sidelobe_level_db") == -18.2

    def test_missing_returns_none(self):
        assert run_evals.extract_metric(GOOD_AUDIT, "no_such_metric") is None


class TestScoreRun:
    TASK = {
        "name": "demo",
        "expected_tools": ["pattern_compute", "system_evaluate"],
        "max_llm_calls": 5,
        "metrics": {"directivity_dbi": {"min": 20.0}},
    }

    def test_all_checks_pass(self, tmp_path):
        run_dir = _make_bundle(tmp_path, GOOD_AUDIT, GOOD_MANIFEST)
        result = run_evals.score_run(self.TASK, run_dir)
        assert result["passed"], result["checks"]

    def test_metric_below_min_fails(self, tmp_path):
        audit = [{"tool": "pattern_compute", "result_summary": "{'directivity_dbi': 9.0}"},
                 {"tool": "system_evaluate", "result_summary": "ok"}]
        run_dir = _make_bundle(tmp_path, audit, GOOD_MANIFEST)
        result = run_evals.score_run(self.TASK, run_dir)
        assert not result["passed"]
        assert not result["checks"]["metric:directivity_dbi"]["passed"]

    def test_bad_status_fails(self, tmp_path):
        run_dir = _make_bundle(
            tmp_path, GOOD_AUDIT, {"status": "max_turns", "usage": {"llm_calls": 3}},
        )
        result = run_evals.score_run(self.TASK, run_dir)
        assert not result["checks"]["status"]["passed"]

    def test_too_many_llm_calls_fails(self, tmp_path):
        run_dir = _make_bundle(
            tmp_path, GOOD_AUDIT, {"status": "success", "usage": {"llm_calls": 9}},
        )
        result = run_evals.score_run(self.TASK, run_dir)
        assert not result["checks"]["llm_calls"]["passed"]

    def test_missing_bundle_files(self, tmp_path):
        run_dir = tmp_path / "empty"
        run_dir.mkdir()
        result = run_evals.score_run(self.TASK, run_dir)
        assert not result["passed"]


class TestGoldenTaskFiles:
    def test_all_golden_tasks_load(self):
        files = sorted((_REPO_ROOT / "evals" / "golden").glob("*.yaml"))
        assert len(files) >= 3
        for f in files:
            task = run_evals.load_task(f)
            assert task["name"]
            assert task["prompt"]
            assert isinstance(task["expected_tools"], list)
