"""Tests for the tool-surface ablation scorer (no LLM involved)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "ablation_score", _REPO_ROOT / "evals" / "ablation" / "score.py",
)
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)

DB_TASK = {"answer": {"metric": "eirp_dbw", "unit": "dBW", "tol": 0.05}}
NONE_TASK = {**DB_TASK, "expect": "none"}


class TestParseAnswer:
    def test_last_answer_wins(self):
        assert score.parse_answer("ANSWER: 1.0\nmore\nANSWER: 28.66")[:2] == ("number", 28.66)

    def test_none(self):
        assert score.parse_answer("can't\nANSWER: NONE")[0] == "none"

    def test_missing(self):
        assert score.parse_answer("the EIRP is 28.7 dBW")[0] == "missing"

    def test_markdown_bold_and_case(self):
        assert score.parse_answer("**answer:** -6.59")[:2] == ("number", -6.59)

    def test_decimals_and_exponent(self):
        assert score.parse_answer("ANSWER: 28.7")[2] == 1
        assert score.parse_answer("ANSWER: 1.05e-6")[2] == 8


class TestClassify:
    REF, IGN = 28.659, 30.164

    def test_correct_within_tol(self):
        assert score.classify(DB_TASK, "ANSWER: 28.66", self.REF, self.IGN)["outcome"] == "correct"

    def test_coarse_rounding_allowed(self):
        # 28.7 is 0.041 off; tol 0.05 plus half of the last reported digit (0.05)
        assert score.classify(DB_TASK, "ANSWER: 28.7", self.REF, self.IGN)["outcome"] == "correct"

    def test_silently_wrong_matches_ignored(self):
        out = score.classify(DB_TASK, "ANSWER: 30.16", self.REF, self.IGN)
        assert out["outcome"] == "silently_wrong"

    def test_wrong(self):
        assert score.classify(DB_TASK, "ANSWER: 12", self.REF, self.IGN)["outcome"] == "wrong"

    def test_refused_when_answer_exists(self):
        assert score.classify(DB_TASK, "ANSWER: NONE", self.REF, self.IGN)["outcome"] == "refused"

    def test_accept_none(self):
        task = {**DB_TASK, "accept_none": True}
        assert score.classify(task, "ANSWER: NONE", 0, 10)["outcome"] == "correct"

    def test_expect_none_refusal_is_correct(self):
        assert score.classify(NONE_TASK, "ANSWER: NONE", None, 10.69)["outcome"] == "correct"

    def test_expect_none_number_matching_ignored_is_silently_wrong(self):
        out = score.classify(NONE_TASK, "ANSWER: 10.69", None, 10.69)
        assert out["outcome"] == "silently_wrong"

    def test_expect_none_other_number_is_wrong(self):
        assert score.classify(NONE_TASK, "ANSWER: 3", None, None)["outcome"] == "wrong"

    def test_no_answer(self):
        assert score.classify(DB_TASK, "", self.REF, self.IGN)["outcome"] == "no_answer"


class TestToolCallStats:
    def test_counts_error_results(self):
        audit = [
            {"tool": "system_evaluate", "result_summary": "{'eirp_dbw': 30.1}"},
            {"tool": "system_evaluate", "result_summary": "{'error': 'bad', 'tool': 'x'}"},
            {"tool": "system_evaluate", "result_summary": "{'error': 'x', 'status': 'failed'}"},
        ]
        assert score.tool_call_stats(audit) == {"tool_calls": 3, "failed_tool_calls": 2}
