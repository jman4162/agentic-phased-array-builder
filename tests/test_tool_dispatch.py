"""Tests for the tool dispatcher."""

from __future__ import annotations

import json

from apab.agent.tool_dispatch import ToolDispatcher


class TestGetToolSchemas:
    def test_returns_list(self):
        dispatcher = ToolDispatcher()
        schemas = dispatcher.get_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) > 0

    def test_schema_has_required_fields(self):
        dispatcher = ToolDispatcher()
        schemas = dispatcher.get_tool_schemas()
        for schema in schemas:
            assert "name" in schema
            assert "description" in schema


class TestDispatch:
    def test_dispatch_pattern_compute(self):
        dispatcher = ToolDispatcher()
        result_str = dispatcher.dispatch(
            "pattern_compute",
            {"nx": 4, "ny": 4, "dx_m": 0.005, "dy_m": 0.005, "freq_hz": 10e9},
        )
        result = json.loads(result_str)
        assert "directivity_dbi" in result

    def test_dispatch_unknown_tool(self):
        dispatcher = ToolDispatcher()
        result_str = dispatcher.dispatch("nonexistent_tool", {})
        result = json.loads(result_str)
        assert "error" in result

    def test_audit_log_populated(self):
        dispatcher = ToolDispatcher()
        dispatcher.dispatch(
            "pattern_compute",
            {"nx": 4, "ny": 4, "dx_m": 0.005, "dy_m": 0.005, "freq_hz": 10e9},
        )
        assert len(dispatcher.audit_log) == 1
        entry = dispatcher.audit_log[0]
        assert entry["tool"] == "pattern_compute"
        assert "timestamp" in entry

    def test_audit_log_on_error(self):
        dispatcher = ToolDispatcher()
        dispatcher.dispatch("nonexistent_tool", {})
        assert len(dispatcher.audit_log) == 1
        assert "error" in dispatcher.audit_log[0]["result_summary"]


class TestArgumentValidation:
    """The agent's dispatcher validates arguments like an MCP client call does."""

    BASE = {
        "nx": 8, "ny": 8, "dx_m": 0.0054, "dy_m": 0.0054, "freq_hz": 28e9,
        "bandwidth_hz": 100e6, "range_m": 500.0, "tx_power_w_per_elem": 0.1,
    }

    def test_out_of_bounds_rejected(self):
        result = json.loads(
            ToolDispatcher().dispatch("system_evaluate", {**self.BASE, "scan_angle_deg": 90})
        )
        assert "scan_angle_deg" in result["error"]

    def test_literal_choice_rejected(self):
        result = json.loads(
            ToolDispatcher().dispatch("system_evaluate", {**self.BASE, "scenario_type": "Radar"})
        )
        assert "scenario_type" in result["error"]

    def test_numeric_string_coerced(self):
        result = json.loads(
            ToolDispatcher().dispatch("system_evaluate", {**self.BASE, "freq_hz": "28e9"})
        )
        assert "error" not in result
        assert result["eirp_dbw"] > 0

    def test_integer_choice_sent_as_string(self):
        # LLMs often send Literal int choices as strings ("1"); accept digits
        radar = {
            **self.BASE, "freq_hz": 10e9, "bandwidth_hz": 1e6, "range_m": 20e3,
            "tx_power_w_per_elem": 5.0, "scenario_type": "radar",
        }
        ok = json.loads(ToolDispatcher().dispatch("system_evaluate", {**radar, "swerling": "1"}))
        assert ok.get("swerling") == 1, ok.get("error")
        bad = json.loads(ToolDispatcher().dispatch("system_evaluate", {**radar, "swerling": "7"}))
        assert "swerling" in bad["error"]

    def test_rejection_logs_one_line_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="apab.agent.tool_dispatch"):
            ToolDispatcher().dispatch("system_evaluate", {**self.BASE, "scan_angle_deg": 90})
        records = [r for r in caplog.records if r.name == "apab.agent.tool_dispatch"]
        assert len(records) == 1
        assert records[0].levelno == logging.WARNING
        assert records[0].exc_info is None
        assert "scan_angle_deg: Input should be less than 90" in records[0].getMessage()

    def test_other_errors_are_not_argument_rejections(self):
        from apab.agent.tool_dispatch import _argument_errors

        assert _argument_errors(ValueError("solver crashed")) is None
        wrapped = RuntimeError("boom")
        wrapped.__cause__ = KeyError("x")
        assert _argument_errors(wrapped) is None
