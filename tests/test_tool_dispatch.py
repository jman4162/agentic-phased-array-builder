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
