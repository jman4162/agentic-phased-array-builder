"""Integration test: verify MCP server has tools and they can be called."""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.integration


class TestMCPToolIntegration:
    def test_all_tools_listed(self):
        """Verify all expected tools are registered on the server."""
        from apab.mcp.server import get_mcp

        server = get_mcp()
        tools = server._tool_manager._tools

        expected_tools = [
            "edgefem_run_unit_cell",
            "edgefem_surface_impedance",
            "edgefem_export_touchstone",
            "pattern_compute",
            "pattern_plot_cuts",
            "pattern_plot_3d",
            "pattern_multi_beam",
            "pattern_null_steer",
            "system_evaluate",
            "system_trade_study",
            "project_init",
            "project_validate",
            "io_import_touchstone",
            "io_save_hdf5",
            "plot_quicklook",
            "emtool_list_adapters",
            "emtool_import_results",
        ]

        for tool_name in expected_tools:
            assert tool_name in tools, f"Tool '{tool_name}' not found"

    async def test_pattern_compute_via_server(self):
        """Call pattern_compute through the MCP tool dispatcher."""
        from apab.agent.tool_dispatch import ToolDispatcher

        dispatcher = ToolDispatcher()
        result_str = dispatcher.dispatch(
            "pattern_compute",
            {"nx": 4, "ny": 4, "dx_m": 0.005, "dy_m": 0.005, "freq_hz": 10e9},
        )
        result = json.loads(result_str)
        assert "directivity_dbi" in result
        assert result["directivity_dbi"] > 0

    async def test_system_evaluate_via_server(self):
        """Call system_evaluate through the MCP tool dispatcher."""
        from apab.agent.tool_dispatch import ToolDispatcher

        dispatcher = ToolDispatcher()
        result_str = dispatcher.dispatch(
            "system_evaluate",
            {
                "nx": 4,
                "ny": 4,
                "dx_m": 0.005,
                "dy_m": 0.005,
                "freq_hz": 10e9,
                "bandwidth_hz": 100e6,
                "range_m": 1000.0,
                "tx_power_w_per_elem": 0.1,
            },
        )
        result = json.loads(result_str)
        assert isinstance(result, dict)
        assert len(result) > 0
