"""Tests for the MCP server factory and tool registration."""

from __future__ import annotations

from apab.mcp.server import create_server, get_mcp


class TestGetMCP:
    def test_returns_fastmcp_instance(self):
        from mcp.server.fastmcp import FastMCP

        server = get_mcp()
        assert isinstance(server, FastMCP)

    def test_singleton(self):
        s1 = get_mcp()
        s2 = get_mcp()
        assert s1 is s2

    def test_server_name(self):
        server = get_mcp()
        assert server.name == "apab"


class TestCreateServer:
    def test_returns_fastmcp(self):
        from mcp.server.fastmcp import FastMCP

        server = create_server()
        assert isinstance(server, FastMCP)

    def test_stores_config(self, sample_config_dict):
        from apab.core.schemas import ProjectConfig

        config = ProjectConfig(**sample_config_dict)
        server = create_server(config=config)
        assert server._apab_config is config  # type: ignore[attr-defined]


class TestToolRegistration:
    def test_tools_registered(self):
        """At least the core tools should be registered."""
        server = get_mcp()
        tools = server._tool_manager._tools
        assert len(tools) > 0

    def test_expected_tool_names(self):
        server = get_mcp()
        tools = server._tool_manager._tools
        tool_names = set(tools.keys())

        expected = {
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
        }
        assert expected.issubset(tool_names), f"Missing: {expected - tool_names}"

    def test_tool_count_at_least_17(self):
        server = get_mcp()
        tools = server._tool_manager._tools
        assert len(tools) >= 17
