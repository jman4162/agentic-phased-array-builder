"""Tests for the APAB system prompt builder."""

from __future__ import annotations

from apab.agent.prompts import build_system_prompt


class TestBuildSystemPrompt:
    def test_basic_prompt_without_args(self):
        prompt = build_system_prompt()
        assert "phased-array antenna" in prompt
        assert "Available Tools" not in prompt

    def test_tool_names_listed(self):
        tools = ["pattern_compute", "system_evaluate", "edgefem_run_unit_cell"]
        prompt = build_system_prompt(tool_names=tools)
        assert "## Available Tools" in prompt
        assert "pattern_compute" in prompt
        assert "system_evaluate" in prompt
        assert "edgefem_run_unit_cell" in prompt

    def test_tools_grouped_by_prefix(self):
        tools = [
            "pattern_compute",
            "pattern_plot_cuts",
            "system_evaluate",
            "edgefem_run_unit_cell",
        ]
        prompt = build_system_prompt(tool_names=tools)
        assert "**Array patterns:**" in prompt
        assert "**System analysis:**" in prompt
        assert "**Unit-cell (EdgeFEM):**" in prompt

    def test_no_contradictory_json_instruction(self):
        prompt = build_system_prompt()
        assert "Do NOT\n  write tool calls as JSON" not in prompt
        assert "Do NOT write tool calls as JSON" not in prompt

    def test_json_fallback_acknowledged(self):
        prompt = build_system_prompt()
        assert "system will attempt to parse it" in prompt

    def test_config_context_included(self):
        config = {
            "project": {"name": "my_array"},
            "array": {
                "size": [8, 8],
                "spacing_m": [0.005, 0.005],
                "taper": "taylor",
            },
        }
        prompt = build_system_prompt(config=config)
        assert "my_array" in prompt
        assert "8×8" in prompt
        assert "taylor" in prompt

    def test_empty_tool_names_no_section(self):
        prompt = build_system_prompt(tool_names=[])
        assert "Available Tools" not in prompt
