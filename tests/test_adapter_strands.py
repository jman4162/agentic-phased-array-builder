"""Tests for the Strands adapter (no strands install required)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from apab.adapters.strands import (
    _require_strands,
    apab_server_parameters,
    apab_system_prompt,
)


class TestRequireStrands:
    def test_raises_helpful_error_when_missing(self):
        with patch.dict(sys.modules, {"strands": None}):
            with pytest.raises(ImportError, match=r"apab\[strands\]"):
                _require_strands()

    def test_passes_when_present(self):
        with patch.dict(sys.modules, {"strands": MagicMock()}):
            _require_strands()  # must not raise


class TestServerParameters:
    def test_launches_apab_mcp_serve_over_stdio(self):
        params = apab_server_parameters()
        assert params.command == sys.executable
        assert params.args == [
            "-m", "apab.cli", "mcp", "serve", "--transport", "stdio",
        ]

    def test_config_path_forwarded(self, tmp_path):
        cfg = tmp_path / "apab.yaml"
        params = apab_server_parameters(config_path=cfg)
        assert params.args[-2:] == ["--config", str(cfg)]


class TestMCPClient:
    def test_constructs_strands_mcp_client(self):
        mock_strands = MagicMock()
        mock_tools_mcp = MagicMock()
        with patch.dict(sys.modules, {
            "strands": mock_strands,
            "strands.tools": MagicMock(),
            "strands.tools.mcp": mock_tools_mcp,
        }):
            from apab.adapters.strands import apab_mcp_client

            client = apab_mcp_client()

        assert client is mock_tools_mcp.MCPClient.return_value
        # The transport factory is a callable handed to MCPClient
        (factory,), _ = mock_tools_mcp.MCPClient.call_args
        assert callable(factory)


class TestSystemPrompt:
    def test_reuses_apab_prompt(self):
        prompt = apab_system_prompt()
        assert "phased-array" in prompt.lower() or "antenna" in prompt.lower()

    def test_includes_project_context(self):
        prompt = apab_system_prompt({"project": {"name": "my_array"}})
        assert "my_array" in prompt
