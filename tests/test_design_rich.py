"""Tests for the rich-powered apab design command."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest


class TestDesignPreFlight:
    """Test that design command checks provider connectivity."""

    @patch("apab.core.config.load_config")
    def test_exits_on_provider_ping_failure(self, mock_load_config, tmp_path):
        """If provider.ping() fails, design should exit with code 1."""
        # Setup config
        mock_config = MagicMock()
        mock_config.project.name = "test"
        mock_config.project.workspace = str(tmp_path / "workspace")
        mock_config.llm.provider = "ollama"
        mock_config.llm.model = "test-model"
        mock_config.llm.base_url = "http://localhost:11434"
        mock_config.llm.redaction_mode = "none"
        mock_load_config.return_value = mock_config

        # Mock provider with failing ping
        mock_provider = MagicMock()
        mock_provider.ping.return_value = (False, "Cannot reach Ollama")

        with patch("apab.agent.orchestrator.AgentOrchestrator") as mock_orch_cls:
            mock_orch = MagicMock()
            mock_orch.provider = mock_provider
            mock_orch_cls.return_value = mock_orch

            # Need to re-import after patching
            from apab.commands.design import cmd_design

            args = argparse.Namespace(config=str(tmp_path / "apab.yaml"))
            with pytest.raises(SystemExit) as exc_info:
                cmd_design(args)
            assert exc_info.value.code == 1


class TestDesignTruncate:
    def test_short_text_unchanged(self):
        from apab.commands.design import _truncate

        assert _truncate("hello", 200) == "hello"

    def test_long_text_truncated(self):
        from apab.commands.design import _truncate

        result = _truncate("x" * 300, 200)
        assert len(result) == 203  # 200 + "..."
        assert result.endswith("...")
