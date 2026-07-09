"""Tests for the LLM provider registry."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from apab.agent.provider_registry import (
    discover_providers,
    get_provider,
)


class TestDiscoverProviders:
    def test_returns_dict(self):
        result = discover_providers()
        assert isinstance(result, dict)


class TestGetProvider:
    @patch("ollama.Client")
    def test_get_ollama(self, mock_client_cls):
        provider = get_provider("ollama")
        assert provider.name == "ollama"

    def test_get_openai(self):
        mock_openai = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            provider = get_provider("openai")
            assert provider.name == "openai"

    def test_get_anthropic(self):
        mock_anthropic = MagicMock()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            provider = get_provider("anthropic")
            assert provider.name == "anthropic"

    def test_get_gemini(self):
        mock_google = MagicMock()
        mock_genai = MagicMock()
        with patch.dict(sys.modules, {
            "google": mock_google,
            "google.generativeai": mock_genai,
        }):
            provider = get_provider("gemini")
            assert provider.name == "gemini"

    def test_get_openai_compatible(self):
        mock_openai = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            provider = get_provider("openai_compatible")
            assert provider.name == "openai_compatible"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_provider("nonexistent_provider")
