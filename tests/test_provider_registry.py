"""Tests for the LLM provider registry."""

from __future__ import annotations

from unittest.mock import patch

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

    def test_get_openai_stub(self):
        provider = get_provider("openai")
        assert provider.name == "openai"
        with pytest.raises(NotImplementedError):
            provider.chat([])

    def test_get_anthropic_stub(self):
        provider = get_provider("anthropic")
        assert provider.name == "anthropic"

    def test_get_gemini_stub(self):
        provider = get_provider("gemini")
        assert provider.name == "gemini"

    def test_get_openai_compatible_stub(self):
        provider = get_provider("openai_compatible")
        assert provider.name == "openai_compatible"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_provider("nonexistent_provider")
