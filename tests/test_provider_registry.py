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


class TestProviderFromSpec:
    def _spec(self, **kw):
        from apab.core.schemas import LLMSpec

        return LLMSpec(provider="openai_compatible", model="m", base_url="http://x/v1", **kw)

    def test_key_read_from_named_env_var(self, monkeypatch):
        from apab.agent.provider_registry import provider_from_spec

        monkeypatch.setenv("MY_ROUTER_KEY", "sk-test-123")
        with patch("apab.agent.provider_registry.get_provider") as gp:
            provider_from_spec(self._spec(api_key_env="MY_ROUTER_KEY"))
        assert gp.call_args.kwargs["api_key"] == "sk-test-123"
        assert gp.call_args.kwargs["base_url"] == "http://x/v1"

    def test_unset_named_env_var_is_an_error(self, monkeypatch):
        from apab.agent.provider_registry import provider_from_spec

        monkeypatch.delenv("MY_ROUTER_KEY", raising=False)
        with pytest.raises(ValueError, match="MY_ROUTER_KEY"):
            provider_from_spec(self._spec(api_key_env="MY_ROUTER_KEY"))

    def test_no_api_key_env_passes_no_key(self):
        from apab.agent.provider_registry import provider_from_spec

        with patch("apab.agent.provider_registry.get_provider") as gp:
            provider_from_spec(self._spec())
        assert "api_key" not in gp.call_args.kwargs

    def test_orchestrator_uses_api_key_env(self, monkeypatch, tmp_path):
        from apab.agent.orchestrator import AgentOrchestrator
        from apab.core.schemas import ProjectConfig, ProjectMeta

        monkeypatch.setenv("MY_ROUTER_KEY", "sk-test-456")
        config = ProjectConfig(
            project=ProjectMeta(name="t", workspace=str(tmp_path)),
            llm=self._spec(api_key_env="MY_ROUTER_KEY"),
        )
        with patch("apab.agent.provider_registry.get_provider") as gp:
            AgentOrchestrator(config)
        assert gp.call_args.kwargs["api_key"] == "sk-test-456"

    def test_ollama_sends_key_as_bearer_header(self):
        with patch("ollama.Client") as client:
            from apab.providers.ollama import OllamaProvider

            OllamaProvider(model="m", api_key="sk-ollama")
        assert client.call_args.kwargs["headers"] == {"Authorization": "Bearer sk-ollama"}
