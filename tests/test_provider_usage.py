"""Tests for the shared ProviderUsage dataclass."""

from __future__ import annotations

import inspect

from apab.providers.usage import ProviderUsage


def test_single_shared_dataclass():
    """All provider modules must use the one ProviderUsage class."""
    from apab.providers import anthropic, gemini, openai

    assert anthropic.ProviderUsage is ProviderUsage
    assert gemini.ProviderUsage is ProviderUsage
    assert openai.ProviderUsage is ProviderUsage


def test_reexported_from_provider_registry():
    from apab.agent.provider_registry import ProviderUsage as RegistryUsage

    assert RegistryUsage is ProviderUsage


def test_defaults():
    usage = ProviderUsage()
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.latency_s == 0.0
    assert usage.cost_estimate_usd == 0.0


def test_all_builtin_providers_declare_last_usage():
    """Every built-in provider class exposes a last_usage property."""
    from apab.providers.anthropic import AnthropicProvider
    from apab.providers.gemini import GeminiProvider
    from apab.providers.ollama import OllamaProvider
    from apab.providers.openai import OpenAIProvider
    from apab.providers.openai_compat import OpenAICompatibleProvider

    for cls in (
        AnthropicProvider,
        GeminiProvider,
        OllamaProvider,
        OpenAIProvider,
        OpenAICompatibleProvider,
    ):
        attr = inspect.getattr_static(cls, "last_usage")
        assert isinstance(attr, property), cls.__name__
