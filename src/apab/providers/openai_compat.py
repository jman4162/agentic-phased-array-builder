"""OpenAI-compatible LLM provider for APAB.

Delegates to ``OpenAIProvider`` with a user-supplied ``base_url``, enabling
any OpenAI-compatible endpoint (vLLM, LM Studio, Together.ai, etc.).
"""

from __future__ import annotations

from typing import Any

from apab.providers.usage import ProviderUsage


class OpenAICompatibleProvider:
    """LLM provider for any OpenAI-compatible API endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "default",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        from apab.providers.openai import OpenAIProvider

        self._delegate = OpenAIProvider(
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "openai_compatible"

    @property
    def last_usage(self) -> ProviderUsage | None:
        return self._delegate.last_usage

    def supports_tool_calling(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat request via the OpenAI-compatible endpoint."""
        return self._delegate.chat(messages, tools, **kwargs)
