"""OpenAI-compatible LLM provider stub for APAB (full implementation in v0.3)."""

from __future__ import annotations

from typing import Any


class OpenAICompatibleProvider:
    """Stub OpenAI-compatible provider — raises on use in v0.2."""

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "openai_compatible"

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
        raise NotImplementedError("OpenAI-compatible provider is a stub in v0.2")
