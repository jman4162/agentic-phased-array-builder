"""Anthropic LLM provider for APAB."""

from __future__ import annotations

import time
from typing import Any

from apab.providers.usage import ProviderUsage

# Approximate pricing per 1M tokens (as of early 2026).
_COST_PER_1M: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-haiku-3-5-20241022": (0.80, 4.00),
}


class AnthropicProvider:
    """LLM provider backed by the Anthropic Messages API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> None:
        import anthropic

        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key

        self._model = model
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(**client_kwargs)
        self._last_usage: ProviderUsage | None = None

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def last_usage(self) -> ProviderUsage | None:
        return self._last_usage

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
        """Send a chat request to Anthropic and return a normalised response."""
        # Anthropic requires system prompt as a separate parameter.
        system_text, api_messages = _extract_system(messages)

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": api_messages,
        }
        if system_text:
            call_kwargs["system"] = system_text
        if tools:
            call_kwargs["tools"] = _convert_tools(tools)

        t0 = time.monotonic()
        response = self._client.messages.create(**call_kwargs)
        latency = time.monotonic() - t0

        self._last_usage = _extract_usage(response, latency, self._model)
        return _normalise_response(response)


def _extract_system(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Separate the system message from conversation messages.

    Anthropic's API takes system as a top-level parameter, not in the
    messages list.
    """
    system_text = None
    api_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            system_text = msg.get("content", "")
        else:
            api_messages.append(msg)
    return system_text, api_messages


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert MCP-style tool schemas to Anthropic's tool format.

    Anthropic uses ``input_schema`` (snake_case) rather than
    ``inputSchema`` (camelCase).
    """
    converted = []
    for tool in tools:
        anthropic_tool = {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("inputSchema", tool.get("parameters", {})),
        }
        converted.append(anthropic_tool)
    return converted


def _normalise_response(response: Any) -> dict[str, Any]:
    """Normalise an Anthropic Message to the APAB standard format."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({
                "name": block.name,
                "arguments": block.input if isinstance(block.input, dict) else {},
            })

    return {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else None,
        "tool_calls": tool_calls if tool_calls else None,
    }


def _extract_usage(
    response: Any,
    latency: float,
    model: str,
) -> ProviderUsage:
    """Extract token usage and estimate cost from the response."""
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", 0) or 0
    completion_tokens = getattr(usage, "output_tokens", 0) or 0

    input_cost, output_cost = _COST_PER_1M.get(model, (3.00, 15.00))
    cost = (prompt_tokens * input_cost + completion_tokens * output_cost) / 1_000_000

    return ProviderUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_s=latency,
        cost_estimate_usd=cost,
    )
