"""OpenAI LLM provider for APAB."""

from __future__ import annotations

import json
import time
from typing import Any

from apab.providers.usage import ProviderUsage

# Approximate pricing per 1M tokens (as of early 2026).
_COST_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


class OpenAIProvider:
    """LLM provider backed by the OpenAI API."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        import openai

        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self._model = model
        self._client = openai.OpenAI(**client_kwargs)
        self._last_usage: ProviderUsage | None = None

    @property
    def name(self) -> str:
        return "openai"

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
        """Send a chat request to OpenAI and return a normalised response."""
        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if tools:
            call_kwargs["tools"] = _convert_tools(tools)

        t0 = time.monotonic()
        response = self._client.chat.completions.create(**call_kwargs)
        latency = time.monotonic() - t0

        self._last_usage = _extract_usage(response, latency, self._model)
        return _normalise_response(response)


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert MCP-style tool schemas to OpenAI's function-calling format.

    MCP tools use ``inputSchema``; OpenAI expects ``parameters`` under
    ``function`` inside a wrapper with ``type: "function"``.
    """
    converted = []
    for tool in tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema", tool.get("parameters", {})),
            },
        }
        converted.append(openai_tool)
    return converted


def _normalise_response(response: Any) -> dict[str, Any]:
    """Normalise an OpenAI ChatCompletion to the APAB standard format."""
    choice = response.choices[0]
    msg = choice.message

    tool_calls = None
    if msg.tool_calls:
        tool_calls = []
        for tc in msg.tool_calls:
            arguments = tc.function.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
            tool_calls.append({
                "name": tc.function.name,
                "arguments": arguments if isinstance(arguments, dict) else {},
            })

    return {
        "role": msg.role,
        "content": msg.content,
        "tool_calls": tool_calls,
    }


def _extract_usage(
    response: Any,
    latency: float,
    model: str,
) -> ProviderUsage:
    """Extract token usage and estimate cost from the response."""
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0

    input_cost, output_cost = _COST_PER_1M.get(model, (2.00, 8.00))
    cost = (prompt_tokens * input_cost + completion_tokens * output_cost) / 1_000_000

    return ProviderUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_s=latency,
        cost_estimate_usd=cost,
    )
