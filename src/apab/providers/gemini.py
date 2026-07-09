"""Google Gemini LLM provider for APAB."""

from __future__ import annotations

import time
from typing import Any

from apab.providers.usage import ProviderUsage

# Approximate pricing per 1M tokens (as of early 2026).
_COST_PER_1M: dict[str, tuple[float, float]] = {
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.0-flash": (0.10, 0.40),
}


class GeminiProvider:
    """LLM provider backed by the Google Gemini API."""

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        import google.generativeai as genai

        if api_key:
            genai.configure(api_key=api_key)

        self._genai = genai
        self._model_name = model
        self._model = genai.GenerativeModel(model)
        self._last_usage: ProviderUsage | None = None

    @property
    def name(self) -> str:
        return "gemini"

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
        """Send a chat request to Gemini and return a normalised response."""
        gemini_tools = [_convert_tools(tools)] if tools else None
        contents = _convert_messages(messages)

        t0 = time.monotonic()
        response = self._model.generate_content(
            contents,
            tools=gemini_tools,
        )
        latency = time.monotonic() - t0

        self._last_usage = _extract_usage(response, latency, self._model_name)
        return _normalise_response(response)


def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert APAB-standard messages to Gemini content format.

    Gemini uses ``"model"`` instead of ``"assistant"`` and merges system
    messages into the first user message.
    """
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        text = msg.get("content", "") or ""

        if role == "system":
            system_parts.append(text)
            continue

        gemini_role = "model" if role == "assistant" else "user"

        # Prepend system context to the first user message.
        if system_parts and gemini_role == "user":
            text = "\n\n".join(system_parts) + "\n\n" + text
            system_parts = []

        contents.append({
            "role": gemini_role,
            "parts": [{"text": text}],
        })

    return contents


def _convert_tools(tools: list[dict[str, Any]]) -> Any:
    """Convert MCP-style tool schemas to a Gemini Tool object.

    Returns a ``genai.types.Tool`` wrapping ``FunctionDeclaration`` objects.
    """
    import google.generativeai as genai

    declarations = []
    for tool in tools:
        schema = tool.get("inputSchema", tool.get("parameters", {}))
        # Strip unsupported keys for Gemini's OpenAPI subset.
        clean_schema = _clean_schema(schema)

        decl = genai.types.FunctionDeclaration(
            name=tool["name"],
            description=tool.get("description", ""),
            parameters=clean_schema if clean_schema.get("properties") else None,
        )
        declarations.append(decl)

    return genai.types.Tool(function_declarations=declarations)


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove JSON Schema keys unsupported by Gemini's OpenAPI subset."""
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key in ("additionalProperties", "$schema"):
            continue
        if isinstance(value, dict):
            cleaned[key] = _clean_schema(value)
        else:
            cleaned[key] = value
    return cleaned


def _normalise_response(response: Any) -> dict[str, Any]:
    """Normalise a Gemini GenerateContentResponse to APAB standard format."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    candidate = response.candidates[0]
    for part in candidate.content.parts:
        if hasattr(part, "function_call") and part.function_call.name:
            fc = part.function_call
            # Gemini returns arguments as a MapComposite; convert to dict.
            args = dict(fc.args) if fc.args else {}
            tool_calls.append({
                "name": fc.name,
                "arguments": args,
            })
        elif hasattr(part, "text") and part.text:
            text_parts.append(part.text)

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
    metadata = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(metadata, "prompt_token_count", 0) or 0
    completion_tokens = getattr(metadata, "candidates_token_count", 0) or 0

    input_cost, output_cost = _COST_PER_1M.get(model, (1.25, 10.00))
    cost = (prompt_tokens * input_cost + completion_tokens * output_cost) / 1_000_000

    return ProviderUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_s=latency,
        cost_estimate_usd=cost,
    )
