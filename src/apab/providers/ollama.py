"""Ollama LLM provider for APAB."""

from __future__ import annotations

from typing import Any


class OllamaProvider:
    """LLM provider backed by a local Ollama instance."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:14b",
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ) -> None:
        import ollama

        self._model = model
        self._client = ollama.Client(host=base_url)

    @property
    def name(self) -> str:
        return "ollama"

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
        """Send a chat request to Ollama and return a normalised response."""
        ollama_tools = _convert_tools(tools) if tools else None

        response = self._client.chat(
            model=self._model,
            messages=messages,
            tools=ollama_tools,
        )

        return _normalise_response(response)


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert MCP-style tool schemas to Ollama's tool format.

    MCP tools use ``inputSchema``; Ollama expects ``parameters`` under
    ``function``.
    """
    converted = []
    for tool in tools:
        ollama_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema", tool.get("parameters", {})),
            },
        }
        converted.append(ollama_tool)
    return converted


def _normalise_response(response: Any) -> dict[str, Any]:
    """Normalise an Ollama ChatResponse to the APAB standard format."""
    msg = response.message

    tool_calls = None
    if msg.tool_calls:
        tool_calls = []
        for tc in msg.tool_calls:
            func = tc.function
            tool_calls.append({
                "name": func.name,
                "arguments": func.arguments if isinstance(func.arguments, dict) else {},
            })

    return {
        "role": msg.role,
        "content": msg.content,
        "tool_calls": tool_calls,
    }
