"""Ollama LLM provider for APAB."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from apab.providers.usage import ProviderUsage

logger = logging.getLogger(__name__)


class OllamaConnectionError(ConnectionError):
    """Raised when the Ollama server is unreachable."""


class OllamaProvider:
    """LLM provider backed by a local Ollama instance."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:14b",
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        import httpx
        import ollama

        self._model = model
        self._base_url = base_url
        # A key (llm.api_key_env) authenticates to a remote Ollama server.
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = ollama.Client(
            host=base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers=headers,
        )
        self._last_usage: ProviderUsage | None = None

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def last_usage(self) -> ProviderUsage | None:
        return self._last_usage

    def supports_tool_calling(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def ping(self) -> tuple[bool, str]:
        """Check if Ollama is reachable and the model is available.

        Returns ``(ok, message)`` where *ok* is ``True`` when the server
        responds and the configured model is listed.
        """
        import httpx

        try:
            resp = httpx.get(
                f"{self._base_url}/api/tags",
                timeout=httpx.Timeout(3.0),
            )
            resp.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException, OSError) as exc:
            return False, f"Cannot reach Ollama at {self._base_url}: {exc}"

        models = [m.get("name", "") for m in resp.json().get("models", [])]
        # Match with or without :latest tag
        found = any(
            m == self._model or m.startswith(f"{self._model}:")
            or self._model.endswith(":latest") and m == self._model.removesuffix(":latest")
            for m in models
        )
        if not found:
            available = ", ".join(models[:10]) or "(none)"
            return False, (
                f"Model '{self._model}' not found. "
                f"Available: {available}. "
                f"Pull it with: ollama pull {self._model}"
            )

        return True, f"Ollama OK — model '{self._model}' available"

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat request to Ollama and return a normalised response."""
        import httpx

        ollama_tools = _convert_tools(tools) if tools else None
        ollama_messages = _prepare_messages(messages)

        t0 = time.monotonic()
        try:
            response = self._client.chat(
                model=self._model,
                messages=ollama_messages,
                tools=ollama_tools,
            )
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self._base_url}. "
                f"Is 'ollama serve' running? ({exc})"
            ) from exc
        except (httpx.ReadTimeout, httpx.TimeoutException) as exc:
            raise OllamaConnectionError(
                f"Ollama response timed out. The model may need "
                f"more time for complex requests. Try increasing "
                f"the timeout or using a smaller model. ({exc})"
            ) from exc
        except ConnectionError as exc:
            raise OllamaConnectionError(
                f"Connection to Ollama lost at {self._base_url}. "
                f"({exc})"
            ) from exc

        # Local inference: token counts from the response, no cost.
        self._last_usage = ProviderUsage(
            prompt_tokens=getattr(response, "prompt_eval_count", 0) or 0,
            completion_tokens=getattr(response, "eval_count", 0) or 0,
            latency_s=time.monotonic() - t0,
            cost_estimate_usd=0.0,
        )
        return _normalise_response(response)


def _prepare_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-wrap normalised tool calls to Ollama's nested format.

    The orchestrator stores tool calls as ``{"name": ..., "arguments": ...}``
    but Ollama's client validates that they are wrapped as
    ``{"function": {"name": ..., "arguments": ...}}``.
    """
    prepared = []
    for msg in messages:
        tc = msg.get("tool_calls")
        if tc and isinstance(tc, list):
            wrapped = []
            for call in tc:
                if "function" in call:
                    wrapped.append(call)
                else:
                    wrapped.append({
                        "function": {
                            "name": call["name"],
                            "arguments": call.get("arguments", {}),
                        }
                    })
            prepared.append({**msg, "tool_calls": wrapped})
        else:
            prepared.append(msg)
    return prepared


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
    """Normalise an Ollama ChatResponse to the APAB standard format.

    Some models (e.g. Qwen 2.5 Coder) emit tool calls as JSON in the
    text content rather than through the structured ``tool_calls``
    field.  When no structured tool calls are present we attempt to
    parse them from the content.
    """
    msg = response.message

    tool_calls = None
    if msg.tool_calls:
        tool_calls = []
        for tc in msg.tool_calls:
            func = tc.function
            tool_calls.append({
                "name": func.name,
                "arguments": _coerce_arguments(func.arguments),
            })

    # Fallback: parse tool calls written as JSON in the text content.
    if not tool_calls and msg.content:
        parsed = _parse_tool_calls_from_text(msg.content)
        if parsed:
            tool_calls = parsed

    content = msg.content
    # If we extracted tool calls from text, clear the raw JSON from content.
    if tool_calls and content:
        content = _strip_json_blocks(content).strip() or None

    # Reasoning models (e.g. qwen3.5) sometimes write the whole final reply
    # into the separate thinking channel and leave content empty, which would
    # end the agent loop with an empty answer. Fall back to the thinking text.
    thinking = getattr(msg, "thinking", None)
    if not tool_calls and not content and isinstance(thinking, str) and thinking.strip():
        logger.info("Ollama reply had empty content; using the thinking text")
        content = thinking.strip()

    return {
        "role": msg.role,
        "content": content,
        "tool_calls": tool_calls,
    }


# Regex matching fenced ```json ... ``` blocks.
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _parse_tool_calls_from_text(text: str) -> list[dict[str, Any]] | None:
    """Try to extract tool-call dicts from text content.

    Looks for JSON objects with a ``"name"`` key and the arguments under
    ``"arguments"`` or, as Llama 3.x writes them, ``"parameters"``: in
    fenced code blocks, as the whole reply, or embedded in prose.
    """
    candidates: list[str] = []

    # First try fenced code blocks.
    for m in _JSON_BLOCK_RE.finditer(text):
        candidates.append(m.group(1).strip())

    # If no fenced blocks, try the whole text as JSON.
    if not candidates:
        stripped = text.strip()
        if stripped.startswith("{"):
            candidates.append(stripped)

    tool_calls: list[dict[str, Any]] = []
    for raw in candidates:
        # Strip JavaScript-style line comments (// ...) that some
        # models insert into JSON output.
        cleaned = re.sub(r"//[^\n]*", "", raw)
        try:
            obj = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            continue

        items = obj if isinstance(obj, list) else [obj]
        for item in items:
            if isinstance(item, dict) and "name" in item:
                tool_calls.append({
                    "name": item["name"],
                    "arguments": _call_arguments(item),
                })

    # Last resort: tool calls written inline in prose, unfenced (Llama 3.1
    # does this: 'We can use X. {"name": "X", "parameters": {...}} Then ...').
    if not tool_calls:
        for item in _inline_json_objects(text):
            if "name" in item and ("parameters" in item or "arguments" in item):
                tool_calls.append({
                    "name": item["name"],
                    "arguments": _call_arguments(item),
                })

    return tool_calls if tool_calls else None


def _inline_json_objects(text: str) -> list[dict[str, Any]]:
    """Decode every top-level JSON object embedded in free text."""
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    i = text.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        i = text.find("{", end)
    return objects


def _coerce_arguments(raw: Any) -> dict[str, Any]:
    """Tool-call arguments as a dict; JSON-string arguments are decoded."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    """Arguments of a text-written tool call (``arguments`` or ``parameters``)."""
    raw = call.get("arguments", call.get("parameters", {}))
    return _coerce_arguments(raw)


def _strip_json_blocks(text: str) -> str:
    """Remove fenced JSON code blocks and bare JSON objects from text."""
    # Remove fenced blocks.
    text = _JSON_BLOCK_RE.sub("", text)
    # Remove bare JSON objects that look like tool calls.
    text = re.sub(
        r'\{\s*"name"\s*:.*?\}(?:\s*\})?',
        "",
        text,
        flags=re.DOTALL,
    )
    return text
