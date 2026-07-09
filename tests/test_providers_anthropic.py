"""Tests for the Anthropic LLM provider."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from apab.providers.anthropic import (
    _convert_tools,
    _extract_system,
    _extract_usage,
    _normalise_response,
)

# ── Sample MCP tool schema ────────────────────────────────────────────

SAMPLE_MCP_TOOLS = [
    {
        "name": "pattern_compute",
        "description": "Compute an array radiation pattern",
        "inputSchema": {
            "type": "object",
            "properties": {
                "nx": {"type": "integer"},
                "ny": {"type": "integer"},
            },
            "required": ["nx", "ny"],
        },
    },
]


# ── System extraction tests ───────────────────────────────────────────

class TestExtractSystem:
    def test_extracts_system_message(self):
        messages = [
            {"role": "system", "content": "You are an antenna engineer."},
            {"role": "user", "content": "Design an array"},
        ]
        system_text, api_msgs = _extract_system(messages)
        assert system_text == "You are an antenna engineer."
        assert len(api_msgs) == 1
        assert api_msgs[0]["role"] == "user"

    def test_no_system_message(self):
        messages = [{"role": "user", "content": "hello"}]
        system_text, api_msgs = _extract_system(messages)
        assert system_text is None
        assert len(api_msgs) == 1

    def test_multiple_non_system_messages_preserved(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        system_text, api_msgs = _extract_system(messages)
        assert system_text == "sys"
        assert len(api_msgs) == 3


# ── Tool conversion tests ─────────────────────────────────────────────

class TestConvertTools:
    def test_mcp_to_anthropic_format(self):
        result = _convert_tools(SAMPLE_MCP_TOOLS)
        assert len(result) == 1
        assert result[0]["name"] == "pattern_compute"
        assert "input_schema" in result[0]
        assert result[0]["input_schema"]["type"] == "object"
        # Should NOT have inputSchema (camelCase).
        assert "inputSchema" not in result[0]

    def test_empty_tools(self):
        assert _convert_tools([]) == []

    def test_missing_input_schema(self):
        tools = [{"name": "test", "description": "test"}]
        result = _convert_tools(tools)
        assert result[0]["input_schema"] == {}


# ── Response normalisation tests ──────────────────────────────────────

class TestNormaliseResponse:
    def test_text_only_response(self):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Here are the results."

        response = MagicMock()
        response.content = [text_block]

        result = _normalise_response(response)
        assert result["role"] == "assistant"
        assert result["content"] == "Here are the results."
        assert result["tool_calls"] is None

    def test_tool_use_response(self):
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "pattern_compute"
        tool_block.input = {"nx": 8, "ny": 8}

        response = MagicMock()
        response.content = [tool_block]

        result = _normalise_response(response)
        assert result["content"] is None
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "pattern_compute"
        assert result["tool_calls"][0]["arguments"] == {"nx": 8, "ny": 8}

    def test_mixed_text_and_tool_use(self):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "I'll compute the pattern."

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "pattern_compute"
        tool_block.input = {"nx": 4, "ny": 4}

        response = MagicMock()
        response.content = [text_block, tool_block]

        result = _normalise_response(response)
        assert result["content"] == "I'll compute the pattern."
        assert len(result["tool_calls"]) == 1

    def test_multiple_tool_calls(self):
        tc1 = MagicMock()
        tc1.type = "tool_use"
        tc1.name = "tool_a"
        tc1.input = {"x": 1}
        tc2 = MagicMock()
        tc2.type = "tool_use"
        tc2.name = "tool_b"
        tc2.input = {"y": 2}

        response = MagicMock()
        response.content = [tc1, tc2]

        result = _normalise_response(response)
        assert len(result["tool_calls"]) == 2


# ── Usage extraction tests ────────────────────────────────────────────

class TestExtractUsage:
    def test_extracts_token_counts(self):
        response = MagicMock()
        response.usage.input_tokens = 150
        response.usage.output_tokens = 80

        usage = _extract_usage(response, latency=2.0, model="claude-sonnet-4-20250514")
        assert usage.prompt_tokens == 150
        assert usage.completion_tokens == 80
        assert usage.latency_s == 2.0

    def test_cost_estimate_sonnet(self):
        response = MagicMock()
        response.usage.input_tokens = 1_000_000
        response.usage.output_tokens = 1_000_000

        usage = _extract_usage(response, latency=0.0, model="claude-sonnet-4-20250514")
        # 3.00 + 15.00 = 18.00
        assert abs(usage.cost_estimate_usd - 18.00) < 0.01

    def test_missing_usage(self):
        response = MagicMock(spec=[])
        usage = _extract_usage(response, latency=0.5, model="claude-sonnet-4-20250514")
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0


# ── Provider integration tests (mocked client) ───────────────────────

@pytest.fixture()
def _mock_anthropic():
    """Inject a fake anthropic module into sys.modules."""
    mock_mod = MagicMock()
    with patch.dict(sys.modules, {"anthropic": mock_mod}):
        yield mock_mod


class TestAnthropicProvider:
    def test_properties(self, _mock_anthropic):
        from apab.providers.anthropic import AnthropicProvider
        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider._model = "claude-sonnet-4-20250514"
        provider._max_tokens = 4096
        provider._client = _mock_anthropic.Anthropic.return_value
        provider._last_usage = None

        assert provider.name == "anthropic"
        assert provider.supports_tool_calling() is True
        assert provider.supports_streaming() is True

    def test_chat_with_tools(self, _mock_anthropic):
        from apab.providers.anthropic import AnthropicProvider

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "pattern_compute"
        tool_block.input = {"nx": 4}

        mock_response = MagicMock()
        mock_response.content = [tool_block]
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 25

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider._model = "claude-sonnet-4-20250514"
        provider._max_tokens = 4096
        provider._client = mock_client
        provider._last_usage = None

        result = provider.chat(
            messages=[
                {"role": "system", "content": "You are an engineer."},
                {"role": "user", "content": "compute a pattern"},
            ],
            tools=SAMPLE_MCP_TOOLS,
        )

        assert result["tool_calls"][0]["name"] == "pattern_compute"
        # System should be extracted to top-level kwarg.
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs.get("system") == "You are an engineer."

    def test_chat_without_tools(self, _mock_anthropic):
        from apab.providers.anthropic import AnthropicProvider

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Done."

        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider._model = "claude-sonnet-4-20250514"
        provider._max_tokens = 4096
        provider._client = mock_client
        provider._last_usage = None

        result = provider.chat(messages=[{"role": "user", "content": "hi"}])

        assert result["content"] == "Done."
        assert result["tool_calls"] is None

    def test_usage_tracked(self, _mock_anthropic):
        from apab.providers.anthropic import AnthropicProvider

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "ok"

        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_response.usage.input_tokens = 300
        mock_response.usage.output_tokens = 150

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider._model = "claude-sonnet-4-20250514"
        provider._max_tokens = 4096
        provider._client = mock_client
        provider._last_usage = None

        assert provider.last_usage is None
        provider.chat(messages=[{"role": "user", "content": "test"}])
        assert provider.last_usage is not None
        assert provider.last_usage.prompt_tokens == 300
        assert provider.last_usage.completion_tokens == 150
