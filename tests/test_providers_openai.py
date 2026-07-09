"""Tests for the OpenAI LLM provider."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from apab.providers.openai import (
    _convert_tools,
    _extract_usage,
    _normalise_response,
)

# ── Sample MCP tool schema used across tests ──────────────────────────

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
    {
        "name": "edgefem_run_unit_cell",
        "description": "Run a unit-cell simulation",
        "inputSchema": {
            "type": "object",
            "properties": {
                "freq_ghz": {"type": "number"},
            },
        },
    },
]


# ── Tool conversion tests ─────────────────────────────────────────────

class TestConvertTools:
    def test_mcp_to_openai_format(self):
        result = _convert_tools(SAMPLE_MCP_TOOLS)
        assert len(result) == 2
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "pattern_compute"
        assert result[0]["function"]["description"] == "Compute an array radiation pattern"
        assert result[0]["function"]["parameters"]["type"] == "object"
        assert "nx" in result[0]["function"]["parameters"]["properties"]

    def test_empty_tools(self):
        assert _convert_tools([]) == []

    def test_missing_input_schema(self):
        tools = [{"name": "test", "description": "test"}]
        result = _convert_tools(tools)
        assert result[0]["function"]["parameters"] == {}

    def test_fallback_to_parameters_key(self):
        tools = [{"name": "test", "description": "t", "parameters": {"type": "object"}}]
        result = _convert_tools(tools)
        assert result[0]["function"]["parameters"] == {"type": "object"}


# ── Response normalisation tests ──────────────────────────────────────

class TestNormaliseResponse:
    def test_text_only_response(self):
        msg = MagicMock()
        msg.role = "assistant"
        msg.content = "Here are the results."
        msg.tool_calls = None

        response = MagicMock()
        response.choices = [MagicMock(message=msg)]

        result = _normalise_response(response)
        assert result["role"] == "assistant"
        assert result["content"] == "Here are the results."
        assert result["tool_calls"] is None

    def test_single_tool_call(self):
        tc = MagicMock()
        tc.function.name = "pattern_compute"
        tc.function.arguments = '{"nx": 8, "ny": 8}'

        msg = MagicMock()
        msg.role = "assistant"
        msg.content = None
        msg.tool_calls = [tc]

        response = MagicMock()
        response.choices = [MagicMock(message=msg)]

        result = _normalise_response(response)
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "pattern_compute"
        assert result["tool_calls"][0]["arguments"] == {"nx": 8, "ny": 8}

    def test_multiple_tool_calls(self):
        tc1 = MagicMock()
        tc1.function.name = "tool_a"
        tc1.function.arguments = '{"x": 1}'
        tc2 = MagicMock()
        tc2.function.name = "tool_b"
        tc2.function.arguments = '{"y": 2}'

        msg = MagicMock()
        msg.role = "assistant"
        msg.content = None
        msg.tool_calls = [tc1, tc2]

        response = MagicMock()
        response.choices = [MagicMock(message=msg)]

        result = _normalise_response(response)
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["arguments"] == {"x": 1}
        assert result["tool_calls"][1]["arguments"] == {"y": 2}

    def test_malformed_json_arguments(self):
        tc = MagicMock()
        tc.function.name = "broken"
        tc.function.arguments = "not valid json{{"

        msg = MagicMock()
        msg.role = "assistant"
        msg.content = None
        msg.tool_calls = [tc]

        response = MagicMock()
        response.choices = [MagicMock(message=msg)]

        result = _normalise_response(response)
        assert result["tool_calls"][0]["arguments"] == {}

    def test_dict_arguments_passed_through(self):
        tc = MagicMock()
        tc.function.name = "test"
        tc.function.arguments = {"already": "parsed"}

        msg = MagicMock()
        msg.role = "assistant"
        msg.content = None
        msg.tool_calls = [tc]

        response = MagicMock()
        response.choices = [MagicMock(message=msg)]

        result = _normalise_response(response)
        assert result["tool_calls"][0]["arguments"] == {"already": "parsed"}


# ── Usage extraction tests ────────────────────────────────────────────

class TestExtractUsage:
    def test_extracts_token_counts(self):
        response = MagicMock()
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50

        usage = _extract_usage(response, latency=1.5, model="gpt-4.1-mini")
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.latency_s == 1.5

    def test_cost_estimate_gpt4_1_mini(self):
        response = MagicMock()
        response.usage.prompt_tokens = 1_000_000
        response.usage.completion_tokens = 1_000_000

        usage = _extract_usage(response, latency=0.0, model="gpt-4.1-mini")
        # 0.40 + 1.60 = 2.00
        assert abs(usage.cost_estimate_usd - 2.00) < 0.01

    def test_missing_usage(self):
        response = MagicMock(spec=[])  # No usage attribute
        usage = _extract_usage(response, latency=0.5, model="gpt-4.1")
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0


# ── Provider integration tests (mocked client) ───────────────────────

@pytest.fixture()
def _mock_openai():
    """Inject a fake openai module into sys.modules."""
    mock_openai = MagicMock()
    with patch.dict(sys.modules, {"openai": mock_openai}):
        yield mock_openai


class TestOpenAIProvider:
    def test_properties(self, _mock_openai):
        from apab.providers.openai import OpenAIProvider
        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._model = "gpt-4.1-mini"
        provider._client = _mock_openai.OpenAI.return_value
        provider._last_usage = None

        assert provider.name == "openai"
        assert provider.supports_tool_calling() is True
        assert provider.supports_streaming() is True

    def test_chat_calls_client(self, _mock_openai):
        from apab.providers.openai import OpenAIProvider

        # Build mock response
        tc = MagicMock()
        tc.function.name = "pattern_compute"
        tc.function.arguments = '{"nx": 4}'

        msg = MagicMock()
        msg.role = "assistant"
        msg.content = "Computing pattern"
        msg.tool_calls = [tc]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=msg)]
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 25

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._model = "gpt-4.1-mini"
        provider._client = mock_client
        provider._last_usage = None

        result = provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            tools=SAMPLE_MCP_TOOLS,
        )

        assert result["role"] == "assistant"
        assert result["tool_calls"][0]["name"] == "pattern_compute"
        mock_client.chat.completions.create.assert_called_once()

    def test_chat_without_tools(self, _mock_openai):
        from apab.providers.openai import OpenAIProvider

        msg = MagicMock()
        msg.role = "assistant"
        msg.content = "Just text"
        msg.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=msg)]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._model = "gpt-4.1-mini"
        provider._client = mock_client
        provider._last_usage = None

        result = provider.chat(messages=[{"role": "user", "content": "hi"}])

        assert result["content"] == "Just text"
        assert result["tool_calls"] is None

    def test_usage_tracked(self, _mock_openai):
        from apab.providers.openai import OpenAIProvider

        msg = MagicMock()
        msg.role = "assistant"
        msg.content = "done"
        msg.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=msg)]
        mock_response.usage.prompt_tokens = 200
        mock_response.usage.completion_tokens = 100

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._model = "gpt-4.1"
        provider._client = mock_client
        provider._last_usage = None

        assert provider.last_usage is None
        provider.chat(messages=[{"role": "user", "content": "test"}])
        assert provider.last_usage is not None
        assert provider.last_usage.prompt_tokens == 200
        assert provider.last_usage.completion_tokens == 100
