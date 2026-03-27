"""Tests for the Ollama LLM provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from apab.providers.ollama import (
    OllamaConnectionError,
    OllamaProvider,
    _convert_tools,
    _normalise_response,
    _parse_tool_calls_from_text,
    _strip_json_blocks,
)


class TestConvertTools:
    def test_mcp_to_ollama_format(self):
        mcp_tools = [
            {
                "name": "pattern_compute",
                "description": "Compute a pattern",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "nx": {"type": "integer"},
                        "ny": {"type": "integer"},
                    },
                    "required": ["nx", "ny"],
                },
            }
        ]

        result = _convert_tools(mcp_tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "pattern_compute"
        assert result[0]["function"]["description"] == "Compute a pattern"
        assert result[0]["function"]["parameters"]["type"] == "object"

    def test_empty_tools(self):
        assert _convert_tools([]) == []

    def test_missing_input_schema(self):
        tools = [{"name": "test", "description": "test"}]
        result = _convert_tools(tools)
        assert result[0]["function"]["parameters"] == {}


class TestNormaliseResponse:
    def test_text_only_response(self):
        response = MagicMock()
        response.message.role = "assistant"
        response.message.content = "Hello!"
        response.message.tool_calls = None

        result = _normalise_response(response)
        assert result["role"] == "assistant"
        assert result["content"] == "Hello!"
        assert result["tool_calls"] is None

    def test_tool_call_response(self):
        tc = MagicMock()
        tc.function.name = "pattern_compute"
        tc.function.arguments = {"nx": 4, "ny": 4}

        response = MagicMock()
        response.message.role = "assistant"
        response.message.content = ""
        response.message.tool_calls = [tc]

        result = _normalise_response(response)
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "pattern_compute"
        assert result["tool_calls"][0]["arguments"] == {"nx": 4, "ny": 4}

    def test_multiple_tool_calls(self):
        tc1 = MagicMock()
        tc1.function.name = "tool_a"
        tc1.function.arguments = {"x": 1}
        tc2 = MagicMock()
        tc2.function.name = "tool_b"
        tc2.function.arguments = {"y": 2}

        response = MagicMock()
        response.message.role = "assistant"
        response.message.content = None
        response.message.tool_calls = [tc1, tc2]

        result = _normalise_response(response)
        assert len(result["tool_calls"]) == 2


class TestOllamaProvider:
    @patch("ollama.Client")
    def test_properties(self, mock_client_cls):
        provider = OllamaProvider()
        assert provider.name == "ollama"
        assert provider.supports_tool_calling() is True
        assert provider.supports_streaming() is True

    @patch("ollama.Client")
    def test_chat_calls_client(self, mock_client_cls):
        # Setup mock
        tc = MagicMock()
        tc.function.name = "pattern_compute"
        tc.function.arguments = {"nx": 4}

        mock_response = MagicMock()
        mock_response.message.role = "assistant"
        mock_response.message.content = "I'll compute the pattern"
        mock_response.message.tool_calls = [tc]
        mock_client_cls.return_value.chat.return_value = mock_response

        provider = OllamaProvider(model="test-model")
        result = provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            tools=[{"name": "pattern_compute", "description": "test", "inputSchema": {}}],
        )

        assert result["role"] == "assistant"
        assert result["tool_calls"][0]["name"] == "pattern_compute"
        mock_client_cls.return_value.chat.assert_called_once()

    @patch("ollama.Client")
    def test_chat_raises_on_connection_error(self, mock_client_cls):
        import httpx

        mock_client_cls.return_value.chat.side_effect = httpx.ConnectError("refused")

        provider = OllamaProvider(model="test-model")
        with pytest.raises(OllamaConnectionError, match="ollama serve"):
            provider.chat(messages=[{"role": "user", "content": "hello"}])

    @patch("httpx.get")
    @patch("ollama.Client")
    def test_ping_success(self, mock_client_cls, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "qwen2.5-coder:14b"}]
        }
        mock_get.return_value = mock_resp

        provider = OllamaProvider(model="qwen2.5-coder:14b")
        ok, msg = provider.ping()
        assert ok is True
        assert "OK" in msg

    @patch("httpx.get")
    @patch("ollama.Client")
    def test_ping_server_down(self, mock_client_cls, mock_get):
        import httpx

        mock_get.side_effect = httpx.ConnectError("refused")

        provider = OllamaProvider(model="test-model")
        ok, msg = provider.ping()
        assert ok is False
        assert "Cannot reach" in msg

    @patch("httpx.get")
    @patch("ollama.Client")
    def test_ping_model_missing(self, mock_client_cls, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "llama3:latest"}]
        }
        mock_get.return_value = mock_resp

        provider = OllamaProvider(model="qwen2.5-coder:14b")
        ok, msg = provider.ping()
        assert ok is False
        assert "ollama pull" in msg


class TestParseToolCallsFromText:
    def test_fenced_json_block(self):
        text = (
            "I will compute the pattern.\n"
            '```json\n{"name": "pattern_compute", '
            '"arguments": {"nx": 8, "ny": 8}}\n```'
        )
        result = _parse_tool_calls_from_text(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "pattern_compute"
        assert result[0]["arguments"] == {"nx": 8, "ny": 8}

    def test_bare_json_object(self):
        text = '{"name": "system_evaluate", "arguments": {"freq_hz": 28e9}}'
        result = _parse_tool_calls_from_text(text)
        assert result is not None
        assert result[0]["name"] == "system_evaluate"

    def test_json_with_js_comments(self):
        text = (
            '```json\n{"name": "pattern_compute", '
            '"arguments": {"nx": 4 // number of elements\n}}\n```'
        )
        result = _parse_tool_calls_from_text(text)
        assert result is not None
        assert result[0]["name"] == "pattern_compute"
        assert result[0]["arguments"]["nx"] == 4

    def test_array_of_tool_calls(self):
        text = (
            '```json\n[{"name": "tool_a", "arguments": {}}, '
            '{"name": "tool_b", "arguments": {"x": 1}}]\n```'
        )
        result = _parse_tool_calls_from_text(text)
        assert result is not None
        assert len(result) == 2
        assert result[0]["name"] == "tool_a"
        assert result[1]["name"] == "tool_b"

    def test_plain_text_returns_none(self):
        text = "Let me explain how phased arrays work."
        result = _parse_tool_calls_from_text(text)
        assert result is None

    def test_malformed_json_returns_none(self):
        text = '```json\n{"name": "broken", arguments: bad}\n```'
        result = _parse_tool_calls_from_text(text)
        assert result is None

    def test_multiple_blocks_only_tool_call_extracted(self):
        text = (
            '```json\n{"result": "ok"}\n```\n'
            "Now calling:\n"
            '```json\n{"name": "pattern_compute", '
            '"arguments": {"nx": 4}}\n```'
        )
        result = _parse_tool_calls_from_text(text)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "pattern_compute"


class TestStripJsonBlocks:
    def test_fenced_block_removed(self):
        text = (
            'Before.\n```json\n{"name": "tool", '
            '"arguments": {}}\n```\nAfter.'
        )
        result = _strip_json_blocks(text)
        assert "Before." in result
        assert "After." in result
        assert "```" not in result

    def test_bare_json_with_name_removed(self):
        text = 'Calling: {"name": "tool", "arguments": {}}'
        result = _strip_json_blocks(text)
        assert '"name"' not in result

    def test_plain_text_unchanged(self):
        text = "No JSON here."
        assert _strip_json_blocks(text) == text


class TestNormaliseResponseFallback:
    def test_fallback_from_text_content(self):
        response = MagicMock()
        response.message.role = "assistant"
        response.message.tool_calls = None
        response.message.content = (
            'I will call the tool.\n```json\n{"name": "pattern_compute", '
            '"arguments": {"nx": 8}}\n```'
        )

        result = _normalise_response(response)
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "pattern_compute"
        # Content should have JSON stripped
        assert "```" not in (result["content"] or "")
