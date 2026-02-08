"""Tests for the Ollama LLM provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from apab.providers.ollama import OllamaProvider, _convert_tools, _normalise_response


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
