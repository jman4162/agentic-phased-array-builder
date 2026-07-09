"""Tests for the OpenAI-compatible LLM provider."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

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


@pytest.fixture()
def _mock_openai():
    """Inject a fake openai module into sys.modules."""
    mock_openai = MagicMock()
    with patch.dict(sys.modules, {"openai": mock_openai}):
        yield mock_openai


class TestOpenAICompatibleProvider:
    def test_properties(self, _mock_openai):
        from apab.providers.openai_compat import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            base_url="http://localhost:8000/v1",
            model="my-model",
        )

        assert provider.name == "openai_compatible"
        assert provider.supports_tool_calling() is True
        assert provider.supports_streaming() is True

    def test_delegates_to_openai_client(self, _mock_openai):
        from apab.providers.openai_compat import OpenAICompatibleProvider

        _mock_openai.OpenAI.return_value = MagicMock()

        OpenAICompatibleProvider(
            base_url="http://my-server:9000/v1",
            model="custom-model",
            api_key="test-key",
        )

        # Verify the OpenAI client was created with the custom base_url
        _mock_openai.OpenAI.assert_called_once_with(
            api_key="test-key",
            base_url="http://my-server:9000/v1",
        )

    def test_chat_delegates(self, _mock_openai):
        from apab.providers.openai_compat import OpenAICompatibleProvider

        # Build mock response
        msg = MagicMock()
        msg.role = "assistant"
        msg.content = "Hello from vLLM"
        msg.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=msg)]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        _mock_openai.OpenAI.return_value = mock_client

        provider = OpenAICompatibleProvider(
            base_url="http://localhost:8000/v1",
            model="my-model",
        )

        result = provider.chat(messages=[{"role": "user", "content": "hi"}])

        assert result["role"] == "assistant"
        assert result["content"] == "Hello from vLLM"
        assert result["tool_calls"] is None
        mock_client.chat.completions.create.assert_called_once()

    def test_chat_with_tools(self, _mock_openai):
        from apab.providers.openai_compat import OpenAICompatibleProvider

        tc = MagicMock()
        tc.function.name = "pattern_compute"
        tc.function.arguments = '{"nx": 4, "ny": 4}'

        msg = MagicMock()
        msg.role = "assistant"
        msg.content = None
        msg.tool_calls = [tc]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=msg)]
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 15

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        _mock_openai.OpenAI.return_value = mock_client

        provider = OpenAICompatibleProvider(
            base_url="http://localhost:8000/v1",
            model="my-model",
        )

        result = provider.chat(
            messages=[{"role": "user", "content": "compute pattern"}],
            tools=SAMPLE_MCP_TOOLS,
        )

        assert result["tool_calls"] is not None
        assert result["tool_calls"][0]["name"] == "pattern_compute"
        assert result["tool_calls"][0]["arguments"] == {"nx": 4, "ny": 4}

    def test_usage_forwarded(self, _mock_openai):
        from apab.providers.openai_compat import OpenAICompatibleProvider

        msg = MagicMock()
        msg.role = "assistant"
        msg.content = "done"
        msg.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=msg)]
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        _mock_openai.OpenAI.return_value = mock_client

        provider = OpenAICompatibleProvider(
            base_url="http://localhost:8000/v1",
            model="my-model",
        )

        assert provider.last_usage is None
        provider.chat(messages=[{"role": "user", "content": "test"}])
        assert provider.last_usage is not None
        assert provider.last_usage.prompt_tokens == 100
        assert provider.last_usage.completion_tokens == 50
