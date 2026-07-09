"""Tests for the Google Gemini LLM provider."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Build a minimal google.generativeai mock ──────────────────────────
# The real SDK isn't installed in the test environment, so we mock the
# entire module tree before importing the provider.

def _build_genai_mock():
    """Return a MagicMock that behaves like google.generativeai."""
    genai = MagicMock()
    # FunctionDeclaration and Tool need to be callable types.
    genai.types.FunctionDeclaration = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    genai.types.Tool = MagicMock(side_effect=lambda **kw: MagicMock(**kw))
    genai.GenerativeModel = MagicMock()
    return genai


@pytest.fixture(autouse=True)
def _mock_genai():
    """Inject a fake google.generativeai into sys.modules for every test."""
    genai = _build_genai_mock()
    mods = {
        "google": MagicMock(),
        "google.generativeai": genai,
    }
    with patch.dict(sys.modules, mods):
        yield genai


# Import provider AFTER mocking so the lazy import succeeds.
# We re-import inside each test class to pick up the fixture.

from apab.providers.gemini import (  # noqa: E402
    _clean_schema,
    _convert_messages,
    _extract_usage,
    _normalise_response,
)

# ── Message conversion tests ──────────────────────────────────────────

class TestConvertMessages:
    def test_basic_messages(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = _convert_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "model"  # Gemini uses "model"

    def test_system_merged_into_first_user(self):
        messages = [
            {"role": "system", "content": "You are an antenna engineer."},
            {"role": "user", "content": "Design an array"},
        ]
        result = _convert_messages(messages)
        assert len(result) == 1
        assert "antenna engineer" in result[0]["parts"][0]["text"]
        assert "Design an array" in result[0]["parts"][0]["text"]

    def test_no_system(self):
        messages = [{"role": "user", "content": "hi"}]
        result = _convert_messages(messages)
        assert len(result) == 1


# ── Schema cleaning tests ─────────────────────────────────────────────

class TestCleanSchema:
    def test_strips_additional_properties(self):
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "additionalProperties": False,
        }
        result = _clean_schema(schema)
        assert "additionalProperties" not in result
        assert result["type"] == "object"

    def test_preserves_valid_keys(self):
        schema = {"type": "object", "properties": {"x": {"type": "number"}}}
        assert _clean_schema(schema) == schema

    def test_recursive_cleaning(self):
        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {"y": {"type": "string"}},
                }
            },
        }
        result = _clean_schema(schema)
        assert "additionalProperties" not in result["properties"]["nested"]


# ── Response normalisation tests ──────────────────────────────────────

class TestNormaliseResponse:
    def test_text_only_response(self):
        part = MagicMock()
        part.function_call.name = ""
        part.text = "The results are ready."

        candidate = MagicMock()
        candidate.content.parts = [part]

        response = MagicMock()
        response.candidates = [candidate]

        result = _normalise_response(response)
        assert result["role"] == "assistant"
        assert result["content"] == "The results are ready."
        assert result["tool_calls"] is None

    def test_function_call_response(self):
        part = MagicMock()
        part.function_call.name = "pattern_compute"
        part.function_call.args = {"nx": 8, "ny": 8}
        part.text = ""

        candidate = MagicMock()
        candidate.content.parts = [part]

        response = MagicMock()
        response.candidates = [candidate]

        result = _normalise_response(response)
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "pattern_compute"
        assert result["tool_calls"][0]["arguments"] == {"nx": 8, "ny": 8}

    def test_multiple_function_calls(self):
        part1 = MagicMock()
        part1.function_call.name = "tool_a"
        part1.function_call.args = {"x": 1}
        part1.text = ""

        part2 = MagicMock()
        part2.function_call.name = "tool_b"
        part2.function_call.args = {"y": 2}
        part2.text = ""

        candidate = MagicMock()
        candidate.content.parts = [part1, part2]

        response = MagicMock()
        response.candidates = [candidate]

        result = _normalise_response(response)
        assert len(result["tool_calls"]) == 2


# ── Usage extraction tests ────────────────────────────────────────────

class TestExtractUsage:
    def test_extracts_token_counts(self):
        response = MagicMock()
        response.usage_metadata.prompt_token_count = 120
        response.usage_metadata.candidates_token_count = 60

        usage = _extract_usage(response, latency=1.0, model="gemini-2.5-pro")
        assert usage.prompt_tokens == 120
        assert usage.completion_tokens == 60
        assert usage.latency_s == 1.0

    def test_cost_estimate_gemini_pro(self):
        response = MagicMock()
        response.usage_metadata.prompt_token_count = 1_000_000
        response.usage_metadata.candidates_token_count = 1_000_000

        usage = _extract_usage(response, latency=0.0, model="gemini-2.5-pro")
        # 1.25 + 10.00 = 11.25
        assert abs(usage.cost_estimate_usd - 11.25) < 0.01

    def test_missing_usage_metadata(self):
        response = MagicMock(spec=[])
        usage = _extract_usage(response, latency=0.5, model="gemini-2.5-pro")
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0


# ── Provider integration tests (mocked) ──────────────────────────────

class TestGeminiProvider:
    def test_properties(self, _mock_genai):
        from apab.providers.gemini import GeminiProvider
        provider = GeminiProvider.__new__(GeminiProvider)
        provider._genai = _mock_genai
        provider._model_name = "gemini-2.5-pro"
        provider._model = _mock_genai.GenerativeModel.return_value
        provider._last_usage = None

        assert provider.name == "gemini"
        assert provider.supports_tool_calling() is True
        assert provider.supports_streaming() is True

    def test_chat_calls_generate_content(self, _mock_genai):
        from apab.providers.gemini import GeminiProvider

        # Build mock response
        part = MagicMock()
        part.function_call.name = "pattern_compute"
        part.function_call.args = {"nx": 4}
        part.text = ""

        candidate = MagicMock()
        candidate.content.parts = [part]

        mock_response = MagicMock()
        mock_response.candidates = [candidate]
        mock_response.usage_metadata.prompt_token_count = 50
        mock_response.usage_metadata.candidates_token_count = 25

        mock_model = _mock_genai.GenerativeModel.return_value
        mock_model.generate_content.return_value = mock_response

        provider = GeminiProvider.__new__(GeminiProvider)
        provider._genai = _mock_genai
        provider._model_name = "gemini-2.5-pro"
        provider._model = mock_model
        provider._last_usage = None

        result = provider.chat(
            messages=[{"role": "user", "content": "hello"}],
            tools=[{
                "name": "pattern_compute",
                "description": "test",
                "inputSchema": {"type": "object", "properties": {"nx": {"type": "integer"}}},
            }],
        )

        assert result["tool_calls"][0]["name"] == "pattern_compute"
        mock_model.generate_content.assert_called_once()
