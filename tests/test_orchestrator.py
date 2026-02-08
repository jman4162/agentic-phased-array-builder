"""Tests for the agent orchestrator."""

from __future__ import annotations

from typing import Any

import pytest

from apab.agent.orchestrator import AgentOrchestrator
from apab.core.schemas import ProjectConfig, ProjectMeta


class FakeProvider:
    """A fake LLM provider for testing the agentic loop."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    @property
    def name(self) -> str:
        return "fake"

    def supports_tool_calling(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return False

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            resp = {"role": "assistant", "content": "Done.", "tool_calls": None}
        self._call_count += 1
        return resp


@pytest.fixture()
def config(tmp_path):
    return ProjectConfig(
        project=ProjectMeta(name="test", workspace=str(tmp_path / "workspace")),
    )


class TestSessionLifecycle:
    def test_start_session_creates_run(self, config, tmp_path):
        provider = FakeProvider([
            {"role": "assistant", "content": "Hello!", "tool_calls": None},
        ])
        orch = AgentOrchestrator(config, provider=provider)
        ctx = orch.start_session("Design an array")

        assert ctx is not None
        assert ctx.run_id is not None
        assert len(orch.messages) == 2  # system + user

    def test_step_returns_response(self, config):
        provider = FakeProvider([
            {"role": "assistant", "content": "I'll help!", "tool_calls": None},
        ])
        orch = AgentOrchestrator(config, provider=provider)
        orch.start_session("Hello")
        response = orch.step()

        assert response["role"] == "assistant"
        assert response["content"] == "I'll help!"


class TestAgenticLoop:
    def test_run_to_completion_no_tools(self, config):
        provider = FakeProvider([
            {"role": "assistant", "content": "Here is the answer.", "tool_calls": None},
        ])
        orch = AgentOrchestrator(config, provider=provider)
        result = orch.run_to_completion("What is 2+2?")

        assert result == "Here is the answer."

    def test_run_to_completion_with_tool_call(self, config):
        """Provider makes one tool call, then returns final text."""
        provider = FakeProvider([
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "name": "pattern_compute",
                        "arguments": {
                            "nx": 4,
                            "ny": 4,
                            "dx_m": 0.005,
                            "dy_m": 0.005,
                            "freq_hz": 10e9,
                        },
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "The directivity is about 17 dBi.",
                "tool_calls": None,
            },
        ])
        orch = AgentOrchestrator(config, provider=provider)
        result = orch.run_to_completion("Compute a 4x4 array pattern")

        assert "directivity" in result.lower() or "17" in result

    def test_max_turns_reached(self, config):
        """If the LLM keeps calling tools, we stop after max_turns."""
        responses = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "name": "pattern_compute",
                        "arguments": {
                            "nx": 4, "ny": 4,
                            "dx_m": 0.005, "dy_m": 0.005,
                            "freq_hz": 10e9,
                        },
                    },
                ],
            }
        ] * 5

        provider = FakeProvider(responses)
        orch = AgentOrchestrator(config, provider=provider)
        result = orch.run_to_completion("Loop forever", max_turns=3)

        assert "maximum" in result.lower()


class TestRedactionModes:
    def test_none_mode(self, config, caplog):
        import logging

        config.llm.redaction_mode = "none"
        provider = FakeProvider([
            {"role": "assistant", "content": "test", "tool_calls": None},
        ])
        orch = AgentOrchestrator(config, provider=provider)

        with caplog.at_level(logging.DEBUG, logger="apab.agent.orchestrator"):
            orch.run_to_completion("hello")

    def test_strict_mode(self, config, caplog):
        import logging

        config.llm.redaction_mode = "strict"
        provider = FakeProvider([
            {"role": "assistant", "content": "secret", "tool_calls": None},
        ])
        orch = AgentOrchestrator(config, provider=provider)

        with caplog.at_level(logging.DEBUG, logger="apab.agent.orchestrator"):
            orch.run_to_completion("hello")
            # Verify "secret" doesn't appear in logs under strict mode
            for record in caplog.records:
                if "apab.agent.orchestrator" in record.name:
                    assert "secret" not in record.message
