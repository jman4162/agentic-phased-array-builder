"""Tests for the agent orchestrator."""

from __future__ import annotations

import json
from typing import Any

import pytest

from apab.agent.orchestrator import AgentOrchestrator
from apab.core.schemas import ProjectConfig, ProjectMeta
from apab.providers.usage import ProviderUsage


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


class UsageFakeProvider(FakeProvider):
    """FakeProvider that also reports per-call usage."""

    @property
    def last_usage(self) -> ProviderUsage | None:
        if self._call_count == 0:
            return None
        return ProviderUsage(
            prompt_tokens=100,
            completion_tokens=20,
            latency_s=0.05,
            cost_estimate_usd=0.001,
        )


_TOOL_CALL_RESPONSE = {
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

_FINAL_RESPONSE = {
    "role": "assistant",
    "content": "All done.",
    "tool_calls": None,
}


class TestEvents:
    def test_event_sequence(self, config):
        provider = FakeProvider([_TOOL_CALL_RESPONSE, _FINAL_RESPONSE])
        orch = AgentOrchestrator(config, provider=provider)

        events: list[tuple[str, dict]] = []
        orch.run_to_completion(
            "Compute a pattern",
            on_event=lambda name, payload: events.append((name, payload)),
        )

        names = [n for n, _ in events]
        assert names == [
            "session_start",
            "turn_start",
            "tool_call",
            "tool_result",
            "turn_start",
            "assistant_message",
        ]

        payloads = dict(events)
        assert payloads["session_start"]["run_id"]
        assert payloads["tool_call"]["name"] == "pattern_compute"
        assert payloads["tool_result"]["tool"] == "pattern_compute"
        assert payloads["assistant_message"]["content"] == "All done."

    def test_max_turns_event(self, config):
        provider = FakeProvider([_TOOL_CALL_RESPONSE] * 5)
        orch = AgentOrchestrator(config, provider=provider)

        events: list[str] = []
        orch.run_to_completion(
            "Loop forever",
            max_turns=2,
            on_event=lambda name, payload: events.append(name),
        )
        assert events[-1] == "max_turns"

    def test_broken_callback_does_not_break_run(self, config):
        provider = FakeProvider([_FINAL_RESPONSE])
        orch = AgentOrchestrator(config, provider=provider)

        def broken(name: str, payload: dict) -> None:
            raise RuntimeError("renderer bug")

        result = orch.run_to_completion("hello", on_event=broken)
        assert result == "All done."


class TestManifest:
    def test_manifest_written_on_success(self, config):
        provider = UsageFakeProvider([_TOOL_CALL_RESPONSE, _FINAL_RESPONSE])
        orch = AgentOrchestrator(config, provider=provider)
        orch.run_to_completion("Compute a pattern")

        manifest_path = orch.run_context.run_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())

        assert manifest["run_id"] == orch.run_context.run_id
        assert manifest["status"] == "success"
        assert manifest["config_hash"]
        assert manifest["provider_name"] == config.llm.provider
        assert isinstance(manifest["artifacts"], list)

        usage = manifest["usage"]
        assert usage["llm_calls"] == 2
        assert usage["prompt_tokens"] == 200
        assert usage["completion_tokens"] == 40
        assert usage["cost_estimate_usd"] == pytest.approx(0.002)

    def test_manifest_status_max_turns(self, config):
        provider = FakeProvider([_TOOL_CALL_RESPONSE] * 5)
        orch = AgentOrchestrator(config, provider=provider)
        orch.run_to_completion("Loop forever", max_turns=2)

        manifest = json.loads(
            (orch.run_context.run_dir / "manifest.json").read_text()
        )
        assert manifest["status"] == "max_turns"

    def test_manifest_status_error(self, config):
        class ExplodingProvider(FakeProvider):
            def chat(self, messages, tools=None, **kwargs):
                raise RuntimeError("provider down")

        orch = AgentOrchestrator(config, provider=ExplodingProvider([]))
        with pytest.raises(RuntimeError, match="provider down"):
            orch.run_to_completion("hello")

        manifest = json.loads(
            (orch.run_context.run_dir / "manifest.json").read_text()
        )
        assert manifest["status"] == "error"

    def test_usage_zero_without_last_usage(self, config):
        provider = FakeProvider([_FINAL_RESPONSE])
        orch = AgentOrchestrator(config, provider=provider)
        orch.run_to_completion("hello")

        manifest = json.loads(
            (orch.run_context.run_dir / "manifest.json").read_text()
        )
        assert manifest["usage"]["llm_calls"] == 0
        assert manifest["usage"]["prompt_tokens"] == 0

    def test_report_builder_reads_manifest(self, config):
        provider = FakeProvider([_FINAL_RESPONSE])
        orch = AgentOrchestrator(config, provider=provider)
        orch.run_to_completion("hello")

        from apab.report.build_report import ReportBuilder

        report = ReportBuilder(orch.run_context.run_dir).build_markdown()
        assert orch.run_context.run_id in report


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
