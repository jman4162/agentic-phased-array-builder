"""Tests for the OpenTelemetry observability layer."""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import patch

import pytest

from apab.agent.orchestrator import AgentOrchestrator
from apab.core.schemas import ObservabilitySpec, ProjectConfig, ProjectMeta, RedactionMode
from apab.observability import (
    capture_args,
    capture_text,
    current_trace_ids,
    init_observability,
    is_enabled,
    shutdown_observability,
    span,
)


@pytest.fixture(autouse=True)
def _clean_tracing_state(monkeypatch):
    """Reset tracer globals and strip env vars that would leak exporters."""
    monkeypatch.delenv("APAB_OBSERVABILITY", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    shutdown_observability()
    yield
    shutdown_observability()


@pytest.fixture()
def config(tmp_path):
    cfg = ProjectConfig(
        project=ProjectMeta(name="test", workspace=str(tmp_path / "workspace")),
    )
    cfg.observability.enabled = True
    return cfg


class FakeProvider:
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

    def chat(self, messages, tools=None, **kwargs):
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            resp = {"role": "assistant", "content": "Done.", "tool_calls": None}
        self._call_count += 1
        return resp


class UsageFakeProvider(FakeProvider):
    @property
    def last_usage(self):
        from apab.providers.usage import ProviderUsage

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


def _read_trace(run_dir) -> list[dict[str, Any]]:
    path = run_dir / "trace.jsonl"
    assert path.exists(), "trace.jsonl missing from run bundle"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _by_name(spans: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {s["name"]: s for s in spans}


class TestRedaction:
    ARGS = {"nx": 4, "freq_hz": 28e9, "api_key": "secret-value"}

    def test_none_captures_values(self):
        attrs = capture_args(self.ARGS, RedactionMode.none)
        assert "secret-value" in attrs["args_json"]
        assert attrs["args_hash"]

    def test_metadata_only_captures_keys(self):
        attrs = capture_args(self.ARGS, RedactionMode.metadata_only)
        assert "args_json" not in attrs
        assert attrs["arg_keys"] == ["api_key", "freq_hz", "nx"]

    def test_strict_captures_hash_only(self):
        attrs = capture_args(self.ARGS, RedactionMode.strict)
        assert set(attrs) == {"args_hash"}
        assert "secret-value" not in json.dumps(attrs)

    def test_hash_stable_across_modes(self):
        h1 = capture_args(self.ARGS, RedactionMode.none)["args_hash"]
        h2 = capture_args(dict(reversed(self.ARGS.items())), RedactionMode.strict)["args_hash"]
        assert h1 == h2

    def test_capture_text_modes(self):
        text = "x" * 300
        assert capture_text(text, RedactionMode.strict) is None
        assert capture_text(text, RedactionMode.metadata_only) == "<300 chars>"
        truncated = capture_text(text, RedactionMode.none)
        assert truncated.endswith("...")
        assert len(truncated) == 203


class TestDisabled:
    def test_span_is_noop_without_init(self):
        assert not is_enabled()
        with span("apab.test", **{"k": "v"}) as s:
            s.set_attribute("more", 1)  # must not raise
        assert current_trace_ids() is None

    def test_disabled_config_produces_no_trace(self, config):
        config.observability.enabled = False
        provider = FakeProvider([_FINAL_RESPONSE])
        orch = AgentOrchestrator(config, provider=provider)
        result = orch.run_to_completion("hello")

        assert result == "All done."
        assert not (orch.run_context.run_dir / "trace.jsonl").exists()

    def test_missing_otel_is_soft(self, config, caplog):
        with patch.dict(sys.modules, {"opentelemetry": None}):
            ok = init_observability(config.observability)
        assert ok is False
        assert not is_enabled()
        assert "apab[observability]" in caplog.text

    def test_env_var_forces_enabled(self, config, monkeypatch):
        config.observability.enabled = False
        monkeypatch.setenv("APAB_OBSERVABILITY", "1")
        provider = FakeProvider([_FINAL_RESPONSE])
        orch = AgentOrchestrator(config, provider=provider)
        orch.run_to_completion("hello")

        assert (orch.run_context.run_dir / "trace.jsonl").exists()


class TestSpanHierarchy:
    def test_session_turn_llm_tool_spans(self, config):
        provider = UsageFakeProvider([_TOOL_CALL_RESPONSE, _FINAL_RESPONSE])
        orch = AgentOrchestrator(config, provider=provider)
        orch.run_to_completion("Compute a pattern")

        spans = _read_trace(orch.run_context.run_dir)
        named = _by_name(spans)

        session = named["apab.session"]
        tool = named["apab.tool.pattern_compute"]
        turns = [s for s in spans if s["name"] == "apab.turn"]
        chats = [s for s in spans if s["name"] == "apab.llm.chat"]

        assert session["parent_span_id"] is None
        assert len(turns) == 2
        assert len(chats) == 2
        assert all(t["parent_span_id"] == session["span_id"] for t in turns)
        turn_ids = {t["span_id"] for t in turns}
        assert all(c["parent_span_id"] in turn_ids for c in chats)
        assert tool["parent_span_id"] in turn_ids

        # One trace id everywhere
        assert len({s["trace_id"] for s in spans}) == 1

    def test_session_attributes(self, config):
        provider = UsageFakeProvider([_TOOL_CALL_RESPONSE, _FINAL_RESPONSE])
        orch = AgentOrchestrator(config, provider=provider)
        orch.run_to_completion("Compute a pattern")

        named = _by_name(_read_trace(orch.run_context.run_dir))
        attrs = named["apab.session"]["attributes"]

        assert attrs["apab.run_id"] == orch.run_context.run_id
        assert attrs["gen_ai.system"] == "fake"
        assert attrs["apab.status"] == "success"
        assert attrs["gen_ai.usage.input_tokens"] == 200
        assert attrs["gen_ai.usage.output_tokens"] == 40
        assert attrs["apab.config_hash"]

    def test_llm_chat_usage_attributes(self, config):
        provider = UsageFakeProvider([_FINAL_RESPONSE])
        orch = AgentOrchestrator(config, provider=provider)
        orch.run_to_completion("hello")

        named = _by_name(_read_trace(orch.run_context.run_dir))
        attrs = named["apab.llm.chat"]["attributes"]

        assert attrs["gen_ai.usage.input_tokens"] == 100
        assert attrs["gen_ai.usage.output_tokens"] == 20
        assert attrs["apab.tool_call_count"] == 0
        assert attrs["apab.response.has_content"] is True

    def test_tool_span_attributes(self, config):
        provider = FakeProvider([_TOOL_CALL_RESPONSE, _FINAL_RESPONSE])
        orch = AgentOrchestrator(config, provider=provider)
        orch.run_to_completion("Compute a pattern")

        named = _by_name(_read_trace(orch.run_context.run_dir))
        attrs = named["apab.tool.pattern_compute"]["attributes"]

        assert attrs["apab.tool.name"] == "pattern_compute"
        assert attrs["apab.tool.status"] == "ok"
        assert attrs["apab.tool.args_hash"]
        assert "args_json" in json.dumps(attrs) or "apab.tool.args_json" in attrs
        assert attrs["apab.tool.result_summary"]


class TestErrorHandling:
    def test_unknown_tool_marks_span_error(self, config):
        bad_call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"name": "no_such_tool", "arguments": {}}],
        }
        provider = FakeProvider([bad_call, _FINAL_RESPONSE])
        orch = AgentOrchestrator(config, provider=provider)
        result = orch.run_to_completion("hello")

        assert result == "All done."  # run continued past the tool error
        named = _by_name(_read_trace(orch.run_context.run_dir))
        assert named["apab.tool.no_such_tool"]["attributes"]["apab.tool.status"] == "error"

    def test_provider_error_marks_session(self, config):
        class ExplodingProvider(FakeProvider):
            def chat(self, messages, tools=None, **kwargs):
                raise RuntimeError("provider down")

        orch = AgentOrchestrator(config, provider=ExplodingProvider([]))
        with pytest.raises(RuntimeError):
            orch.run_to_completion("hello")

        named = _by_name(_read_trace(orch.run_context.run_dir))
        assert named["apab.session"]["status"] == "ERROR"
        assert named["apab.session"]["attributes"]["apab.status"] == "error"


class TestTraceCorrelation:
    def test_audit_and_manifest_share_trace_id(self, config):
        provider = FakeProvider([_TOOL_CALL_RESPONSE, _FINAL_RESPONSE])
        orch = AgentOrchestrator(config, provider=provider)
        orch.run_to_completion("Compute a pattern")

        run_dir = orch.run_context.run_dir
        manifest = json.loads((run_dir / "manifest.json").read_text())
        audit = json.loads((run_dir / "audit.json").read_text())
        spans = _read_trace(run_dir)

        assert manifest["trace_id"]
        assert manifest["trace_id"] == spans[0]["trace_id"]
        assert all(e["trace_id"] == manifest["trace_id"] for e in audit)
        assert all(e["span_id"] for e in audit)

    def test_no_trace_ids_when_disabled(self, config):
        config.observability.enabled = False
        provider = FakeProvider([_TOOL_CALL_RESPONSE, _FINAL_RESPONSE])
        orch = AgentOrchestrator(config, provider=provider)
        orch.run_to_completion("Compute a pattern")

        run_dir = orch.run_context.run_dir
        manifest = json.loads((run_dir / "manifest.json").read_text())
        audit = json.loads((run_dir / "audit.json").read_text())

        assert manifest["trace_id"] == ""
        assert all("trace_id" not in e for e in audit)


class TestStrictCapture:
    def test_strict_spans_never_leak_values(self, config):
        config.observability.capture_mode = RedactionMode.strict
        provider = FakeProvider([_TOOL_CALL_RESPONSE, _FINAL_RESPONSE])
        orch = AgentOrchestrator(config, provider=provider)
        orch.run_to_completion("Compute a pattern")

        named = _by_name(_read_trace(orch.run_context.run_dir))
        attrs = named["apab.tool.pattern_compute"]["attributes"]

        assert "apab.tool.args_json" not in attrs
        assert "apab.tool.arg_keys" not in attrs
        assert "apab.tool.result_summary" not in attrs
        assert attrs["apab.tool.args_hash"]


class TestObservabilitySpec:
    def test_defaults(self):
        spec = ObservabilitySpec()
        assert spec.enabled is False
        assert spec.service_name == "apab"
        assert spec.trace_jsonl is True
        assert spec.capture_mode is None
        assert spec.set_global is False

    def test_yaml_roundtrip(self, config):
        dumped = config.model_dump()
        assert dumped["observability"]["enabled"] is True
        restored = ProjectConfig.model_validate(dumped)
        assert restored.observability.enabled is True
