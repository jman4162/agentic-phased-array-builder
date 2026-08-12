"""Tests for server-side MCP observability: tool spans, traceparent, exporters."""

from __future__ import annotations

import json
from typing import Any

import pytest

from apab.core.schemas import ObservabilitySpec
from apab.mcp.server import get_mcp
from apab.observability import init_observability, shutdown_observability, span
from apab.observability.tracing import init_remote_parent_from_env

TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
TRACE_ID = "0af7651916cd43dd8448eb211c80319c"


@pytest.fixture(autouse=True)
def _clean_tracing_state(monkeypatch):
    monkeypatch.delenv("APAB_OBSERVABILITY", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("APAB_TRACE_JSONL", raising=False)
    monkeypatch.delenv("TRACEPARENT", raising=False)
    shutdown_observability()
    yield
    shutdown_observability()


def _in_memory() -> tuple[Any, list[Any]]:
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    return exporter, [SimpleSpanProcessor(exporter)]


async def _echo_tool_registered() -> str:
    """Register a tiny tool on the singleton once; return its name."""
    server = get_mcp()

    async def obs_test_echo(text: str) -> dict[str, str]:
        """Echo back the input (observability test fixture)."""
        return {"echo": text, "status": "completed"}

    existing = {t.name for t in await server.list_tools()}
    if "obs_test_echo" not in existing:
        server.tool()(obs_test_echo)
    return "obs_test_echo"


class TestToolSpans:
    async def test_call_tool_opens_named_span(self):
        exporter, processors = _in_memory()
        init_observability(ObservabilitySpec(enabled=True), extra_processors=processors)
        name = await _echo_tool_registered()

        server = get_mcp()
        await server.call_tool(name, {"text": "hi"})

        spans = exporter.get_finished_spans()
        tool_spans = [s for s in spans if s.name == f"apab.tool.{name}"]
        assert len(tool_spans) == 1
        attrs = dict(tool_spans[0].attributes or {})
        assert attrs["apab.tool.name"] == name
        assert "apab.tool.args_hash" in attrs
        assert attrs["apab.tool.status"] == "ok"

    async def test_disabled_tracing_is_noop(self):
        name = await _echo_tool_registered()
        server = get_mcp()
        result = await server.call_tool(name, {"text": "hi"})
        assert result is not None


class TestTraceparent:
    def test_remote_parent_adopted_for_root_span(self, monkeypatch):
        exporter, processors = _in_memory()
        init_observability(ObservabilitySpec(enabled=True), extra_processors=processors)
        monkeypatch.setenv("TRACEPARENT", TRACEPARENT)
        assert init_remote_parent_from_env() is True

        with span("apab.test.root"):
            pass

        (finished,) = exporter.get_finished_spans()
        assert format(finished.context.trace_id, "032x") == TRACE_ID

    def test_nested_span_parents_locally(self, monkeypatch):
        exporter, processors = _in_memory()
        init_observability(ObservabilitySpec(enabled=True), extra_processors=processors)
        monkeypatch.setenv("TRACEPARENT", TRACEPARENT)
        init_remote_parent_from_env()

        with span("outer"):
            with span("inner"):
                pass

        by_name = {s.name: s for s in exporter.get_finished_spans()}
        outer, inner = by_name["outer"], by_name["inner"]
        # Both live on the remote trace; inner parents on outer, not the
        # remote span directly.
        assert format(inner.context.trace_id, "032x") == TRACE_ID
        assert inner.parent is not None
        assert inner.parent.span_id == outer.context.span_id

    def test_no_env_no_adoption(self):
        assert init_remote_parent_from_env() is False


class TestTraceJsonlFallback:
    def test_env_path_used_without_run_ctx(self, tmp_path, monkeypatch):
        trace_path = tmp_path / "trace.jsonl"
        monkeypatch.setenv("APAB_TRACE_JSONL", str(trace_path))
        init_observability(ObservabilitySpec(enabled=True))

        with span("apab.test.file"):
            pass
        shutdown_observability()

        lines = [json.loads(x) for x in trace_path.read_text().splitlines()]
        assert any(s["name"] == "apab.test.file" for s in lines)

    def test_no_env_no_file(self, tmp_path):
        init_observability(ObservabilitySpec(enabled=True))
        with span("apab.test.nofile"):
            pass
        shutdown_observability()
        assert not (tmp_path / "trace.jsonl").exists()
