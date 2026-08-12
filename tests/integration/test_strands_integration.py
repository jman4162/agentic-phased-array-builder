"""Real (un-mocked) Strands -> APAB MCP integration.

Spawns the actual server subprocess through the Strands MCPClient and
verifies the full observability loop: the server emits an apab.tool.*
span into APAB_TRACE_JSONL, parented on the trace this test injects via
TRACEPARENT. Requires strands-agents installed (pip install
'apab[strands]'); skipped otherwise. Run with: pytest -m integration
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

strands = pytest.importorskip("strands")

from apab.adapters.strands import apab_mcp_client  # noqa: E402

TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"

pytestmark = pytest.mark.integration


def test_tool_call_traced_end_to_end(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    client = apab_mcp_client(
        env={
            "APAB_OBSERVABILITY": "1",
            "APAB_TRACE_JSONL": str(trace_file),
            "TRACEPARENT": TRACEPARENT,
        }
    )

    with client:
        tools = client.list_tools_sync()
        assert len(tools) > 0

        result = client.call_tool_sync(
            tool_use_id="obs-1",
            name="emtool_list_adapters",
            arguments={},
        )
        assert result["status"] == "success"

    lines = [json.loads(x) for x in trace_file.read_text().splitlines()]
    tool_spans = [s for s in lines if s["name"] == "apab.tool.emtool_list_adapters"]
    assert tool_spans, f"no tool span in {[s['name'] for s in lines]}"
    span = tool_spans[0]
    assert span["trace_id"] == TRACE_ID
    assert span["attributes"]["apab.tool.status"] == "ok"
