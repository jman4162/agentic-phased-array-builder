"""Strands Agents adapter: use APAB's MCP tools from a Strands agent.

The adapter talks to APAB over its public MCP surface: it launches
``apab mcp serve --transport stdio`` as a subprocess and hands the
connection to Strands' ``MCPClient``. This keeps the integration on the
stable MCP protocol rather than APAB's in-process tool dispatcher,
whose FastMCP internals are private.

Requires the ``strands`` extra::

    pip install "apab[strands]"

Typical use::

    from strands import Agent
    from apab.adapters.strands import apab_mcp_client, apab_system_prompt

    client = apab_mcp_client(config_path="apab.yaml")
    with client:
        agent = Agent(
            model=...,  # any Strands model provider
            tools=client.list_tools_sync(),
            system_prompt=apab_system_prompt(),
        )
        agent("Design a 28 GHz 8x8 patch array and report its metrics.")
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strands.tools.mcp import MCPClient


def _require_strands() -> None:
    try:
        import strands  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The Strands adapter requires the strands-agents package. "
            "Install it with: pip install 'apab[strands]'"
        ) from exc


def _observability_env() -> dict[str, str]:
    """Env vars that carry tracing across the stdio process boundary.

    When tracing is active in this process, the current span context is
    injected as ``TRACEPARENT`` so the spawned server's tool spans parent
    onto the caller's trace. ``APAB_OBSERVABILITY`` and
    ``APAB_TRACE_JSONL`` pass through so the server actually emits spans.
    """
    env: dict[str, str] = {}
    from apab.observability.tracing import is_enabled

    if is_enabled():
        try:
            from opentelemetry.trace.propagation.tracecontext import (
                TraceContextTextMapPropagator,
            )

            carrier: dict[str, str] = {}
            TraceContextTextMapPropagator().inject(carrier)
            if "traceparent" in carrier:
                env["TRACEPARENT"] = carrier["traceparent"]
        except ImportError:
            pass
        env.setdefault("APAB_OBSERVABILITY", "1")
    for passthrough in ("APAB_OBSERVABILITY", "APAB_TRACE_JSONL", "TRACEPARENT"):
        if passthrough in os.environ:
            env.setdefault(passthrough, os.environ[passthrough])
    return env


def apab_server_parameters(
    config_path: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> Any:
    """Build MCP ``StdioServerParameters`` that launch APAB's server.

    The server runs in a subprocess with the same Python interpreter,
    so it sees the same installed apab and its tools. Observability env
    vars (``TRACEPARENT``, ``APAB_OBSERVABILITY``, ``APAB_TRACE_JSONL``)
    are forwarded automatically; *env* entries override them. The MCP
    client merges these on top of its safe default environment.
    """
    from mcp import StdioServerParameters

    args = ["-m", "apab.cli", "mcp", "serve", "--transport", "stdio"]
    if config_path is not None:
        args += ["--config", str(config_path)]
    merged = _observability_env()
    if env:
        merged.update(env)
    return StdioServerParameters(
        command=sys.executable, args=args, env=merged or None
    )


def apab_mcp_client(
    config_path: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> MCPClient:
    """Return a Strands ``MCPClient`` connected to APAB over stdio.

    Use it as a context manager; tools are available inside the block
    via ``client.list_tools_sync()``.
    """
    _require_strands()
    from mcp.client.stdio import stdio_client
    from strands.tools.mcp import MCPClient

    params = apab_server_parameters(config_path, env=env)
    return MCPClient(lambda: stdio_client(params))


def apab_system_prompt(config: dict[str, Any] | None = None) -> str:
    """APAB's own agent system prompt, reusable for a Strands agent."""
    from apab.agent.prompts import build_system_prompt

    return build_system_prompt(config)
