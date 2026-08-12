"""MCP server factory and runner for APAB."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from mcp.server.fastmcp import FastMCP

from apab.core.config import load_config
from apab.core.schemas import ObservabilitySpec, ProjectConfig, RedactionMode
from apab.observability.redaction import capture_args, capture_text
from apab.observability.tracing import (
    init_observability,
    init_remote_parent_from_env,
    set_span_error,
    shutdown_observability,
    span,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcp.types import ContentBlock

logger = logging.getLogger(__name__)


class _InstrumentedFastMCP(FastMCP):
    """FastMCP that opens an ``apab.tool.<name>`` span around every call.

    ``call_tool`` is the single public choke point every registered tool
    passes through, so instrumenting here covers all tool modules —
    current and future — without touching their registrations. Span
    naming and attributes match the agent orchestrator's, so served and
    in-process tool calls look identical on a trace.
    """

    def _capture_mode(self) -> RedactionMode:
        config = getattr(self, "_apab_config", None)
        if config is not None and config.observability.capture_mode is not None:
            return RedactionMode(config.observability.capture_mode)
        return RedactionMode.metadata_only

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> Sequence[ContentBlock] | dict[str, Any]:
        capture_mode = self._capture_mode()
        captured = capture_args(arguments, capture_mode)
        attrs: dict[str, Any] = {
            "apab.tool.name": name,
            "apab.tool.args_hash": captured["args_hash"],
        }
        if "args_json" in captured:
            attrs["apab.tool.args_json"] = captured["args_json"]
        if "arg_keys" in captured:
            attrs["apab.tool.arg_keys"] = captured["arg_keys"]

        with span(f"apab.tool.{name}", **attrs) as s:
            try:
                result = await super().call_tool(name, arguments)
            except Exception as exc:
                set_span_error(s, exc)
                raise
            status = "ok"
            if isinstance(result, dict) and (
                "error" in result or result.get("status") == "failed"
            ):
                status = "error"
            s.set_attribute("apab.tool.status", status)
            if isinstance(result, dict):
                summary = capture_text(str(result), capture_mode)
                if summary is not None:
                    s.set_attribute("apab.tool.result_summary", summary)
            return result


def create_server(
    config: ProjectConfig | None = None,
    config_path: str | Path | None = None,
) -> FastMCP:
    """Create and return a fully-configured :class:`FastMCP` server.

    All APAB tool modules are imported for their side-effect of registering
    tools on the shared ``mcp`` instance.

    Parameters
    ----------
    config:
        An already-loaded project configuration.  Takes precedence over
        *config_path*.
    config_path:
        Path to an ``apab.yaml`` file.  Ignored when *config* is given.
    """
    if config is None and config_path is not None:
        config = load_config(Path(config_path))

    # Store config on the server instance so tools can access it via context.
    server = _get_server()
    setattr(server, "_apab_config", config)

    # Server-side observability: without this a served process emits no
    # spans at all, so agent-driven runs are invisible. Env-gated as
    # everywhere else (APAB_OBSERVABILITY=1 activates the default spec);
    # init is idempotent, so embedding callers may have gone first.
    spec = config.observability if config is not None else ObservabilitySpec()
    init_observability(spec)
    # Adopt a caller's trace via TRACEPARENT so client and server side of
    # the stdio transport share one trace.
    init_remote_parent_from_env()

    return server


def _get_server() -> FastMCP:
    """Return the singleton MCP server, registering all tools on first call."""
    global _server
    if _server is not None:
        return _server

    _server = _InstrumentedFastMCP(
        name="apab",
        instructions=(
            "APAB — Agentic Phased Array Builder. "
            "Tools for phased-array antenna design: unit-cell simulation, "
            "mutual coupling, array patterns, system-level analysis, "
            "trade studies, import/export, and plotting."
        ),
    )

    # Import tool modules to trigger @mcp.tool() registrations.
    import apab.mcp.tools_array  # noqa: F401

    try:
        import apab.mcp.tools_edgefem  # noqa: F401
    except ImportError:
        logger.info(
            "EdgeFEM tools not available (edgefem not installed). "
            "Install with: pip install apab[edgefem]"
        )

    # Resources register the same way tools do, by import side effect. Without
    # the resources import the apab:// resources only existed when something
    # else happened to import the module.
    import apab.mcp.resources  # noqa: F401
    import apab.mcp.tools_emtool  # noqa: F401
    import apab.mcp.tools_io  # noqa: F401
    import apab.mcp.tools_plot  # noqa: F401
    import apab.mcp.tools_system  # noqa: F401

    return _server


_server: FastMCP | None = None


def get_mcp() -> FastMCP:
    """Return the shared MCP server instance (creating it if needed)."""
    return _get_server()


def run_server(transport: str = "stdio") -> None:
    """Start the MCP server with the given transport.

    Parameters
    ----------
    transport:
        One of ``"stdio"``, ``"sse"``, or ``"streamable-http"``.
    """
    server = _get_server()
    try:
        server.run(transport=cast(Any, transport))
    finally:
        shutdown_observability()
