"""MCP server factory and runner for APAB."""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from apab.core.config import load_config
from apab.core.schemas import ProjectConfig


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
    server._apab_config = config  # type: ignore[attr-defined]

    return server


def _get_server() -> FastMCP:
    """Return the singleton MCP server, registering all tools on first call."""
    global _server
    if _server is not None:
        return _server

    _server = FastMCP(
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
    import apab.mcp.tools_edgefem  # noqa: F401
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
    server.run(transport=transport)  # type: ignore[arg-type]
