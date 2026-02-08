"""apab mcp serve command — start the MCP server."""

from __future__ import annotations

import argparse
from pathlib import Path


def cmd_mcp_serve(args: argparse.Namespace) -> None:
    """Start the APAB MCP server."""
    from apab.core.config import load_config
    from apab.mcp.server import create_server, run_server

    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
        create_server(config=config)
    else:
        create_server()

    transport = args.transport
    if transport == "http":
        transport = "streamable-http"

    print(f"[apab mcp serve] Starting MCP server (transport={transport})")
    run_server(transport=transport)
