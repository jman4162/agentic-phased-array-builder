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


def apab_server_parameters(
    config_path: str | Path | None = None,
) -> Any:
    """Build MCP ``StdioServerParameters`` that launch APAB's server.

    The server runs in a subprocess with the same Python interpreter,
    so it sees the same installed apab and its tools.
    """
    from mcp import StdioServerParameters

    args = ["-m", "apab.cli", "mcp", "serve", "--transport", "stdio"]
    if config_path is not None:
        args += ["--config", str(config_path)]
    return StdioServerParameters(command=sys.executable, args=args)


def apab_mcp_client(
    config_path: str | Path | None = None,
) -> MCPClient:
    """Return a Strands ``MCPClient`` connected to APAB over stdio.

    Use it as a context manager; tools are available inside the block
    via ``client.list_tools_sync()``.
    """
    _require_strands()
    from mcp.client.stdio import stdio_client
    from strands.tools.mcp import MCPClient

    params = apab_server_parameters(config_path)
    return MCPClient(lambda: stdio_client(params))


def apab_system_prompt(config: dict[str, Any] | None = None) -> str:
    """APAB's own agent system prompt, reusable for a Strands agent."""
    from apab.agent.prompts import build_system_prompt

    return build_system_prompt(config)
