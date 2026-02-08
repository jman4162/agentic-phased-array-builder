"""Tool dispatch: extract MCP tool schemas and execute tool calls."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Module-level executor reused across all dispatches to avoid per-call overhead.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


class ToolDispatcher:
    """Bridges the LLM's tool calls to the MCP server's tool implementations."""

    def __init__(self, redaction_mode: str = "none") -> None:
        self._audit_log: list[dict[str, Any]] = []
        self._redaction_mode = redaction_mode

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Extract tool schemas from the MCP server in LLM-friendly format."""
        from apab.mcp.server import get_mcp

        server = get_mcp()
        tools = server._tool_manager._tools

        schemas = []
        for name, tool in tools.items():
            schema = {
                "name": name,
                "description": tool.description or "",
                "inputSchema": tool.parameters if hasattr(tool, "parameters") else {},
            }
            schemas.append(schema)
        return schemas

    def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call and return the JSON result string.

        Uses asyncio to call the async tool function from a sync context.
        """
        from apab.mcp.server import get_mcp

        server = get_mcp()
        tools = server._tool_manager._tools

        if tool_name not in tools:
            error = {"error": f"Unknown tool: {tool_name}"}
            self._log_call(tool_name, arguments, error)
            return json.dumps(error)

        tool = tools[tool_name]
        fn = tool.fn

        try:
            # Run the async tool function
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # We're inside an async context — use the shared executor
                result = _executor.submit(
                    asyncio.run, fn(**arguments)
                ).result()
            else:
                result = asyncio.run(fn(**arguments))

            self._log_call(tool_name, arguments, result)
            return json.dumps(result, default=str)

        except Exception as e:
            error = {"error": str(e), "tool": tool_name}
            self._log_call(tool_name, arguments, error)
            logger.exception("Tool dispatch failed: %s", tool_name)
            return json.dumps(error)

    def _log_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> None:
        """Append an entry to the audit log, respecting redaction mode."""
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
        }
        if self._redaction_mode == "strict":
            entry["arguments"] = "[REDACTED]"
            entry["result_summary"] = "[REDACTED]"
        elif self._redaction_mode == "metadata_only":
            entry["arguments"] = list(arguments.keys())
            entry["result_summary"] = _summarise(result)
        else:
            entry["arguments"] = arguments
            entry["result_summary"] = _summarise(result)

        self._audit_log.append(entry)

    def save_audit_log(self, path: str | Any) -> None:
        """Persist the audit log to a JSON file."""
        from pathlib import Path

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self._audit_log, indent=2, default=str))

    @property
    def audit_log(self) -> list[dict[str, Any]]:
        """Return a copy of the audit log."""
        return list(self._audit_log)


def _summarise(result: Any, max_len: int = 200) -> str:
    """Create a short summary of a result for the audit log."""
    text = str(result)
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text
