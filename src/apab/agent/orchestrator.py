"""Agent orchestrator: LLM ↔ tool-calling loop."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from apab.agent.prompts import build_system_prompt
from apab.agent.provider_registry import LLMProvider, get_provider
from apab.agent.tool_dispatch import ToolDispatcher
from apab.core.schemas import ProjectConfig, RedactionMode
from apab.core.workspace import RunContext, Workspace

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Drives the agentic LLM ↔ tool loop.

    Parameters
    ----------
    config:
        Project configuration.
    workspace:
        Workspace manager for run bundles.
    provider:
        An already-instantiated LLM provider.  If ``None`` one is
        created from *config.llm*.
    """

    def __init__(
        self,
        config: ProjectConfig,
        workspace: Workspace | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        self.config = config
        self.workspace = workspace or Workspace(Path(config.project.workspace))
        self.provider = provider or get_provider(
            config.llm.provider,
            model=config.llm.model,
            base_url=config.llm.base_url,
        )
        mode = config.llm.redaction_mode
        self.dispatcher = ToolDispatcher(
            redaction_mode=mode.value if hasattr(mode, "value") else str(mode),
        )
        self._messages: list[dict[str, Any]] = []
        self._run_ctx: RunContext | None = None

    # ── session lifecycle ─────────────────────────────────────────────

    def start_session(self, user_request: str) -> RunContext:
        """Begin a new agent session and return a :class:`RunContext`."""
        self.workspace.ensure_dirs()
        self._run_ctx = self.workspace.new_run()

        system_prompt = build_system_prompt(self.config.model_dump())
        self._messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_request},
        ]

        logger.info(
            "Session started: run_id=%s, provider=%s",
            self._run_ctx.run_id,
            self.provider.name,
        )
        return self._run_ctx

    def step(self) -> dict[str, Any]:
        """Execute one LLM turn and return the raw response dict."""
        tools = self.dispatcher.get_tool_schemas()
        response = self.provider.chat(
            messages=self._messages,
            tools=tools,
        )

        # Apply redaction before logging
        self._log_egress(response)

        # Append assistant message
        self._messages.append({
            "role": "assistant",
            "content": response.get("content"),
            "tool_calls": response.get("tool_calls"),
        })

        return response

    def execute_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute all tool calls from a response and append results to messages."""
        tool_calls = response.get("tool_calls") or []
        results = []

        for tc in tool_calls:
            tool_name = tc["name"]
            arguments = tc.get("arguments", {})

            logger.info("Calling tool: %s(%s)", tool_name, json.dumps(arguments, default=str))
            result_str = self.dispatcher.dispatch(tool_name, arguments)

            result_msg = {
                "role": "tool",
                "content": result_str,
                "name": tool_name,
            }
            self._messages.append(result_msg)
            results.append({"tool": tool_name, "result": result_str})

        return results

    def run_to_completion(
        self,
        user_request: str,
        max_turns: int = 20,
    ) -> str:
        """Run the full agentic loop and return the final text response.

        Parameters
        ----------
        user_request:
            The user's natural-language request.
        max_turns:
            Maximum number of LLM turns before forcibly stopping.
        """
        self.start_session(user_request)

        try:
            for turn in range(max_turns):
                response = self.step()
                tool_calls = response.get("tool_calls")

                if not tool_calls:
                    # No tool calls => final response
                    return response.get("content") or ""

                self.execute_tool_calls(response)

            return (
                "Reached maximum number of turns. "
                "The last response may be incomplete."
            )
        finally:
            self._persist_audit_log()

    # ── internals ─────────────────────────────────────────────────────

    def _log_egress(self, response: dict[str, Any]) -> None:
        """Log outgoing LLM interactions with redaction applied."""
        mode = self.config.llm.redaction_mode

        if mode == RedactionMode.none:
            logger.debug("LLM response: %s", json.dumps(response, default=str)[:500])
        elif mode == RedactionMode.metadata_only:
            has_tools = bool(response.get("tool_calls"))
            logger.debug(
                "LLM response: has_content=%s, has_tool_calls=%s",
                bool(response.get("content")),
                has_tools,
            )
        elif mode == RedactionMode.strict:
            logger.debug("LLM response: [REDACTED]")

    def _persist_audit_log(self) -> None:
        """Save the audit log to the run directory if a session is active."""
        if self._run_ctx is not None and self.dispatcher.audit_log:
            audit_path = self._run_ctx.run_dir / "audit.json"
            try:
                self.dispatcher.save_audit_log(audit_path)
                logger.info("Audit log saved to %s", audit_path)
            except Exception:
                logger.exception("Failed to save audit log")

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return a copy of the conversation messages."""
        return list(self._messages)

    @property
    def run_context(self) -> RunContext | None:
        """Return the current run context, or ``None`` if not started."""
        return self._run_ctx
