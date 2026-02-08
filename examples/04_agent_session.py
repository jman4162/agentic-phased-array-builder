#!/usr/bin/env python3
"""Example 04: Programmatic agent session.

Demonstrates using the AgentOrchestrator API directly with a fake
(scripted) provider for testing, without requiring an actual LLM.
"""

import json
from pathlib import Path
from typing import Any

from apab.agent.orchestrator import AgentOrchestrator
from apab.core.schemas import ProjectConfig, ProjectMeta


class DemoProvider:
    """A demo provider that scripts a simple tool-calling workflow."""

    def __init__(self) -> None:
        self._calls = 0

    @property
    def name(self) -> str:
        return "demo"

    def supports_tool_calling(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return False

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._calls += 1

        if self._calls == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "name": "pattern_compute",
                        "arguments": {
                            "nx": 8,
                            "ny": 8,
                            "dx_m": 0.005,
                            "dy_m": 0.005,
                            "freq_hz": 28e9,
                        },
                    }
                ],
            }

        # After receiving the tool result, return final text
        tool_results = [m for m in messages if m.get("role") == "tool"]
        if tool_results:
            last_result = json.loads(tool_results[-1]["content"])
            directivity = last_result.get("directivity_dbi", "?")
            return {
                "role": "assistant",
                "content": (
                    f"I computed the pattern for an 8×8 array at 28 GHz. "
                    f"The directivity is {directivity:.1f} dBi."
                ),
                "tool_calls": None,
            }

        return {
            "role": "assistant",
            "content": "Analysis complete.",
            "tool_calls": None,
        }


def main() -> None:
    print("=" * 50)
    print("APAB Example 04: Programmatic Agent Session")
    print("=" * 50)

    config = ProjectConfig(
        project=ProjectMeta(name="demo_project", workspace="/tmp/apab_demo"),
    )

    provider = DemoProvider()
    orch = AgentOrchestrator(config, provider=provider)

    result = orch.run_to_completion(
        "Design a 28 GHz phased array with 8×8 elements."
    )

    print(f"\nFinal response: {result}")
    print(f"Turns taken: {len([m for m in orch.messages if m.get('role') == 'assistant'])}")
    print(f"Tools called: {len([m for m in orch.messages if m.get('role') == 'tool'])}")
    print(f"Audit log entries: {len(orch.dispatcher.audit_log)}")


if __name__ == "__main__":
    main()
