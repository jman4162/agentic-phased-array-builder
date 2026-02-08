"""End-to-end: Agent loop with mock LLM → verify tools called."""

from __future__ import annotations

import json
from typing import Any

import pytest

pytestmark = pytest.mark.integration


class ScriptedProvider:
    """A scripted LLM provider that returns predetermined responses."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self._script = list(script)
        self._idx = 0
        self.call_history: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "scripted"

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
        self.call_history.append({
            "messages_count": len(messages),
            "has_tools": tools is not None,
        })
        if self._idx < len(self._script):
            response = self._script[self._idx]
        else:
            response = {"role": "assistant", "content": "Done.", "tool_calls": None}
        self._idx += 1
        return response


class TestAgentLoopE2E:
    def test_pattern_compute_workflow(self, tmp_path):
        from apab.agent.orchestrator import AgentOrchestrator
        from apab.core.schemas import ProjectConfig, ProjectMeta

        config = ProjectConfig(
            project=ProjectMeta(name="test", workspace=str(tmp_path / "workspace")),
        )

        provider = ScriptedProvider([
            # Turn 1: call pattern_compute tool
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "name": "pattern_compute",
                        "arguments": {
                            "nx": 4,
                            "ny": 4,
                            "dx_m": 0.005,
                            "dy_m": 0.005,
                            "freq_hz": 10e9,
                        },
                    },
                ],
            },
            # Turn 2: call system_evaluate tool
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "name": "system_evaluate",
                        "arguments": {
                            "nx": 4,
                            "ny": 4,
                            "dx_m": 0.005,
                            "dy_m": 0.005,
                            "freq_hz": 10e9,
                            "bandwidth_hz": 100e6,
                            "range_m": 1000.0,
                            "tx_power_w_per_elem": 0.1,
                        },
                    },
                ],
            },
            # Turn 3: final response
            {
                "role": "assistant",
                "content": "The 4x4 array has a directivity of approximately 17 dBi.",
                "tool_calls": None,
            },
        ])

        orch = AgentOrchestrator(config, provider=provider)
        result = orch.run_to_completion("Design a 4x4 array at 10 GHz")

        assert "17" in result or "directivity" in result.lower()
        assert len(provider.call_history) == 3

        # Verify tool results were fed back into messages
        tool_msgs = [m for m in orch.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2

        # First tool call result should contain directivity
        first_result = json.loads(tool_msgs[0]["content"])
        assert "directivity_dbi" in first_result

    def test_max_turns_safety(self, tmp_path):
        from apab.agent.orchestrator import AgentOrchestrator
        from apab.core.schemas import ProjectConfig, ProjectMeta

        config = ProjectConfig(
            project=ProjectMeta(name="test", workspace=str(tmp_path / "workspace")),
        )

        # Provider that always returns tool calls (never finishes)
        infinite_script = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "name": "pattern_compute",
                        "arguments": {
                            "nx": 4, "ny": 4,
                            "dx_m": 0.005, "dy_m": 0.005,
                            "freq_hz": 10e9,
                        },
                    },
                ],
            }
        ] * 10

        provider = ScriptedProvider(infinite_script)
        orch = AgentOrchestrator(config, provider=provider)
        result = orch.run_to_completion("loop", max_turns=3)

        assert "maximum" in result.lower()
        assert len(provider.call_history) == 3
