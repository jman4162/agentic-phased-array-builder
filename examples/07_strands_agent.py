#!/usr/bin/env python3
"""Example 07: APAB tools in a Strands agent.

Runs a Strands agent against APAB's MCP tools over stdio, with a local
Ollama model. Demonstrates that APAB's tool layer works from a foreign
agent framework, and that Strands' OpenTelemetry tracing captures the
tool calls.

Requirements:
    pip install "apab[strands,ollama]" "strands-agents[ollama,otel]"
    ollama serve   (with a tool-calling model pulled, e.g. qwen2.5-coder:14b)

Optional: export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 to
send traces to Jaeger (see lab/README.md); otherwise spans print to
the console.
"""

import os
import sys

MODEL = os.environ.get("APAB_OLLAMA_MODEL", "qwen2.5-coder:14b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def main() -> int:
    print("=" * 50)
    print("APAB Example 07: Strands Agent with APAB MCP Tools")
    print("=" * 50)

    try:
        from strands import Agent
        from strands.models.ollama import OllamaModel
        from strands.telemetry import StrandsTelemetry
    except ImportError as exc:
        print(f"Strands is not installed ({exc}).")
        print('Install with: pip install "apab[strands]" "strands-agents[ollama,otel]"')
        return 1

    from apab.adapters.strands import apab_mcp_client, apab_system_prompt

    # Strands emits OpenTelemetry spans for the agent loop, model calls,
    # and every APAB tool invocation.
    telemetry = StrandsTelemetry()
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        telemetry.setup_otlp_exporter()
        print(f"Tracing to {os.environ['OTEL_EXPORTER_OTLP_ENDPOINT']}")
    else:
        telemetry.setup_console_exporter()
        print("Tracing to console (set OTEL_EXPORTER_OTLP_ENDPOINT for Jaeger)")

    model = OllamaModel(host=OLLAMA_HOST, model_id=MODEL)
    client = apab_mcp_client()

    try:
        with client:
            tools = client.list_tools_sync()
            print(f"APAB MCP server exposes {len(tools)} tools\n")

            agent = Agent(
                model=model,
                tools=tools,
                system_prompt=apab_system_prompt(),
            )
            result = agent(
                "Compute the array pattern for an 8x8 phased array at "
                "28 GHz with half-wavelength spacing, then evaluate the "
                "system metrics and summarize directivity and sidelobe "
                "level."
            )
            print(f"\n{result}")
    except ConnectionError as exc:
        print(f"Cannot reach Ollama at {OLLAMA_HOST}: {exc}")
        print("Start it with: ollama serve")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
