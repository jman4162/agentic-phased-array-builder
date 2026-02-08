"""apab design command — interactive agent session."""

from __future__ import annotations

import argparse
from pathlib import Path


def cmd_design(args: argparse.Namespace) -> None:
    """Start an interactive design session with the APAB agent."""
    from apab.agent.orchestrator import AgentOrchestrator
    from apab.core.config import load_config
    from apab.core.workspace import Workspace

    config = load_config(Path(args.config))
    workspace = Workspace(Path(config.project.workspace))
    workspace.ensure_dirs()

    orch = AgentOrchestrator(config, workspace)

    print(f"[apab design] Project: {config.project.name}")
    print(f"[apab design] Provider: {config.llm.provider} / {config.llm.model}")
    print("[apab design] Type your request (Ctrl+D to exit):\n")

    try:
        while True:
            try:
                user_input = input("you> ").strip()
            except EOFError:
                print("\n[apab design] Session ended.")
                break

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "q"}:
                print("[apab design] Session ended.")
                break

            result = orch.run_to_completion(user_input)
            print(f"\nassistant> {result}\n")

    except KeyboardInterrupt:
        print("\n[apab design] Session interrupted.")
