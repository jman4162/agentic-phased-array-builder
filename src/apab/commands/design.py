"""apab design command — interactive agent session with rich output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from apab.commands._render import make_event_renderer
from apab.commands._render import truncate as _truncate  # noqa: F401  (re-export for tests)


def cmd_design(args: argparse.Namespace) -> None:
    """Start an interactive design session with the APAB agent."""
    from rich.console import Console
    from rich.panel import Panel

    from apab.agent.orchestrator import AgentOrchestrator
    from apab.core.config import load_config
    from apab.core.workspace import Workspace

    console = Console()

    config = load_config(Path(args.config))
    workspace = Workspace(Path(config.project.workspace))
    workspace.ensure_dirs()

    orch = AgentOrchestrator(config, workspace)

    # Pre-flight: check provider connectivity
    if hasattr(orch.provider, "ping"):
        ok, msg = orch.provider.ping()
        if not ok:
            console.print(f"[red bold]Provider check failed:[/red bold] {msg}")
            console.print("[dim]Run 'apab doctor' for full diagnostics.[/dim]")
            sys.exit(1)

    console.print(Panel.fit(
        f"[bold]{config.project.name}[/bold]\n"
        f"Provider: {config.llm.provider} / {config.llm.model}\n"
        f"Type your request. Commands: exit, quit, Ctrl+D",
        title="APAB Design Session",
        border_style="cyan",
    ))
    console.print()

    try:
        while True:
            try:
                user_input = console.input("[bold cyan]you>[/bold cyan] ").strip()
            except EOFError:
                console.print("\n[dim]Session ended.[/dim]")
                break

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "q"}:
                console.print("[dim]Session ended.[/dim]")
                break

            on_event, stop_spinner = make_event_renderer(
                console, final_title="Assistant", pad_final=True,
            )
            try:
                orch.run_to_completion(user_input, on_event=on_event)
            finally:
                stop_spinner()

    except KeyboardInterrupt:
        console.print("\n[dim]Session interrupted.[/dim]")
