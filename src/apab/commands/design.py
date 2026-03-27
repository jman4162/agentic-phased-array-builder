"""apab design command — interactive agent session with rich output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _truncate(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


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

    max_turns = 20

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

            # Unrolled agent loop with visibility
            orch.start_session(user_input)

            for _turn in range(max_turns):
                with console.status("[bold cyan]Agent is thinking...[/bold cyan]"):
                    response = orch.step()

                tool_calls = response.get("tool_calls")

                if not tool_calls:
                    # Final text response
                    content = response.get("content") or ""
                    console.print()
                    console.print(Panel(
                        content,
                        title="[bold green]Assistant[/bold green]",
                        border_style="green",
                        padding=(1, 2),
                    ))
                    console.print()
                    break

                # Show tool calls
                for tc in tool_calls:
                    console.print(
                        f"  [dim]\u2192 Calling[/dim] [bold]{tc['name']}[/bold]"
                        f"[dim]({', '.join(f'{k}=' for k in tc.get('arguments', {}))})[/dim]"
                    )

                results = orch.execute_tool_calls(response)

                for r in results:
                    console.print(
                        f"  [dim]\u2190 {r['tool']}:[/dim] {_truncate(r['result'])}"
                    )
            else:
                console.print(
                    "[yellow]Reached maximum turns. Response may be incomplete.[/yellow]"
                )

    except KeyboardInterrupt:
        console.print("\n[dim]Session interrupted.[/dim]")
