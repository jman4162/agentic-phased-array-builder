"""Shared rich rendering for agent-loop progress events."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import Console


def truncate(text: str, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def make_event_renderer(
    console: Console,
    final_title: str = "Result",
    pad_final: bool = False,
) -> tuple[Callable[[str, dict[str, Any]], None], Callable[[], None]]:
    """Build an ``on_event`` callback rendering agent progress to *console*.

    Returns ``(on_event, stop)``. Call ``stop()`` in a ``finally`` block so
    the thinking spinner is cleared even if the loop raises.
    """
    from rich.panel import Panel

    state: dict[str, Any] = {"status": None}

    def stop() -> None:
        if state["status"] is not None:
            state["status"].stop()
            state["status"] = None

    def on_event(name: str, payload: dict[str, Any]) -> None:
        if name == "turn_start":
            stop()
            status = console.status("[bold cyan]Agent is thinking...[/bold cyan]")
            status.start()
            state["status"] = status
        elif name == "assistant_message":
            stop()
            if pad_final:
                console.print()
            console.print(Panel(
                payload["content"],
                title=f"[bold green]{final_title}[/bold green]",
                border_style="green",
                padding=(1, 2),
            ))
            if pad_final:
                console.print()
        elif name == "tool_call":
            stop()
            args = payload.get("arguments", {})
            console.print(
                f"  [dim]→ Calling[/dim] [bold]{payload['name']}[/bold]"
                f"[dim]({', '.join(f'{k}=' for k in args)})[/dim]"
            )
        elif name == "tool_result":
            console.print(
                f"  [dim]← {payload['tool']}:[/dim] "
                f"{truncate(payload['result'])}"
            )
        elif name == "max_turns":
            stop()
            console.print(
                "[yellow]Reached maximum turns. "
                "Response may be incomplete.[/yellow]"
            )

    return on_event, stop
