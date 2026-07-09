"""apab run command — non-interactive workflow execution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from apab.commands._render import make_event_renderer
from apab.commands._render import truncate as _truncate


def cmd_run(args: argparse.Namespace) -> None:
    """Run a workflow non-interactively from config."""
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

    # Build a prompt from the config's array/sweep/unit_cell settings
    prompt_parts = [
        f"Analyse a phased-array antenna for project '{config.project.name}'.",
    ]

    if config.array:
        arr = config.array
        prompt_parts.append(
            f"Array: {arr.size[0]}×{arr.size[1]}, "
            f"spacing {arr.spacing_m[0]*1000:.1f}mm "
            f"× {arr.spacing_m[1]*1000:.1f}mm, "
            f"taper '{arr.taper}', "
            f"steer θ={arr.steer.theta_deg}° φ={arr.steer.phi_deg}°."
        )

    if config.sweep:
        sw = config.sweep
        prompt_parts.append(
            f"Sweep: {sw.freq_hz.start/1e9:.1f}"
            f"–{sw.freq_hz.stop/1e9:.1f} GHz "
            f"({sw.freq_hz.n} points)."
        )
    elif config.array:
        # Derive design frequency from half-wave spacing
        from scipy.constants import c as speed_of_light

        freq_ghz = speed_of_light / (2 * config.array.spacing_m[0]) / 1e9
        prompt_parts.append(
            f"Design frequency: {freq_ghz:.1f} GHz "
            f"(half-wave spacing)."
        )

    prompt_parts.append(
        "Compute the array pattern, evaluate system metrics, "
        "and provide a summary with key metrics."
    )

    prompt = " ".join(prompt_parts)

    console.print(Panel.fit(
        f"[bold]{config.project.name}[/bold]\n"
        f"Provider: {config.llm.provider} / {config.llm.model}\n"
        f"Prompt: {_truncate(prompt, 120)}",
        title="APAB Run",
        border_style="cyan",
    ))
    console.print()

    on_event, stop_spinner = make_event_renderer(console, final_title="Result")
    try:
        orch.run_to_completion(prompt, on_event=on_event)
    finally:
        stop_spinner()
