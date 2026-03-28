"""apab optimize command — autonomous design optimization loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_optimize(args: argparse.Namespace) -> None:
    """Run the autonomous optimization loop."""
    from rich.console import Console
    from rich.panel import Panel

    from apab.agent.orchestrator import AgentOrchestrator
    from apab.core.config import load_config
    from apab.core.workspace import Workspace
    from apab.optimize.protocol import load_protocol
    from apab.optimize.runner import OptimizeRunner
    from apab.optimize.tracker import ResultsTracker

    console = Console()

    config = load_config(Path(args.config))
    workspace = Workspace(Path(config.project.workspace))
    workspace.ensure_dirs()

    protocol = load_protocol(Path(args.protocol))
    results_path = Path(config.project.workspace) / "optimize" / "results.tsv"
    tracker = ResultsTracker(results_path)

    orch = AgentOrchestrator(config, workspace)

    # Pre-flight check
    if hasattr(orch.provider, "ping"):
        ok, msg = orch.provider.ping()
        if not ok:
            console.print(f"[red bold]Provider check failed:[/red bold] {msg}")
            console.print("[dim]Run 'apab doctor' for diagnostics.[/dim]")
            sys.exit(1)

    max_exp = args.max_experiments

    console.print(Panel.fit(
        f"[bold]{config.project.name}[/bold]\n"
        f"Protocol: {args.protocol}\n"
        f"Metric: {protocol.metric} ({protocol.direction})\n"
        f"Constraints: "
        f"{', '.join(f'{c.metric} {c.op} {c.value}' for c in protocol.constraints) or 'none'}\n"
        f"Provider: {config.llm.provider} / {config.llm.model}\n"
        f"Max experiments: {max_exp}",
        title="APAB Optimize",
        border_style="cyan",
    ))
    console.print()

    runner = OptimizeRunner(orch, protocol, tracker, max_exp)

    try:
        for exp_num in range(1, max_exp + 1):
            is_baseline = tracker.best is None
            label = "baseline" if is_baseline else f"#{exp_num:03d}"
            console.print(f"[bold]Experiment {label}[/bold]")

            prompt = runner.build_prompt(is_baseline=is_baseline)

            with console.status("[bold cyan]Agent is thinking...[/bold cyan]"):
                metrics, description = runner.run_experiment(prompt)

            if not metrics:
                console.print("  [yellow]No metrics returned. Skipping.[/yellow]")
                continue

            status, reason = runner.evaluate(metrics)

            # Record
            desc = description or f"experiment {exp_num}"
            tracker.record(metrics, status, desc)

            # Display
            metric_val = metrics.get(protocol.metric, 0)
            status_icon = {
                "baseline": "[blue]\u2713 BASELINE[/blue]",
                "keep": "[green]\u2713 KEEP[/green]",
                "discard": "[red]\u2717 DISCARD[/red]",
            }.get(status, status)

            metric_parts = ", ".join(
                f"{k}={v:.1f}" for k, v in list(metrics.items())[:4]
            )
            console.print(f"  [dim]\u2190[/dim] {metric_parts}")
            console.print(
                f"  {status_icon} "
                f"({protocol.metric}={metric_val:.1f}"
                f"{', ' + reason if reason else ''})"
            )

            if status == "keep":
                console.print(
                    "  [green bold]New best![/green bold]"
                )
            console.print()

    except KeyboardInterrupt:
        console.print("\n[dim]Optimization stopped by user.[/dim]")

    # Summary
    best = tracker.best
    if best:
        console.print(Panel.fit(
            f"Best: {protocol.metric} = "
            f"{best.metrics.get(protocol.metric, 0):.1f}\n"
            f"Design: {best.description}\n"
            f"Experiment: #{best.experiment_id:03d}\n"
            f"Total experiments: {len(tracker.results)}",
            title="[bold green]Optimization Complete[/bold green]",
            border_style="green",
        ))
    console.print(f"[dim]Results saved to {results_path}[/dim]")
