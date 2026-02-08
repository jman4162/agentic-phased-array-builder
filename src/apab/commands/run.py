"""apab run command — non-interactive workflow execution."""

from __future__ import annotations

import argparse
from pathlib import Path


def cmd_run(args: argparse.Namespace) -> None:
    """Run a workflow non-interactively from config."""
    from apab.agent.orchestrator import AgentOrchestrator
    from apab.core.config import load_config
    from apab.core.workspace import Workspace

    config = load_config(Path(args.config))
    workspace = Workspace(Path(config.project.workspace))
    workspace.ensure_dirs()

    # Build a prompt from the config's array/sweep/unit_cell settings
    prompt_parts = [
        f"Analyse a phased-array antenna for project '{config.project.name}'.",
    ]

    if config.array:
        arr = config.array
        prompt_parts.append(
            f"Array: {arr.size[0]}×{arr.size[1]}, "
            f"spacing {arr.spacing_m[0]*1000:.1f}mm × {arr.spacing_m[1]*1000:.1f}mm, "
            f"taper '{arr.taper}', "
            f"steer θ={arr.steer.theta_deg}° φ={arr.steer.phi_deg}°."
        )

    if config.sweep:
        sw = config.sweep
        prompt_parts.append(
            f"Sweep: {sw.freq_hz.start/1e9:.1f}–{sw.freq_hz.stop/1e9:.1f} GHz "
            f"({sw.freq_hz.n} points)."
        )

    prompt_parts.append(
        "Compute the array pattern, evaluate system metrics, "
        "and provide a summary with key metrics."
    )

    prompt = " ".join(prompt_parts)

    print(f"[apab run] Project: {config.project.name}")
    print(f"[apab run] Provider: {config.llm.provider} / {config.llm.model}")
    print(f"[apab run] Prompt: {prompt[:120]}...")
    print()

    orch = AgentOrchestrator(config, workspace)
    result = orch.run_to_completion(prompt)
    print(result)
