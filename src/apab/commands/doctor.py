"""apab doctor command — environment health checks."""

from __future__ import annotations

import argparse
import sys
import time


def _check_python() -> tuple[str, str, str]:
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= (3, 10):
        return "pass", f"Python {version_str}", ""
    return "fail", f"Python {version_str}", "Requires Python >= 3.10"


def _check_core_deps() -> tuple[str, str, str]:
    missing = []
    for mod in ("numpy", "scipy", "pydantic", "yaml", "matplotlib", "mcp"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        return "fail", "Core dependencies", f"Missing: {', '.join(missing)}"
    return "pass", "Core dependencies", "All importable"


def _check_edgefem() -> tuple[str, str, str]:
    try:
        __import__("edgefem")
        return "pass", "EdgeFEM", "Installed"
    except ImportError:
        return "warn", "EdgeFEM", "Not installed (optional — needed for full-wave simulation)"


def _check_ollama_package() -> tuple[str, str, str]:
    try:
        __import__("ollama")
        return "pass", "Ollama package", "Installed"
    except ImportError:
        return "fail", "Ollama package", "pip install apab[ollama]"


def _check_ollama_server(base_url: str) -> tuple[str, str, str]:
    try:
        import httpx

        resp = httpx.get(f"{base_url}/api/tags", timeout=httpx.Timeout(3.0))
        resp.raise_for_status()
        return "pass", "Ollama server", f"Reachable at {base_url}"
    except Exception:
        return "fail", "Ollama server", f"Not reachable at {base_url}. Run: ollama serve"


def _check_model(base_url: str, model: str) -> tuple[str, str, str]:
    try:
        import httpx

        resp = httpx.get(f"{base_url}/api/tags", timeout=httpx.Timeout(3.0))
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        found = any(
            m == model or m.startswith(f"{model}:")
            for m in models
        )
        if found:
            return "pass", f"Model '{model}'", "Available"
        available = ", ".join(models[:5]) or "(none)"
        return (
            "fail",
            f"Model '{model}'",
            f"Not found. Available: {available}. "
            f"Run: ollama pull {model}",
        )
    except Exception:
        return "skip", f"Model '{model}'", "Skipped (server not reachable)"


def _check_model_responds(base_url: str, model: str) -> tuple[str, str, str]:
    try:
        import ollama

        client = ollama.Client(host=base_url)
        t0 = time.time()
        client.chat(model=model, messages=[{"role": "user", "content": "Say hello."}])
        elapsed = time.time() - t0
        return "pass", "Model responds", f"OK ({elapsed:.1f}s)"
    except Exception as exc:
        return "fail", "Model responds", f"Error: {exc}"


def cmd_doctor(args: argparse.Namespace) -> None:
    """Run environment health checks."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Determine provider settings
    base_url = "http://localhost:11434"
    model = "qwen2.5-coder:14b"
    provider = "ollama"

    config_path = getattr(args, "config", "apab.yaml")
    try:
        from pathlib import Path

        from apab.core.config import load_config

        config = load_config(Path(config_path))
        provider = config.llm.provider
        model = config.llm.model
        if config.llm.base_url:
            base_url = config.llm.base_url
        console.print(f"[dim]Using config: {config_path}[/dim]")
    except Exception:
        console.print(f"[dim]No config found at {config_path} — using defaults[/dim]")

    console.print()

    # Run checks
    checks: list[tuple[str, str, str]] = []
    checks.append(_check_python())
    checks.append(_check_core_deps())
    checks.append(_check_edgefem())

    if provider == "ollama":
        checks.append(_check_ollama_package())
        checks.append(_check_ollama_server(base_url))

        # Only check model if server is reachable
        server_ok = checks[-1][0] == "pass"
        if server_ok:
            checks.append(_check_model(base_url, model))
            model_ok = checks[-1][0] == "pass"
            if model_ok:
                console.print("[dim]Pinging model (this may take a moment)...[/dim]")
                checks.append(_check_model_responds(base_url, model))
        else:
            checks.append(("skip", f"Model '{model}'", "Skipped (server not reachable)"))
            checks.append(("skip", "Model responds", "Skipped (server not reachable)"))
    else:
        console.print(f"[dim]Provider '{provider}' — skipping Ollama-specific checks[/dim]")

    # Display results
    status_icons = {
        "pass": "[green]\u2713[/green]",
        "fail": "[red]\u2717[/red]",
        "warn": "[yellow]![/yellow]",
        "skip": "[dim]-[/dim]",
    }

    table = Table(title="APAB Environment Check", show_header=True, header_style="bold")
    table.add_column("Check", style="bold", min_width=22)
    table.add_column("Status", justify="center", min_width=6)
    table.add_column("Details", min_width=30)

    for status, name, detail in checks:
        table.add_row(name, status_icons.get(status, "?"), detail)

    console.print(table)

    # Summary
    failures = sum(1 for s, _, _ in checks if s == "fail")
    warnings = sum(1 for s, _, _ in checks if s == "warn")
    if failures:
        console.print(
            f"\n[red bold]{failures} check(s) failed.[/red bold] "
            "Fix the issues above and re-run."
        )
        sys.exit(1)
    elif warnings:
        console.print(
            f"\n[yellow]{warnings} warning(s)[/yellow], but all "
            "required checks passed. Ready to design."
        )
    else:
        console.print("\n[green bold]All checks passed.[/green bold] Ready to design.")
