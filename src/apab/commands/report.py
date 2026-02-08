"""apab report command — generate reports from run bundles."""

from __future__ import annotations

import argparse
from pathlib import Path


def cmd_report(args: argparse.Namespace) -> None:
    """Generate a report from a run bundle."""
    from apab.core.config import load_config
    from apab.report.build_report import ReportBuilder

    config = load_config(Path(args.config))
    workspace = Path(config.project.workspace)
    run_dir = workspace / "runs" / args.run_id

    if not run_dir.exists():
        print(f"[apab report] Run directory not found: {run_dir}")
        return

    builder = ReportBuilder(run_dir)

    if args.format == "markdown":
        report = builder.build_markdown()
        out_path = run_dir / "report.md"
        out_path.write_text(report)
        print(f"[apab report] Markdown report written to: {out_path}")
    else:
        print("[apab report] HTML reports are deferred to v0.3.")
