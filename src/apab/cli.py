"""APAB command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys


def cmd_init(args: argparse.Namespace) -> None:
    """Scaffold a new APAB project."""
    from apab.commands.init import cmd_init as _init

    _init(args)


def cmd_design(args: argparse.Namespace) -> None:
    """Start an interactive agent design session."""
    from apab.commands.design import cmd_design as _design

    _design(args)


def cmd_run(args: argparse.Namespace) -> None:
    """Run a non-interactive workflow from config."""
    from apab.commands.run import cmd_run as _run

    _run(args)


def cmd_report(args: argparse.Namespace) -> None:
    """Generate a report from a run bundle."""
    from apab.commands.report import cmd_report as _report

    _report(args)


def cmd_mcp_serve(args: argparse.Namespace) -> None:
    """Start the APAB MCP server."""
    from apab.commands.mcp_serve import cmd_mcp_serve as _serve

    _serve(args)


def build_parser() -> argparse.ArgumentParser:
    """Build the APAB argument parser."""
    parser = argparse.ArgumentParser(
        prog="apab",
        description="Agentic Phased Array Builder — LLM-driven phased-array antenna design",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {_get_version()}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init
    p_init = subparsers.add_parser("init", help="Scaffold a new APAB project")
    p_init.add_argument("--name", default="my_apab_project", help="Project name")
    p_init.add_argument("--dir", default=".", help="Directory to create project in")
    p_init.set_defaults(func=cmd_init)

    # design
    p_design = subparsers.add_parser("design", help="Interactive agent design session")
    p_design.add_argument("--config", default="apab.yaml", help="Config file path")
    p_design.set_defaults(func=cmd_design)

    # run
    p_run = subparsers.add_parser("run", help="Non-interactive run from config")
    p_run.add_argument("--config", default="apab.yaml", help="Config file path")
    p_run.set_defaults(func=cmd_run)

    # report
    p_report = subparsers.add_parser("report", help="Generate report from run bundle")
    p_report.add_argument("run_id", help="Run ID to generate report for")
    p_report.add_argument("--config", default="apab.yaml", help="Config file path")
    p_report.add_argument("--format", choices=["markdown", "html"], default="markdown")
    p_report.set_defaults(func=cmd_report)

    # mcp serve
    p_mcp = subparsers.add_parser("mcp", help="MCP server commands")
    mcp_sub = p_mcp.add_subparsers(dest="mcp_command")
    p_serve = mcp_sub.add_parser("serve", help="Start the MCP server")
    p_serve.add_argument("--config", default="apab.yaml", help="Config file path")
    p_serve.add_argument(
        "--transport", choices=["stdio", "http"], default="stdio", help="Transport mode"
    )
    p_serve.set_defaults(func=cmd_mcp_serve)

    return parser


def _get_version() -> str:
    from apab import __version__

    return __version__


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        logging.getLogger(__name__).exception("Command failed")
        print(f"\nerror: {e}", file=sys.stderr)
        sys.exit(1)
