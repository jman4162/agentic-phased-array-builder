"""MCP resource definitions for APAB."""

from __future__ import annotations

import json
from pathlib import Path

from apab.mcp.server import get_mcp

mcp = get_mcp()


@mcp.resource("apab://config")
async def resource_config() -> str:
    """Return the current project configuration as YAML."""
    server = mcp
    config = getattr(server, "_apab_config", None)
    if config is None:
        return json.dumps({"error": "No configuration loaded"})
    result: str = config.model_dump_json(indent=2)
    return result


@mcp.resource("apab://runs")
async def resource_runs() -> str:
    """List all run IDs in the workspace."""
    server = mcp
    config = getattr(server, "_apab_config", None)
    if config is None:
        return json.dumps({"runs": [], "error": "No configuration loaded"})

    workspace = Path(config.project.workspace)
    runs_dir = workspace / "runs"
    if not runs_dir.exists():
        return json.dumps({"runs": []})

    runs = sorted(
        [d.name for d in runs_dir.iterdir() if d.is_dir()],
        reverse=True,
    )
    return json.dumps({"runs": runs})


@mcp.resource("apab://runs/{run_id}/manifest")
async def resource_run_manifest(run_id: str) -> str:
    """Return the manifest for a specific run."""
    server = mcp
    config = getattr(server, "_apab_config", None)
    if config is None:
        return json.dumps({"error": "No configuration loaded"})

    workspace = Path(config.project.workspace)
    manifest_path = workspace / "runs" / run_id / "manifest.json"
    if not manifest_path.exists():
        return json.dumps({"error": f"Manifest not found for run {run_id}"})

    return manifest_path.read_text()


@mcp.resource("apab://runs/{run_id}/artifacts/{path}")
async def resource_run_artifact(run_id: str, path: str) -> str:
    """Return the contents of a specific artifact."""
    server = mcp
    config = getattr(server, "_apab_config", None)
    if config is None:
        return json.dumps({"error": "No configuration loaded"})

    workspace = Path(config.project.workspace)
    artifact_path = workspace / "runs" / run_id / path

    # Security: ensure the path stays within the workspace.
    try:
        artifact_path.resolve().relative_to(workspace.resolve())
    except ValueError:
        return json.dumps({"error": "Path escapes workspace"})

    if not artifact_path.exists():
        return json.dumps({"error": f"Artifact not found: {path}"})

    return artifact_path.read_text()
