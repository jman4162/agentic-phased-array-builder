"""MCP tools for project I/O: init, validate, save, import."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from apab.mcp.server import get_mcp

logger = logging.getLogger(__name__)
mcp = get_mcp()


@mcp.tool()
async def project_init(
    name: Annotated[str, Field(description="Project name")],
    workspace: Annotated[str, Field(description="Workspace directory path")] = "./workspace",
) -> dict[str, Any]:
    """Initialize a new APAB project scaffold."""
    try:
        from apab.core.config import save_config
        from apab.core.schemas import ProjectConfig, ProjectMeta
        from apab.core.workspace import Workspace

        logger.info("Initializing project %r in %s", name, workspace)

        config = ProjectConfig(
            project=ProjectMeta(name=name, workspace=workspace),
        )

        ws = Workspace(Path(workspace))
        ws.ensure_dirs()

        config_path = Path("apab.yaml")
        save_config(config, config_path)

        return {
            "config_path": str(config_path),
            "workspace": workspace,
            "status": "initialized",
        }
    except Exception as e:
        logger.exception("project_init failed")
        return {"error": str(e), "status": "failed"}


@mcp.tool()
async def project_validate(
    config_path: Annotated[str, Field(description="Path to apab.yaml")],
) -> dict[str, Any]:
    """Validate an APAB project configuration file."""
    try:
        from apab.core.config import load_config

        logger.info("Validating config %s", config_path)
        config = load_config(Path(config_path))
        return {
            "valid": True,
            "project_name": config.project.name,
            "workspace": config.project.workspace,
        }
    except Exception as e:
        logger.warning("Config validation failed: %s", e)
        return {"valid": False, "error": str(e)}


@mcp.tool()
async def io_import_touchstone(
    filepath: Annotated[str, Field(description="Path to Touchstone (.sNp) file")],
) -> dict[str, Any]:
    """Import a Touchstone file and return S-parameter metadata."""
    try:
        from apab.emtool.importers import import_touchstone

        logger.info("Importing Touchstone file: %s", filepath)
        data = import_touchstone(filepath)

        return {
            "n_ports": data["n_ports"],
            "n_freqs": len(data["freqs"]),
            "freq_min_hz": float(data["freqs"][0]),
            "freq_max_hz": float(data["freqs"][-1]),
            "z0": data["z0"],
            "comments": data.get("comments", []),
            "status": "imported",
        }
    except Exception as e:
        logger.exception("io_import_touchstone failed")
        return {"error": str(e), "status": "failed"}


@mcp.tool()
async def io_save_hdf5(
    run_id: Annotated[str, Field(description="Run ID for artifact storage")],
    data_json: Annotated[str, Field(description="JSON-serialized data to save")],
    filename: Annotated[str, Field(description="Output filename within run dir")] = "data.json",
    workspace: Annotated[str, Field(description="Workspace root")] = "./workspace",
) -> dict[str, str]:
    """Save data to a run's artifact directory (as JSON for now, HDF5 in v0.3)."""
    try:
        from apab.core.workspace import validate_path_within

        run_dir = Path(workspace) / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        out_path = run_dir / filename
        validate_path_within(out_path, Path(workspace))

        logger.info("Saving artifact to %s", out_path)
        data = json.loads(data_json)
        out_path.write_text(json.dumps(data, indent=2))

        return {"filepath": str(out_path), "status": "saved"}
    except Exception as e:
        logger.exception("io_save_hdf5 failed")
        return {"error": str(e), "status": "failed"}
