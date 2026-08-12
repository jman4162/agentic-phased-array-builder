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
    run_id: Annotated[
        str, Field(description="Run ID; with workspace, persists arrays as HDF5")
    ] = "",
    workspace: Annotated[str, Field(description="Workspace root")] = "",
) -> dict[str, Any]:
    """Import a Touchstone file; persist the arrays when a run is given.

    Without run_id/workspace only metadata is returned (the parsed
    S-matrices are discarded). With them, the full complex data lands in
    the run's artifacts/emtool/ directory so downstream tools can use it.
    """
    try:
        from apab.emtool.importers import import_touchstone

        logger.info("Importing Touchstone file: %s", filepath)
        data = import_touchstone(filepath)

        result = {
            "n_ports": data["n_ports"],
            "n_freqs": len(data["freqs"]),
            "freq_min_hz": float(data["freqs"][0]),
            "freq_max_hz": float(data["freqs"][-1]),
            "z0": data["z0"],
            "comments": data.get("comments", []),
            "status": "imported",
        }
        if run_id and workspace:
            result["artifact_path"] = _persist_touchstone_h5(
                data, Path(filepath), run_id, Path(workspace)
            )
        return result
    except Exception as e:
        logger.exception("io_import_touchstone failed")
        return {"error": str(e), "status": "failed"}


def _persist_touchstone_h5(
    data: dict[str, Any], source: Path, run_id: str, workspace: Path
) -> str:
    """Write parsed Touchstone arrays into the run's emtool artifacts."""
    import h5py
    import numpy as np

    from apab.core.workspace import validate_path_within

    emtool_dir = workspace / "runs" / run_id / "artifacts" / "emtool"
    emtool_dir.mkdir(parents=True, exist_ok=True)
    out_path = emtool_dir / (source.stem + ".h5")
    validate_path_within(out_path, workspace)

    with h5py.File(out_path, "w") as fh:
        fh.create_dataset("freqs_hz", data=np.asarray(data["freqs"], dtype=float))
        fh.create_dataset(
            "s_params", data=np.asarray(data["s_params"], dtype=complex)
        )
        fh.attrs["n_ports"] = int(data["n_ports"])
        fh.attrs["z0"] = float(data["z0"])
        fh.attrs["source"] = str(source)
    logger.info("Persisted Touchstone arrays to %s", out_path)
    return str(out_path)


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
