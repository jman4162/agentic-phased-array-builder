"""MCP tools for external EM tool adapters."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from apab.mcp.server import get_mcp

logger = logging.getLogger(__name__)
mcp = get_mcp()


@mcp.tool()
async def emtool_list_adapters() -> dict[str, Any]:
    """List all discovered external EM tool adapters."""
    try:
        from apab.emtool.registry import discover_em_adapters

        logger.info("Listing EM tool adapters")
        adapters = discover_em_adapters()
        return {
            "adapters": list(adapters.keys()),
            "count": len(adapters),
        }
    except Exception as e:
        logger.exception("emtool_list_adapters failed")
        return {"error": str(e), "status": "failed"}


@mcp.tool()
async def emtool_import_results(
    filepath: Annotated[str, Field(description="Path to result file (.sNp or .csv)")],
    file_type: Annotated[
        str, Field(description="File type: 'touchstone' or 'farfield_csv'")
    ] = "touchstone",
    run_id: Annotated[
        str, Field(description="Run ID; with workspace, persists arrays as HDF5")
    ] = "",
    workspace: Annotated[str, Field(description="Workspace root")] = "",
) -> dict[str, Any]:
    """Import results from an external EM tool; persist arrays when a run is given."""
    try:
        from apab.emtool.importers import import_farfield_csv, import_touchstone

        logger.info("Importing EM results from %s (type=%s)", filepath, file_type)

        if file_type == "touchstone":
            data = import_touchstone(filepath)
            result = {
                "n_ports": data["n_ports"],
                "n_freqs": len(data["freqs"]),
                "freq_min_hz": float(data["freqs"][0]),
                "freq_max_hz": float(data["freqs"][-1]),
                "z0": data["z0"],
                "status": "imported",
            }
            if run_id and workspace:
                from apab.mcp.tools_io import _persist_touchstone_h5

                result["artifact_path"] = _persist_touchstone_h5(
                    data, Path(filepath), run_id, Path(workspace)
                )
            return result
        elif file_type == "farfield_csv":
            data = import_farfield_csv(filepath)
            result = {
                "n_points": len(data["theta_deg"]),
                "theta_range": [float(min(data["theta_deg"])), float(max(data["theta_deg"]))],
                "phi_range": [float(min(data["phi_deg"])), float(max(data["phi_deg"]))],
                "status": "imported",
            }
            if run_id and workspace:
                result["artifact_path"] = _persist_farfield_h5(
                    data, Path(filepath), run_id, Path(workspace)
                )
            return result
        else:
            return {"error": f"Unknown file_type: {file_type}", "status": "failed"}
    except Exception as e:
        logger.exception("emtool_import_results failed")
        return {"error": str(e), "status": "failed"}


def _persist_farfield_h5(
    data: dict[str, Any], source: Path, run_id: str, workspace: Path
) -> str:
    """Write parsed far-field arrays into the run's emtool artifacts."""
    import h5py
    import numpy as np

    from apab.core.workspace import validate_path_within

    emtool_dir = workspace / "runs" / run_id / "artifacts" / "emtool"
    emtool_dir.mkdir(parents=True, exist_ok=True)
    out_path = emtool_dir / (source.stem + ".h5")
    validate_path_within(out_path, workspace)

    with h5py.File(out_path, "w") as fh:
        for key in ("theta_deg", "phi_deg", "gain_db"):
            fh.create_dataset(key, data=np.asarray(data[key], dtype=float))
        fh.attrs["source"] = str(source)
        for k, v in (data.get("metadata") or {}).items():
            fh.attrs[f"meta_{k}"] = str(v)
    logger.info("Persisted far-field arrays to %s", out_path)
    return str(out_path)
