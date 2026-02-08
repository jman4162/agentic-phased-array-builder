"""MCP tools for external EM tool adapters."""

from __future__ import annotations

import logging
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
) -> dict[str, Any]:
    """Import results from an external EM tool."""
    try:
        from apab.emtool.importers import import_farfield_csv, import_touchstone

        logger.info("Importing EM results from %s (type=%s)", filepath, file_type)

        if file_type == "touchstone":
            data = import_touchstone(filepath)
            return {
                "n_ports": data["n_ports"],
                "n_freqs": len(data["freqs"]),
                "freq_min_hz": float(data["freqs"][0]),
                "freq_max_hz": float(data["freqs"][-1]),
                "z0": data["z0"],
                "status": "imported",
            }
        elif file_type == "farfield_csv":
            data = import_farfield_csv(filepath)
            return {
                "n_points": len(data["theta_deg"]),
                "theta_range": [float(min(data["theta_deg"])), float(max(data["theta_deg"]))],
                "phi_range": [float(min(data["phi_deg"])), float(max(data["phi_deg"]))],
                "status": "imported",
            }
        else:
            return {"error": f"Unknown file_type: {file_type}", "status": "failed"}
    except Exception as e:
        logger.exception("emtool_import_results failed")
        return {"error": str(e), "status": "failed"}
