"""MCP tool comparing simulated against measured S-parameter data.

The measured side must satisfy the measurement artifact contract
(docs/measurement-contract.md): a ``<name>.meta.yaml`` provenance sidecar
is required, and its ``synthetic`` flag propagates into the report.
"""

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
async def compare_sim_measured(
    sim_path: Annotated[
        str, Field(description="Simulated Touchstone file (e.g. an EdgeFEM export)")
    ],
    measured_path: Annotated[
        str, Field(description="Measured Touchstone file; requires a .meta.yaml sidecar")
    ],
    run_id: Annotated[str, Field(description="Run ID for the report artifact")],
    workspace: Annotated[str, Field(description="Workspace root")] = "./workspace",
    port: Annotated[int, Field(description="Reflection port to compare (0-based)")] = 0,
) -> dict[str, Any]:
    """Compare |S_ii| in dB between a simulation and a measurement.

    Interpolates the simulated trace onto the measured frequency points
    over the overlapping band and reports RMSE and maximum absolute
    deviation. Writes a JSON report into the run's artifacts/report/
    directory and returns its path with the summary numbers.
    """
    try:
        import numpy as np

        from apab.core.workspace import validate_path_within
        from apab.emtool.importers import import_touchstone, load_provenance

        provenance = load_provenance(measured_path)

        sim = import_touchstone(sim_path)
        meas = import_touchstone(measured_path)
        if port >= sim["n_ports"] or port >= meas["n_ports"]:
            return {
                "error": f"port {port} out of range (sim {sim['n_ports']}, "
                f"measured {meas['n_ports']} ports)",
                "status": "failed",
            }

        sim_f = np.asarray(sim["freqs"], dtype=float)
        meas_f = np.asarray(meas["freqs"], dtype=float)
        sim_db = _refl_db(sim["s_params"], port)
        meas_db = _refl_db(meas["s_params"], port)

        lo = max(sim_f.min(), meas_f.min())
        hi = min(sim_f.max(), meas_f.max())
        if lo >= hi:
            return {
                "error": "no overlapping frequency band between the two files",
                "status": "failed",
            }
        mask = (meas_f >= lo) & (meas_f <= hi)
        grid = meas_f[mask]
        sim_on_grid = np.interp(grid, sim_f, sim_db)
        deviation = sim_on_grid - meas_db[mask]

        report = {
            "quantity": f"|S{port + 1}{port + 1}| (dB)",
            "sim_path": str(sim_path),
            "measured_path": str(measured_path),
            "band_hz": [float(lo), float(hi)],
            "n_points": int(mask.sum()),
            "rmse_db": float(np.sqrt(np.mean(deviation**2))),
            "max_abs_deviation_db": float(np.max(np.abs(deviation))),
            "worst_freq_hz": float(grid[int(np.argmax(np.abs(deviation)))]),
            "measured_provenance": provenance.model_dump(),
            "synthetic_measurement": provenance.synthetic,
        }

        report_dir = Path(workspace) / "runs" / run_id / "artifacts" / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        out_path = report_dir / "compare_sim_measured.json"
        validate_path_within(out_path, Path(workspace))
        out_path.write_text(json.dumps(report, indent=2))

        return {
            "report_path": str(out_path),
            "rmse_db": report["rmse_db"],
            "max_abs_deviation_db": report["max_abs_deviation_db"],
            "worst_freq_hz": report["worst_freq_hz"],
            "n_points": report["n_points"],
            "synthetic_measurement": provenance.synthetic,
            "status": "completed",
        }
    except Exception as e:
        logger.exception("compare_sim_measured failed")
        return {"error": str(e), "status": "failed"}


def _refl_db(s_params: list[Any], port: int) -> Any:
    """|S_pp| in dB per frequency point, floored at -200 dB."""
    import numpy as np

    mags = np.array([abs(s[port][port]) for s in s_params], dtype=float)
    return 20.0 * np.log10(np.maximum(mags, 1e-10))
