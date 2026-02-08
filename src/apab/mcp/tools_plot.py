"""MCP tools for quick-look plotting."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from pydantic import Field

from apab.mcp.server import get_mcp

logger = logging.getLogger(__name__)
mcp = get_mcp()


@mcp.tool()
async def plot_quicklook(
    nx: Annotated[int, Field(description="Number of elements in x")],
    ny: Annotated[int, Field(description="Number of elements in y")],
    dx_m: Annotated[float, Field(description="Element spacing in x (metres)")],
    dy_m: Annotated[float, Field(description="Element spacing in y (metres)")],
    freq_hz: Annotated[float, Field(description="Operating frequency (Hz)")],
    theta0: Annotated[float, Field(description="Steering theta (degrees)")] = 0.0,
    phi0: Annotated[float, Field(description="Steering phi (degrees)")] = 0.0,
    taper: Annotated[str, Field(description="Taper window name")] = "uniform",
    output_path: Annotated[str, Field(description="Output PNG file path")] = "quicklook.png",
) -> dict[str, Any]:
    """Generate a quick-look summary plot: pattern cuts + array geometry."""
    try:
        from apab.core.workspace import reject_path_traversal

        reject_path_traversal(output_path)

        import numpy as np
        import phased_array as pa

        from apab.core.schemas import ArraySpec, ScanPoint
        from apab.pattern.wrappers_pam import PAMPatternEngine

        logger.info("Generating quicklook plot to %s", output_path)

        spec = ArraySpec(
            size=[nx, ny],
            spacing_m=[dx_m, dy_m],
            taper=taper,
            steer=ScanPoint(theta_deg=theta0, phi_deg=phi0),
        )
        engine = PAMPatternEngine()
        geom = engine.create_geometry(spec, freq_hz)
        steering = engine.compute_steering_weights(geom, freq_hz, theta0, phi0)
        taper_w = engine.apply_taper(spec)
        weights = steering * taper_w

        k = pa.frequency_to_k(freq_hz)
        theta_cut, e_plane_db, h_plane_db = pa.compute_pattern_cuts(
            geom.x, geom.y, weights, k, theta0_deg=theta0, phi0_deg=phi0,
        )

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: pattern cuts
        ax = axes[0]
        theta_deg = np.rad2deg(theta_cut)
        ax.plot(theta_deg, e_plane_db, label="E-plane")
        ax.plot(theta_deg, h_plane_db, label="H-plane")
        ax.set_xlabel("Theta (deg)")
        ax.set_ylabel("Normalized pattern (dB)")
        ax.set_title("Pattern cuts")
        ax.legend()
        ax.grid(True)
        ax.set_ylim(bottom=-40)

        # Right: element positions
        ax = axes[1]
        ax.scatter(geom.x, geom.y, c=np.abs(weights), cmap="viridis", s=15)
        ax.set_xlabel("x (wavelengths)")
        ax.set_ylabel("y (wavelengths)")
        ax.set_title(f"Array layout — {nx}×{ny}")
        ax.set_aspect("equal")
        ax.grid(True)

        title = (
            f"{nx}×{ny} array @ {freq_hz/1e9:.2f} GHz,"
            f" steer=({theta0}\u00b0,{phi0}\u00b0)"
        )
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        return {"output_path": output_path, "status": "saved"}
    except Exception as e:
        logger.exception("plot_quicklook failed")
        return {"error": str(e), "status": "failed"}
