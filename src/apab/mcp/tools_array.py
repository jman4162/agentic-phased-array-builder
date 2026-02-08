"""MCP tools for array-pattern computation (phased-array-modeling)."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from pydantic import Field

from apab.core.schemas import ArraySpec, ScanPoint
from apab.mcp.server import get_mcp

logger = logging.getLogger(__name__)
mcp = get_mcp()


def _build_array_spec(
    nx: int,
    ny: int,
    dx_m: float,
    dy_m: float,
    taper: str,
    theta0: float,
    phi0: float,
) -> ArraySpec:
    """Build an ArraySpec from individual parameters."""

    return ArraySpec(
        size=[nx, ny],
        spacing_m=[dx_m, dy_m],
        taper=taper,
        steer=ScanPoint(theta_deg=theta0, phi_deg=phi0),
    )


@mcp.tool()
async def pattern_compute(
    nx: Annotated[int, Field(description="Number of elements in x")],
    ny: Annotated[int, Field(description="Number of elements in y")],
    dx_m: Annotated[float, Field(description="Element spacing in x (metres)")],
    dy_m: Annotated[float, Field(description="Element spacing in y (metres)")],
    freq_hz: Annotated[float, Field(description="Operating frequency (Hz)")],
    theta0: Annotated[float, Field(description="Steering theta (degrees)")] = 0.0,
    phi0: Annotated[float, Field(description="Steering phi (degrees)")] = 0.0,
    taper: Annotated[str, Field(description="Taper window name")] = "uniform",
) -> dict[str, Any]:
    """Compute a full 2-D array radiation pattern."""
    try:
        from apab.pattern.wrappers_pam import PAMPatternEngine

        logger.info(
            "Computing pattern: %dx%d array @ %.2e Hz, taper=%s", nx, ny, freq_hz, taper,
        )

        spec = _build_array_spec(nx, ny, dx_m, dy_m, taper, theta0, phi0)
        engine = PAMPatternEngine()
        result = engine.full_pattern(spec, freq_hz, theta0, phi0)

        logger.info("Pattern computed: directivity=%.2f dBi", result.directivity_dbi)
        return {
            "directivity_dbi": result.directivity_dbi,
            "sidelobe_level_db": result.sidelobe_level_db,
            "metadata": result.metadata,
            "theta_deg_len": len(result.theta_deg),
            "phi_deg_len": len(result.phi_deg),
        }
    except Exception as e:
        logger.exception("pattern_compute failed")
        return {"error": str(e), "status": "failed"}


@mcp.tool()
async def pattern_plot_cuts(
    nx: Annotated[int, Field(description="Number of elements in x")],
    ny: Annotated[int, Field(description="Number of elements in y")],
    dx_m: Annotated[float, Field(description="Element spacing in x (metres)")],
    dy_m: Annotated[float, Field(description="Element spacing in y (metres)")],
    freq_hz: Annotated[float, Field(description="Operating frequency (Hz)")],
    theta0: Annotated[float, Field(description="Steering theta (degrees)")] = 0.0,
    phi0: Annotated[float, Field(description="Steering phi (degrees)")] = 0.0,
    taper: Annotated[str, Field(description="Taper window name")] = "uniform",
    output_path: Annotated[str, Field(description="Output PNG file path")] = "pattern_cuts.png",
) -> dict[str, Any]:
    """Compute and plot E-plane / H-plane pattern cuts."""
    try:
        from apab.core.workspace import reject_path_traversal

        reject_path_traversal(output_path)

        import numpy as np
        import phased_array as pa

        from apab.pattern.wrappers_pam import PAMPatternEngine

        logger.info("Plotting pattern cuts to %s", output_path)

        spec = _build_array_spec(nx, ny, dx_m, dy_m, taper, theta0, phi0)
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

        fig, ax = plt.subplots(figsize=(8, 5))
        theta_deg = np.rad2deg(theta_cut)
        ax.plot(theta_deg, e_plane_db, label="E-plane")
        ax.plot(theta_deg, h_plane_db, label="H-plane")
        ax.set_xlabel("Theta (deg)")
        ax.set_ylabel("Normalized pattern (dB)")
        ax.set_title(f"Pattern cuts — {nx}×{ny} array @ {freq_hz/1e9:.2f} GHz")
        ax.legend()
        ax.grid(True)
        ax.set_ylim(bottom=-40)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        return {"output_path": output_path, "status": "saved"}
    except Exception as e:
        logger.exception("pattern_plot_cuts failed")
        return {"error": str(e), "status": "failed"}


@mcp.tool()
async def pattern_plot_3d(
    nx: Annotated[int, Field(description="Number of elements in x")],
    ny: Annotated[int, Field(description="Number of elements in y")],
    dx_m: Annotated[float, Field(description="Element spacing in x (metres)")],
    dy_m: Annotated[float, Field(description="Element spacing in y (metres)")],
    freq_hz: Annotated[float, Field(description="Operating frequency (Hz)")],
    theta0: Annotated[float, Field(description="Steering theta (degrees)")] = 0.0,
    phi0: Annotated[float, Field(description="Steering phi (degrees)")] = 0.0,
    taper: Annotated[str, Field(description="Taper window name")] = "uniform",
    output_path: Annotated[str, Field(description="Output PNG file path")] = "pattern_3d.png",
) -> dict[str, Any]:
    """Compute and plot a 3-D pattern surface."""
    try:
        from apab.core.workspace import reject_path_traversal

        reject_path_traversal(output_path)

        import numpy as np

        from apab.pattern.wrappers_pam import PAMPatternEngine

        logger.info("Plotting 3D pattern to %s", output_path)

        spec = _build_array_spec(nx, ny, dx_m, dy_m, taper, theta0, phi0)
        engine = PAMPatternEngine()
        result = engine.full_pattern(spec, freq_hz, theta0, phi0)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        theta = np.array(result.theta_deg)
        phi = np.array(result.phi_deg)
        pattern_db = np.array(result.pattern_db)

        fig, ax = plt.subplots(figsize=(8, 6))
        c = ax.pcolormesh(phi, theta, pattern_db, shading="auto", cmap="jet")
        fig.colorbar(c, ax=ax, label="dB")
        ax.set_xlabel("Phi (deg)")
        ax.set_ylabel("Theta (deg)")
        ax.set_title(f"Full pattern — {nx}×{ny} array @ {freq_hz/1e9:.2f} GHz")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        return {
            "output_path": output_path,
            "directivity_dbi": result.directivity_dbi,
            "status": "saved",
        }
    except Exception as e:
        logger.exception("pattern_plot_3d failed")
        return {"error": str(e), "status": "failed"}


@mcp.tool()
async def pattern_multi_beam(
    nx: Annotated[int, Field(description="Number of elements in x")],
    ny: Annotated[int, Field(description="Number of elements in y")],
    dx_m: Annotated[float, Field(description="Element spacing in x (metres)")],
    dy_m: Annotated[float, Field(description="Element spacing in y (metres)")],
    freq_hz: Annotated[float, Field(description="Operating frequency (Hz)")],
    beam_directions: Annotated[
        list[list[float]],
        Field(description="List of [theta, phi] pairs in degrees"),
    ],
    taper: Annotated[str, Field(description="Taper window name")] = "uniform",
) -> dict[str, Any]:
    """Compute a multi-beam pattern via weight superposition."""
    try:
        from apab.pattern.wrappers_pam import PAMPatternEngine

        logger.info("Computing multi-beam pattern: %d beams", len(beam_directions))

        spec = ArraySpec(
            size=[nx, ny],
            spacing_m=[dx_m, dy_m],
            taper=taper,
            steer=ScanPoint(theta_deg=0, phi_deg=0),
        )
        engine = PAMPatternEngine()
        dirs = [(d[0], d[1]) for d in beam_directions]
        result = engine.multi_beam(spec, freq_hz, dirs)

        return {
            "directivity_dbi": result.directivity_dbi,
            "n_beams": len(beam_directions),
            "metadata": result.metadata,
        }
    except Exception as e:
        logger.exception("pattern_multi_beam failed")
        return {"error": str(e), "status": "failed"}


@mcp.tool()
async def pattern_null_steer(
    nx: Annotated[int, Field(description="Number of elements in x")],
    ny: Annotated[int, Field(description="Number of elements in y")],
    dx_m: Annotated[float, Field(description="Element spacing in x (metres)")],
    dy_m: Annotated[float, Field(description="Element spacing in y (metres)")],
    freq_hz: Annotated[float, Field(description="Operating frequency (Hz)")],
    theta0: Annotated[float, Field(description="Main beam theta (degrees)")],
    phi0: Annotated[float, Field(description="Main beam phi (degrees)")],
    null_directions: Annotated[
        list[list[float]],
        Field(description="List of [theta, phi] null directions in degrees"),
    ],
    taper: Annotated[str, Field(description="Taper window name")] = "uniform",
) -> dict[str, Any]:
    """Compute a pattern with steered nulls."""
    try:
        from apab.pattern.wrappers_pam import PAMPatternEngine

        logger.info("Computing null-steered pattern: %d nulls", len(null_directions))

        spec = ArraySpec(
            size=[nx, ny],
            spacing_m=[dx_m, dy_m],
            taper=taper,
            steer=ScanPoint(theta_deg=theta0, phi_deg=phi0),
        )
        engine = PAMPatternEngine()
        null_dirs = [(d[0], d[1]) for d in null_directions]
        result = engine.null_steering(spec, freq_hz, theta0, phi0, null_dirs)

        return {
            "directivity_dbi": result.directivity_dbi,
            "n_nulls": len(null_directions),
            "metadata": result.metadata,
        }
    except Exception as e:
        logger.exception("pattern_null_steer failed")
        return {"error": str(e), "status": "failed"}
