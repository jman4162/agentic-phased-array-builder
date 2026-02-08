"""MCP tools for EdgeFEM unit-cell simulation."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from pydantic import Field

from apab.mcp.server import get_mcp

logger = logging.getLogger(__name__)
mcp = get_mcp()


@mcp.tool()
async def edgefem_run_unit_cell(
    period_x: Annotated[float, Field(description="Unit-cell period in x (metres)")],
    period_y: Annotated[float, Field(description="Unit-cell period in y (metres)")],
    substrate_height: Annotated[float, Field(description="Substrate height (metres)")],
    substrate_eps_r: Annotated[float, Field(description="Substrate relative permittivity")],
    freq_start: Annotated[float, Field(description="Start frequency (Hz)")],
    freq_stop: Annotated[float, Field(description="Stop frequency (Hz)")],
    n_freq: Annotated[int, Field(description="Number of frequency points")],
    theta_deg: Annotated[float, Field(description="Scan angle theta (degrees)")] = 0.0,
    phi_deg: Annotated[float, Field(description="Scan angle phi (degrees)")] = 0.0,
    polarization: Annotated[str, Field(description="Polarization: 'H' or 'V'")] = "H",
    patch_w_m: Annotated[float | None, Field(description="Patch width (metres)")] = None,
    patch_l_m: Annotated[float | None, Field(description="Patch length (metres)")] = None,
) -> dict[str, Any]:
    """Run an EdgeFEM unit-cell frequency sweep and return S-parameter results."""
    try:
        from apab.core.schemas import (
            FreqRange,
            GeometryKind,
            GeometrySpec,
            LatticeSpec,
            ScanSpec,
            SolverSpec,
            SweepSpec,
            UnitCellSpec,
        )
        from apab.coupling.sparams import EdgeFEMAdapter

        logger.info(
            "Running unit-cell sweep: freq=%.2e–%.2e Hz, n=%d, theta=%.1f, phi=%.1f",
            freq_start, freq_stop, n_freq, theta_deg, phi_deg,
        )

        lattice = LatticeSpec(dx_m=period_x, dy_m=period_y)
        geom_params: dict[str, Any] = {}
        if patch_w_m is not None:
            geom_params["patch_w_m"] = patch_w_m
        if patch_l_m is not None:
            geom_params["patch_l_m"] = patch_l_m
        geom_params["substrate_h_m"] = substrate_height
        geom_params["er"] = substrate_eps_r

        geometry = GeometrySpec(kind=GeometryKind("patch"), params=geom_params)
        unit_cell = UnitCellSpec(lattice=lattice, geometry=geometry)

        sweep = SweepSpec(
            freq_hz=FreqRange(start=freq_start, stop=freq_stop, n=n_freq),
            scan=ScanSpec(
                theta_deg=[theta_deg],
                phi_deg=[phi_deg],
                n_theta=1,
                n_phi=1,
            ),
            polarization=[polarization],
        )

        from apab.core.schemas import SimRequest

        request = SimRequest(unit_cell=unit_cell, sweep=sweep, solver=SolverSpec())

        adapter = EdgeFEMAdapter()
        result = adapter.run_full_sweep(request)

        logger.info("Unit-cell sweep completed: %d freq points", len(result.freq_hz))
        return {
            "freq_hz": result.freq_hz,
            "scan_points": [sp.model_dump() for sp in result.scan_points],
            "polarizations": result.polarizations,
            "metadata": result.metadata,
            "status": "completed",
        }
    except Exception as e:
        logger.exception("edgefem_run_unit_cell failed")
        return {"error": str(e), "status": "failed"}


@mcp.tool()
async def edgefem_surface_impedance(
    period_x: Annotated[float, Field(description="Unit-cell period in x (metres)")],
    period_y: Annotated[float, Field(description="Unit-cell period in y (metres)")],
    substrate_height: Annotated[float, Field(description="Substrate height (metres)")],
    substrate_eps_r: Annotated[float, Field(description="Substrate relative permittivity")],
    freq_hz: Annotated[float, Field(description="Frequency (Hz)")],
    theta_deg: Annotated[float, Field(description="Scan angle theta (degrees)")] = 0.0,
    phi_deg: Annotated[float, Field(description="Scan angle phi (degrees)")] = 0.0,
    polarization: Annotated[str, Field(description="Polarization: 'H' or 'V'")] = "H",
) -> dict[str, Any]:
    """Compute the surface impedance of a unit cell at a single frequency."""
    try:
        from apab.core.schemas import (
            GeometryKind,
            GeometrySpec,
            LatticeSpec,
            UnitCellSpec,
        )
        from apab.coupling.sparams import EdgeFEMAdapter

        logger.info("Computing surface impedance at %.2e Hz", freq_hz)

        lattice = LatticeSpec(dx_m=period_x, dy_m=period_y)
        geom_params = {"substrate_h_m": substrate_height, "er": substrate_eps_r}
        geometry = GeometrySpec(kind=GeometryKind("patch"), params=geom_params)
        unit_cell = UnitCellSpec(lattice=lattice, geometry=geometry)

        adapter = EdgeFEMAdapter()
        z = adapter.surface_impedance(unit_cell, freq_hz, theta_deg, phi_deg, polarization)

        return {
            "impedance_real": float(z.real),
            "impedance_imag": float(z.imag),
            "impedance_mag": float(abs(z)),
            "freq_hz": freq_hz,
        }
    except Exception as e:
        logger.exception("edgefem_surface_impedance failed")
        return {"error": str(e), "status": "failed"}


@mcp.tool()
async def edgefem_export_touchstone(
    period_x: Annotated[float, Field(description="Unit-cell period in x (metres)")],
    period_y: Annotated[float, Field(description="Unit-cell period in y (metres)")],
    substrate_height: Annotated[float, Field(description="Substrate height (metres)")],
    substrate_eps_r: Annotated[float, Field(description="Substrate relative permittivity")],
    freq_start: Annotated[float, Field(description="Start frequency (Hz)")],
    freq_stop: Annotated[float, Field(description="Stop frequency (Hz)")],
    n_freq: Annotated[int, Field(description="Number of frequency points")],
    filepath: Annotated[str, Field(description="Output Touchstone file path")],
    theta_deg: Annotated[float, Field(description="Scan angle theta (degrees)")] = 0.0,
    phi_deg: Annotated[float, Field(description="Scan angle phi (degrees)")] = 0.0,
    polarization: Annotated[str, Field(description="Polarization: 'H' or 'V'")] = "H",
) -> dict[str, str]:
    """Run a frequency sweep and export results as a Touchstone file."""
    try:
        from apab.core.workspace import reject_path_traversal

        reject_path_traversal(filepath)

        from apab.core.schemas import (
            FreqRange,
            GeometryKind,
            GeometrySpec,
            LatticeSpec,
            ScanSpec,
            SweepSpec,
            UnitCellSpec,
        )
        from apab.coupling.sparams import EdgeFEMAdapter

        logger.info("Exporting Touchstone to %s", filepath)

        lattice = LatticeSpec(dx_m=period_x, dy_m=period_y)
        geom_params = {"substrate_h_m": substrate_height, "er": substrate_eps_r}
        geometry = GeometrySpec(kind=GeometryKind("patch"), params=geom_params)
        unit_cell = UnitCellSpec(lattice=lattice, geometry=geometry)

        sweep = SweepSpec(
            freq_hz=FreqRange(start=freq_start, stop=freq_stop, n=n_freq),
            scan=ScanSpec(theta_deg=[theta_deg], phi_deg=[phi_deg], n_theta=1, n_phi=1),
            polarization=[polarization],
        )

        adapter = EdgeFEMAdapter()
        freqs, r_array, t_array = adapter.run_frequency_sweep(
            unit_cell, sweep, theta=theta_deg, phi=phi_deg, pol=polarization,
        )
        adapter.export_touchstone(filepath, freqs, r_array, t_array)

        return {"filepath": filepath, "status": "exported"}
    except Exception as e:
        logger.exception("edgefem_export_touchstone failed")
        return {"error": str(e), "status": "failed"}
