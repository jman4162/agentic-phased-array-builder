"""EdgeFEM adapter for unit-cell S-parameter simulation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from apab.core.schemas import (
    ScanPoint,
    SimRequest,
    SimResult,
    SweepSpec,
    UnitCellSpec,
)

logger = logging.getLogger(__name__)

# Mapping from human-readable mesh quality strings to EdgeFEM density values.
_MESH_QUALITY_MAP: dict[str, float] = {
    "coarse": 10.0,
    "medium": 20.0,
    "fine": 40.0,
    "very_fine": 80.0,
}


class EdgeFEMAdapter:
    """Adapter bridging APAB schemas to the edgefem full-wave solver.

    This class translates :class:`UnitCellSpec` / :class:`SweepSpec` objects
    into ``edgefem.designs.UnitCellDesign`` calls, runs frequency sweeps and
    single-point evaluations, and packages results as :class:`SimResult`.
    """

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self.settings: dict[str, Any] = settings or {}
        quality_str = str(self.settings.get("mesh_quality", "medium"))
        self._mesh_density: float = _MESH_QUALITY_MAP.get(
            quality_str, float(quality_str) if quality_str.replace(".", "", 1).isdigit() else 20.0
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_cell(self, spec: UnitCellSpec) -> Any:
        """Create an ``edgefem.designs.UnitCellDesign`` from an APAB spec.

        Parameters
        ----------
        spec:
            APAB unit-cell specification containing lattice, geometry and
            substrate information.

        Returns
        -------
        UnitCellDesign
            A fully meshed EdgeFEM design object ready for simulation.
        """
        from edgefem.designs import UnitCellDesign

        params = spec.geometry.params

        # Substrate parameters (required for patch geometry).
        substrate_h = params.get("substrate_h_m", 0.0005)
        er = params.get("er", 2.2)
        tan_d = params.get("tan_d", 0.0)
        has_ground = params.get("has_ground_plane", True)

        design = UnitCellDesign(
            period_x=spec.lattice.dx_m,
            period_y=spec.lattice.dy_m,
            substrate_height=substrate_h,
            substrate_eps_r=er,
            substrate_tan_d=tan_d,
            has_ground_plane=has_ground,
        )

        # Add geometry elements.
        kind = spec.geometry.kind
        if kind == "patch":
            patch_w = params.get("patch_w_m", spec.lattice.dx_m * 0.6)
            patch_l = params.get("patch_l_m", spec.lattice.dy_m * 0.6)
            design.add_patch(width=patch_w, length=patch_l)
        else:
            logger.warning("Geometry kind '%s' not yet mapped; skipping element creation.", kind)

        # Mesh the design.
        design_freq = params.get("design_freq", 10e9)
        design.generate_mesh(density=self._mesh_density, design_freq=design_freq)

        return design

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_frequency_sweep(
        self,
        spec: UnitCellSpec,
        sweep: SweepSpec,
        theta: float = 0.0,
        phi: float = 0.0,
        pol: str = "TE",
    ) -> tuple[list[float], list[Any], list[Any]]:
        """Run a frequency sweep for a single (theta, phi, pol) combination.

        Returns
        -------
        tuple
            ``(freqs, R_array, T_array)`` where *freqs* is a list of
            frequency values in Hz and *R_array* / *T_array* are the
            complex reflection / transmission arrays returned by EdgeFEM.
        """
        design = self._build_cell(spec)
        freqs, r_array, t_array = design.frequency_sweep_reflection(
            f_start=sweep.freq_hz.start,
            f_stop=sweep.freq_hz.stop,
            n_points=sweep.freq_hz.n,
            theta=theta,
            phi=phi,
            pol=pol,
            verbose=False,
        )
        # Ensure freqs is a plain Python list for schema compatibility.
        if hasattr(freqs, "tolist"):
            freqs = freqs.tolist()
        return freqs, r_array, t_array

    def run_single_point(
        self,
        spec: UnitCellSpec,
        freq: float,
        theta: float = 0.0,
        phi: float = 0.0,
        pol: str = "TE",
    ) -> tuple[complex, complex]:
        """Evaluate reflection and transmission at a single operating point.

        Returns
        -------
        tuple
            ``(R, T)`` complex reflection and transmission coefficients.
        """
        design = self._build_cell(spec)
        r, t = design.reflection_transmission(freq, theta=theta, phi=phi, pol=pol)
        return r, t

    def surface_impedance(
        self,
        spec: UnitCellSpec,
        freq: float,
        theta: float = 0.0,
        phi: float = 0.0,
        pol: str = "TE",
    ) -> complex:
        """Return the surface impedance of the unit cell at a single point."""
        design = self._build_cell(spec)
        return design.surface_impedance(freq, theta=theta, phi=phi, pol=pol)  # type: ignore[no-any-return]

    def export_touchstone(
        self,
        path: str,
        freqs: Any,
        r_array: Any,
        t_array: Any | None = None,
    ) -> None:
        """Write simulation results to a Touchstone file via EdgeFEM.

        Parameters
        ----------
        path:
            Destination file path (e.g. ``"out.s1p"``).
        freqs:
            Frequency vector.
        r_array:
            Complex reflection data.
        t_array:
            Complex transmission data (optional).
        """
        from edgefem.designs import UnitCellDesign

        # UnitCellDesign provides a static-like export helper.
        UnitCellDesign.export_touchstone(path, freqs, r_array, t_array)

    def radiation_pattern(
        self,
        spec: UnitCellSpec,
        freq: float,
        n_theta: int = 91,
        n_phi: int = 73,
    ) -> dict[str, Any]:
        """Return a radiation pattern dict (stub).

        Unit-cell simulations do not directly produce far-field radiation
        patterns.  This method is reserved for future integration with
        embedded-element pattern extraction.
        """
        return {}

    # ------------------------------------------------------------------
    # Full multi-dimensional sweep
    # ------------------------------------------------------------------

    def run_full_sweep(self, request: SimRequest) -> SimResult:
        """Execute a complete parameter sweep over frequency, scan angle, and polarization.

        The method iterates over every combination of
        ``(theta, phi, polarization)`` defined in ``request.sweep``, runs a
        frequency sweep for each, and assembles the results into a single
        :class:`SimResult`.

        The ``s_params`` field of the returned :class:`SimResult` is a nested
        list with the layout ``[scan_idx][pol_idx][freq_idx]`` where each
        innermost element is a complex reflection coefficient encoded as a
        two-element ``[real, imag]`` list (for JSON serialisability).
        """
        spec = request.unit_cell
        sweep = request.sweep

        # Build scan grid.
        scan = sweep.scan
        thetas = np.linspace(scan.theta_deg[0], scan.theta_deg[-1], scan.n_theta).tolist()
        phis = np.linspace(scan.phi_deg[0], scan.phi_deg[-1], scan.n_phi).tolist()

        scan_points: list[ScanPoint] = []
        for theta in thetas:
            for phi_val in phis:
                scan_points.append(ScanPoint(theta_deg=theta, phi_deg=phi_val))

        pols = sweep.polarization
        # Map APAB polarization labels to EdgeFEM conventions.
        pol_map = {"H": "TE", "V": "TM", "TE": "TE", "TM": "TM"}

        all_freqs: list[float] | None = None
        s_params: list[list[list[list[float]]]] = []  # [scan][pol][freq] -> [re, im]

        for sp in scan_points:
            pol_data: list[list[list[float]]] = []
            for pol_label in pols:
                efem_pol = pol_map.get(pol_label, pol_label)
                freqs, r_array, _t_array = self.run_frequency_sweep(
                    spec, sweep, theta=sp.theta_deg, phi=sp.phi_deg, pol=efem_pol
                )
                if all_freqs is None:
                    all_freqs = list(freqs)

                # Convert complex reflection coefficients to [re, im] pairs.
                freq_data: list[list[float]] = []
                for r_val in r_array:
                    c = complex(r_val)
                    freq_data.append([c.real, c.imag])
                pol_data.append(freq_data)
            s_params.append(pol_data)

        return SimResult(
            freq_hz=all_freqs or [],
            scan_points=scan_points,
            polarizations=pols,
            s_params=s_params,
            metadata={
                "adapter": "EdgeFEMAdapter",
                "mesh_density": self._mesh_density,
                "settings": self.settings,
            },
        )
