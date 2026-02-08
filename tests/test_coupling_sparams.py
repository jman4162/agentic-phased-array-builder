"""Unit tests for the EdgeFEMAdapter (src/apab/coupling/sparams.py).

All tests mock ``edgefem.designs.UnitCellDesign`` so they run without the
real EdgeFEM solver installed.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from apab.core.schemas import (
    FreqRange,
    GeometrySpec,
    LatticeSpec,
    ScanSpec,
    SimRequest,
    SolverSpec,
    SweepSpec,
    UnitCellSpec,
)

# ---------------------------------------------------------------------------
# Install a fake ``edgefem`` package in sys.modules so that
# ``from edgefem.designs import UnitCellDesign`` resolves at import time
# even when the real package is not installed.
# ---------------------------------------------------------------------------

_edgefem = ModuleType("edgefem")
_edgefem_designs = ModuleType("edgefem.designs")
_MockUnitCellDesignCls = MagicMock(name="UnitCellDesign")
_edgefem_designs.UnitCellDesign = _MockUnitCellDesignCls  # type: ignore[attr-defined]
_edgefem.designs = _edgefem_designs  # type: ignore[attr-defined]
sys.modules.setdefault("edgefem", _edgefem)
sys.modules.setdefault("edgefem.designs", _edgefem_designs)

from apab.coupling.sparams import EdgeFEMAdapter  # noqa: E402

# ── Fixtures ──


@pytest.fixture
def unit_cell_spec() -> UnitCellSpec:
    """Return a minimal patch unit-cell spec."""
    return UnitCellSpec(
        lattice=LatticeSpec(type="rect", dx_m=0.005, dy_m=0.005),
        geometry=GeometrySpec(
            kind="patch",
            params={
                "patch_w_m": 0.003,
                "patch_l_m": 0.003,
                "substrate_h_m": 0.0005,
                "er": 2.2,
            },
        ),
    )


@pytest.fixture
def sweep_spec() -> SweepSpec:
    """Return a minimal sweep spec."""
    return SweepSpec(
        freq_hz=FreqRange(start=26e9, stop=30e9, n=5),
        scan=ScanSpec(theta_deg=[0, 60], phi_deg=[0, 90], n_theta=3, n_phi=2),
        polarization=["H"],
    )


@pytest.fixture
def mock_design() -> MagicMock:
    """Create a MagicMock mimicking an ``edgefem.designs.UnitCellDesign`` instance."""
    design = MagicMock()
    design.add_patch = MagicMock()
    design.generate_mesh = MagicMock()

    # frequency_sweep_reflection returns (freqs, R, T)
    freqs = np.linspace(26e9, 30e9, 5)
    r_array = np.full(5, 0.1 + 0.2j)
    t_array = np.full(5, 0.8 - 0.1j)
    design.frequency_sweep_reflection = MagicMock(return_value=(freqs, r_array, t_array))

    # reflection_transmission returns (R, T) scalars
    design.reflection_transmission = MagicMock(return_value=(0.1 + 0.2j, 0.8 - 0.1j))

    # surface_impedance returns a single complex value
    design.surface_impedance = MagicMock(return_value=(120.0 + 30.0j))

    # export_touchstone is a classmethod-like helper on the class
    design.export_touchstone = MagicMock()

    return design


# ── Tests ──


class TestBuildCell:
    def test_creates_design_with_correct_parameters(
        self,
        unit_cell_spec: UnitCellSpec,
        mock_design: MagicMock,
    ) -> None:
        """_build_cell should instantiate UnitCellDesign with lattice/substrate params."""
        with patch("edgefem.designs.UnitCellDesign", return_value=mock_design) as mock_cls:
            adapter = EdgeFEMAdapter()
            design = adapter._build_cell(unit_cell_spec)

            mock_cls.assert_called_once_with(
                period_x=0.005,
                period_y=0.005,
                substrate_height=0.0005,
                substrate_eps_r=2.2,
                substrate_tan_d=0.0,
                has_ground_plane=True,
            )
            mock_design.add_patch.assert_called_once_with(width=0.003, length=0.003)
            mock_design.generate_mesh.assert_called_once()
            assert design is mock_design

    def test_mesh_quality_mapping(
        self,
        unit_cell_spec: UnitCellSpec,
        mock_design: MagicMock,
    ) -> None:
        """Settings 'mesh_quality' should map to the correct density."""
        with patch("edgefem.designs.UnitCellDesign", return_value=mock_design):
            adapter = EdgeFEMAdapter(settings={"mesh_quality": "fine"})
            adapter._build_cell(unit_cell_spec)

        call_kwargs = mock_design.generate_mesh.call_args
        assert call_kwargs is not None
        # density should be passed as keyword arg
        density = call_kwargs.kwargs.get("density") or call_kwargs[1].get("density")
        assert density == 40.0


class TestRunFrequencySweep:
    def test_calls_correct_methods(
        self,
        unit_cell_spec: UnitCellSpec,
        sweep_spec: SweepSpec,
        mock_design: MagicMock,
    ) -> None:
        """run_frequency_sweep should delegate to design.frequency_sweep_reflection."""
        with patch("edgefem.designs.UnitCellDesign", return_value=mock_design):
            adapter = EdgeFEMAdapter()
            freqs, r_array, t_array = adapter.run_frequency_sweep(
                unit_cell_spec, sweep_spec, theta=15.0, phi=45.0, pol="TM"
            )

        mock_design.frequency_sweep_reflection.assert_called_once_with(
            f_start=26e9,
            f_stop=30e9,
            n_points=5,
            theta=15.0,
            phi=45.0,
            pol="TM",
            verbose=False,
        )
        assert len(freqs) == 5
        assert r_array is not None
        assert t_array is not None

    def test_returns_python_list_freqs(
        self,
        unit_cell_spec: UnitCellSpec,
        sweep_spec: SweepSpec,
        mock_design: MagicMock,
    ) -> None:
        """Returned frequencies should be a plain Python list."""
        with patch("edgefem.designs.UnitCellDesign", return_value=mock_design):
            adapter = EdgeFEMAdapter()
            freqs, _, _ = adapter.run_frequency_sweep(unit_cell_spec, sweep_spec)
        assert isinstance(freqs, list)


class TestRunSinglePoint:
    def test_returns_complex_tuple(
        self,
        unit_cell_spec: UnitCellSpec,
        mock_design: MagicMock,
    ) -> None:
        with patch("edgefem.designs.UnitCellDesign", return_value=mock_design):
            adapter = EdgeFEMAdapter()
            r, t = adapter.run_single_point(unit_cell_spec, freq=28e9, theta=0, phi=0)

        assert isinstance(r, complex)
        assert isinstance(t, complex)
        mock_design.reflection_transmission.assert_called_once_with(
            28e9, theta=0, phi=0, pol="TE"
        )


class TestRunFullSweep:
    def test_produces_correct_sim_result_structure(
        self,
        unit_cell_spec: UnitCellSpec,
        sweep_spec: SweepSpec,
        mock_design: MagicMock,
    ) -> None:
        """run_full_sweep should return a SimResult with matching dimensions."""
        from apab.core.schemas import SimResult

        with patch("edgefem.designs.UnitCellDesign", return_value=mock_design):
            adapter = EdgeFEMAdapter()
            request = SimRequest(
                unit_cell=unit_cell_spec,
                sweep=sweep_spec,
                solver=SolverSpec(backend="edgefem"),
            )
            result = adapter.run_full_sweep(request)

        # Check type.
        assert isinstance(result, SimResult)

        # n_theta=3, n_phi=2 -> 6 scan points.
        assert len(result.scan_points) == 6

        # 1 polarization.
        assert result.polarizations == ["H"]

        # 5 frequency points.
        assert len(result.freq_hz) == 5

        # s_params: [6 scan][1 pol][5 freq] -> each entry is [re, im].
        assert len(result.s_params) == 6
        assert len(result.s_params[0]) == 1  # 1 pol
        assert len(result.s_params[0][0]) == 5  # 5 freq points
        assert len(result.s_params[0][0][0]) == 2  # [re, im]

        # Metadata should identify the adapter.
        assert result.metadata["adapter"] == "EdgeFEMAdapter"


class TestExportTouchstone:
    def test_delegates_to_edgefem(self) -> None:
        with patch("edgefem.designs.UnitCellDesign") as mock_cls:
            adapter = EdgeFEMAdapter()
            freqs = [1e9, 2e9]
            r_arr = [0.1, 0.2]
            adapter.export_touchstone("out.s1p", freqs, r_arr)
            mock_cls.export_touchstone.assert_called_once_with("out.s1p", freqs, r_arr, None)


class TestRadiationPattern:
    def test_returns_empty_dict(self, unit_cell_spec: UnitCellSpec) -> None:
        adapter = EdgeFEMAdapter()
        # radiation_pattern is a stub -- should not call EdgeFEM.
        result = adapter.radiation_pattern(unit_cell_spec, freq=28e9)
        assert result == {}


# ── Integration marker example ──


@pytest.mark.integration
class TestEdgeFEMIntegration:
    """Integration tests that require a real EdgeFEM installation.

    Run with: ``pytest -m integration``
    """

    def test_real_frequency_sweep(self) -> None:
        """Placeholder -- requires edgefem to be installed."""
        pytest.skip("EdgeFEM not available in CI; run locally with -m integration")
