"""Tests for MCP EdgeFEM tools (mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from apab.mcp.tools_edgefem import (
    edgefem_export_touchstone,
    edgefem_run_unit_cell,
    edgefem_surface_impedance,
)


def _mock_adapter():
    """Create a mock EdgeFEMAdapter with sensible return values."""
    adapter = MagicMock()

    # Mock run_full_sweep
    mock_result = MagicMock()
    mock_result.freq_hz = [9e9, 9.5e9, 10e9]
    mock_result.scan_points = []
    mock_result.polarizations = ["H"]
    mock_result.metadata = {"solver": "edgefem"}
    adapter.run_full_sweep.return_value = mock_result

    # Mock surface_impedance
    adapter.surface_impedance.return_value = complex(50.0, 10.0)

    # Mock run_frequency_sweep (returns (freqs, r_array, t_array))
    adapter.run_frequency_sweep.return_value = (
        [9e9, 9.5e9, 10e9],
        [0.1 + 0.01j, 0.2 + 0.02j, 0.15 + 0.01j],
        [0.9 + 0.01j, 0.8 + 0.02j, 0.85 + 0.01j],
    )

    # Mock export_touchstone
    adapter.export_touchstone.return_value = None

    return adapter


_PATCH_TARGET = "apab.coupling.sparams.EdgeFEMAdapter"


@pytest.mark.asyncio
class TestEdgefemRunUnitCell:
    async def test_returns_result(self):
        mock_instance = _mock_adapter()
        with patch(_PATCH_TARGET, return_value=mock_instance):
            result = await edgefem_run_unit_cell(
                period_x=0.005,
                period_y=0.005,
                substrate_height=0.001,
                substrate_eps_r=3.5,
                freq_start=9e9,
                freq_stop=11e9,
                n_freq=3,
            )
        assert result["status"] == "completed"
        assert len(result["freq_hz"]) == 3

    async def test_default_params(self):
        mock_instance = _mock_adapter()
        with patch(_PATCH_TARGET, return_value=mock_instance):
            result = await edgefem_run_unit_cell(
                period_x=0.005,
                period_y=0.005,
                substrate_height=0.001,
                substrate_eps_r=3.5,
                freq_start=9e9,
                freq_stop=11e9,
                n_freq=3,
            )
        assert result["status"] == "completed"


@pytest.mark.asyncio
class TestEdgefemSurfaceImpedance:
    async def test_returns_impedance(self):
        mock_instance = _mock_adapter()
        with patch(_PATCH_TARGET, return_value=mock_instance):
            result = await edgefem_surface_impedance(
                period_x=0.005,
                period_y=0.005,
                substrate_height=0.001,
                substrate_eps_r=3.5,
                freq_hz=10e9,
            )
        assert "impedance_real" in result
        assert "impedance_imag" in result
        assert result["impedance_real"] == pytest.approx(50.0)
        assert result["impedance_imag"] == pytest.approx(10.0)


@pytest.mark.asyncio
class TestEdgefemExportTouchstone:
    async def test_returns_exported(self):
        mock_instance = _mock_adapter()
        with patch(_PATCH_TARGET, return_value=mock_instance):
            result = await edgefem_export_touchstone(
                period_x=0.005,
                period_y=0.005,
                substrate_height=0.001,
                substrate_eps_r=3.5,
                freq_start=9e9,
                freq_stop=11e9,
                n_freq=3,
                filepath="/tmp/test.s1p",
            )
        assert result["status"] == "exported"
