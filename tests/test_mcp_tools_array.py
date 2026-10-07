"""Tests for MCP array-pattern tools."""

from __future__ import annotations

import os

import pytest

from apab.mcp.tools_array import (
    pattern_compute,
    pattern_multi_beam,
    pattern_null_steer,
    pattern_plot_3d,
    pattern_plot_cuts,
)


@pytest.mark.asyncio
class TestPatternCompute:
    async def test_returns_directivity(self):
        result = await pattern_compute(
            nx=4, ny=4, dx_m=0.005, dy_m=0.005, freq_hz=10e9,
        )
        assert "directivity_dbi" in result
        assert isinstance(result["directivity_dbi"], float)

    async def test_with_steering(self):
        result = await pattern_compute(
            nx=4, ny=4, dx_m=0.005, dy_m=0.005, freq_hz=10e9,
            theta0=15.0, phi0=0.0,
        )
        assert result["metadata"]["theta0_deg"] == 15.0

    async def test_with_taper(self):
        result = await pattern_compute(
            nx=4, ny=4, dx_m=0.005, dy_m=0.005, freq_hz=10e9,
            taper="taylor",
        )
        assert result["directivity_dbi"] is not None


@pytest.mark.asyncio
class TestPatternPlotCuts:
    async def test_saves_plot(self, tmp_path):
        path = tmp_path / "cuts.png"
        result = await pattern_plot_cuts(
            nx=4, ny=4, dx_m=0.005, dy_m=0.005, freq_hz=10e9,
            output_path=str(path),
        )
        assert result["status"] == "saved"
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    async def test_relative_path_goes_to_artifacts(self, tmp_path):
        result = await pattern_plot_cuts(
            nx=4, ny=4, dx_m=0.005, dy_m=0.005, freq_hz=10e9, output_path="cuts.png",
        )
        assert result["status"] == "saved"
        assert result["output_path"] == str((tmp_path / "artifacts" / "cuts.png").resolve())
        assert os.path.exists(result["output_path"])

    async def test_absolute_path_outside_workspace_refused(self, tmp_path):
        outside = tmp_path.parent / f"{tmp_path.name}_outside" / "cuts.png"
        result = await pattern_plot_cuts(
            nx=4, ny=4, dx_m=0.005, dy_m=0.005, freq_hz=10e9, output_path=str(outside),
        )
        assert result["status"] == "failed"
        assert "outside the allowed root" in result["error"]
        assert not outside.exists()


@pytest.mark.asyncio
class TestPatternPlot3D:
    async def test_saves_plot(self, tmp_path):
        path = tmp_path / "pattern3d.png"
        result = await pattern_plot_3d(
            nx=4, ny=4, dx_m=0.005, dy_m=0.005, freq_hz=10e9,
            output_path=str(path),
        )
        assert result["status"] == "saved"
        assert os.path.exists(path)


@pytest.mark.asyncio
class TestPatternMultiBeam:
    async def test_multi_beam(self):
        result = await pattern_multi_beam(
            nx=8, ny=8, dx_m=0.005, dy_m=0.005, freq_hz=10e9,
            beam_directions=[[0.0, 0.0], [15.0, 0.0]],
        )
        assert result["n_beams"] == 2
        assert "directivity_dbi" in result


@pytest.mark.asyncio
class TestPatternNullSteer:
    async def test_null_steer(self):
        result = await pattern_null_steer(
            nx=8, ny=8, dx_m=0.005, dy_m=0.005, freq_hz=10e9,
            theta0=0.0, phi0=0.0,
            null_directions=[[30.0, 0.0]],
        )
        assert result["n_nulls"] == 1
        assert "directivity_dbi" in result


@pytest.mark.asyncio
class TestToolContract:
    """The two convention contract tests, back-ported from antenna-cad."""

    async def test_errors_returned_not_raised(self):
        """A failing tool returns {"error", "status": "failed"}, never raises."""
        result = await pattern_plot_cuts(
            nx=0, ny=0, dx_m=0.005, dy_m=0.005, freq_hz=10e9,
            output_path="/nonexistent-root/nowhere/out.png",
        )
        assert result["status"] == "failed"
        assert "error" in result

    async def test_path_traversal_rejected(self):
        result = await pattern_plot_cuts(
            nx=4, ny=4, dx_m=0.005, dy_m=0.005, freq_hz=10e9,
            output_path="../../etc/evil.png",
        )
        assert result["status"] == "failed"
        assert "traversal" in result["error"].lower() or ".." in result["error"]
