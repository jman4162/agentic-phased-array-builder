"""Tests for MCP array-pattern tools."""

from __future__ import annotations

import os
import tempfile

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
    async def test_saves_plot(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            result = await pattern_plot_cuts(
                nx=4, ny=4, dx_m=0.005, dy_m=0.005, freq_hz=10e9,
                output_path=path,
            )
            assert result["status"] == "saved"
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)


@pytest.mark.asyncio
class TestPatternPlot3D:
    async def test_saves_plot(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            result = await pattern_plot_3d(
                nx=4, ny=4, dx_m=0.005, dy_m=0.005, freq_hz=10e9,
                output_path=path,
            )
            assert result["status"] == "saved"
            assert os.path.exists(path)
        finally:
            os.unlink(path)


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
