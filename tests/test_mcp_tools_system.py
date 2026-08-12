"""Tests for MCP system-level tools."""

from __future__ import annotations

import pytest

from apab.mcp.tools_system import system_evaluate, system_trade_study


@pytest.mark.asyncio
class TestSystemEvaluate:
    async def test_returns_metrics(self):
        result = await system_evaluate(
            nx=4,
            ny=4,
            dx_m=0.005,
            dy_m=0.005,
            freq_hz=10e9,
            bandwidth_hz=100e6,
            range_m=1000.0,
            tx_power_w_per_elem=0.1,
        )
        assert isinstance(result, dict)
        # Should contain some metric keys from PAS
        assert len(result) > 0

    async def test_radar_scenario(self):
        result = await system_evaluate(
            nx=4,
            ny=4,
            dx_m=0.005,
            dy_m=0.005,
            freq_hz=10e9,
            bandwidth_hz=100e6,
            range_m=1000.0,
            tx_power_w_per_elem=0.1,
            scenario_type="radar",
            target_rcs_dbsm=0.0,
        )
        assert isinstance(result, dict)


@pytest.mark.asyncio
class TestSystemTradeStudy:
    async def test_small_study(self):
        result = await system_trade_study(
            freq_hz=10e9,
            bandwidth_hz=100e6,
            range_m=1000.0,
            n_samples=5,
            seed=42,
            variables=[
                {"name": "array.nx", "type": "int", "low": 4, "high": 8},
                {"name": "rf.tx_power_w_per_elem", "type": "float", "low": 0.05, "high": 0.5},
            ],
        )
        assert result["status"] == "completed"
        assert "n_feasible" in result


class TestRadarSurface:
    """The full radar detection surface is reachable over MCP."""

    async def test_radar_options_forwarded(self):
        result = await system_evaluate(
            nx=16,
            ny=16,
            dx_m=0.015,
            dy_m=0.015,
            freq_hz=10e9,
            bandwidth_hz=50e6,
            range_m=20e3,
            tx_power_w_per_elem=2.0,
            scenario_type="radar",
            target_rcs_dbsm=0.0,
            pd_required=0.9,
            pfa=1e-6,
            n_pulses=16,
            swerling=1,
            integration_type="noncoherent",
            duty_cycle=0.1,
            clutter_type="sea",
            sea_state=3,
            cfar_type="CA",
            cfar_ref_cells=16,
            prf_hz=2000.0,
            search_az_extent_deg=90.0,
            search_el_extent_deg=30.0,
            search_frame_time_ms=2000.0,
        )
        assert "error" not in result, result.get("error")
        # Detection statistics ran with the requested configuration
        assert result["swerling"] == 1
        assert result["n_pulses"] == 16
        assert result["pd_required"] == 0.9
        assert 0.0 <= result["pd_achieved"] <= 1.0
        # Clutter and CFAR engaged
        assert result["clutter_type"] == "sea"
        assert result["cfar_type"] == "CA"
        assert result["cfar_loss_db"] > 0
        assert "scnr_db" in result
        # Search timeline engaged
        assert "search_frame_time_s" in result
        assert "timeline_occupancy" in result

    async def test_radar_defaults_unchanged(self):
        """Not passing radar options reproduces the scenario-model defaults."""
        result = await system_evaluate(
            nx=8,
            ny=8,
            dx_m=0.015,
            dy_m=0.015,
            freq_hz=10e9,
            bandwidth_hz=10e6,
            range_m=10e3,
            tx_power_w_per_elem=1.0,
            scenario_type="radar",
        )
        assert "error" not in result
        assert "pd_achieved" in result
        # Timeline metrics absent without prf/search extents (model defaults)
        assert "timeline_occupancy" not in result
