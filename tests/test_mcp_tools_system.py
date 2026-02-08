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
