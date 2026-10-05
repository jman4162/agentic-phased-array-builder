"""Tests for MCP system-level tools."""

from __future__ import annotations

import json
import typing

import pytest
from phased_array_systems.scenarios import RadarDetectionScenario

from apab.core.schemas import ArraySpec, ScanPoint
from apab.mcp import tools_system
from apab.mcp.server import get_mcp
from apab.mcp.tools_system import system_evaluate, system_trade_study
from apab.system.wrappers_pas import PASSystemEngine

# A 16x16 X-band array used by most radar tests.
ARRAY = dict(nx=16, ny=16, dx_m=0.015, dy_m=0.015, tx_power_w_per_elem=2.0)
RADAR = dict(freq_hz=10e9, bandwidth_hz=50e6, range_m=20e3, scenario_type="radar")
COMMS = dict(freq_hz=10e9, bandwidth_hz=100e6, range_m=1000.0)


class TestSystemEvaluate:
    async def test_returns_metrics(self):
        result = await system_evaluate(
            nx=4, ny=4, dx_m=0.005, dy_m=0.005, tx_power_w_per_elem=0.1, **COMMS
        )
        assert "error" not in result, result.get("error")
        assert "eirp_dbw" in result

    async def test_output_is_strict_json(self):
        """PAS inf sentinels become null, with the original kept in nonfinite_metrics."""
        result = await system_evaluate(
            nx=4, ny=4, dx_m=0.005, dy_m=0.005, tx_power_w_per_elem=0.1, **COMMS
        )
        json.dumps(result, allow_nan=False)
        assert result["imd3_dbc"] is None
        assert result["nonfinite_metrics"]["imd3_dbc"] == "inf"

    async def test_radar_scenario(self):
        result = await system_evaluate(**ARRAY, **RADAR)
        assert "error" not in result, result.get("error")
        assert "pd_achieved" in result

    async def test_radar_defaults_match_direct_pas(self):
        """With no radar options set, the tool equals a direct PAS evaluation."""
        result = await system_evaluate(**ARRAY, **RADAR)

        engine = PASSystemEngine()
        arch = engine.build_architecture(
            ArraySpec(
                size=[16, 16],
                spacing_m=[0.015, 0.015],
                taper="uniform",
                steer=ScanPoint(theta_deg=0, phi_deg=0),
            ),
            {"tx_power_w_per_elem": 2.0, "freq_hz": 10e9},
        )
        scenario = RadarDetectionScenario(
            freq_hz=10e9, bandwidth_hz=50e6, range_m=20e3, target_rcs_dbsm=0.0
        )
        expected = engine.evaluate(arch, scenario)
        for key in ("snr_single_pulse_db", "pd_achieved", "detection_range_m", "g_ant_db"):
            assert result[key] == pytest.approx(expected[key]), key
        assert "timeline_occupancy" not in result


class TestRadarOptions:
    async def test_radar_options_forwarded(self):
        result = await system_evaluate(
            **ARRAY,
            **RADAR,
            pd_required=0.9,
            pfa=1e-6,
            n_pulses=16,
            swerling=1,
            integration_type="noncoherent",
            duty_cycle=0.1,
            clutter_type="sea",
            sea_state=3,
            antenna_height_m=30.0,
            polarization="VV",
            cfar_type="CA",
            cfar_ref_cells=16,
            prf_hz=2000.0,
            search_az_extent_deg=90.0,
            search_el_extent_deg=30.0,
            search_frame_time_ms=2000.0,
        )
        assert "error" not in result, result.get("error")
        assert result["swerling"] == 1
        assert result["n_pulses"] == 16
        assert result["pd_required"] == 0.9
        assert 0.0 <= result["pd_achieved"] <= 1.0
        assert result["clutter_type"] == "sea"
        assert result["cfar_type"] == "CA"
        assert result["cfar_loss_db"] > 0
        assert "scnr_db" in result
        assert "search_frame_time_s" in result
        assert "timeline_occupancy" in result

    async def test_clutter_geometry_changes_result(self):
        low = await system_evaluate(**ARRAY, **RADAR, clutter_type="sea", antenna_height_m=5.0)
        high = await system_evaluate(**ARRAY, **RADAR, clutter_type="sea", antenna_height_m=300.0)
        assert low["clutter_rcs_dbsm"] != pytest.approx(high["clutter_rcs_dbsm"])

    async def test_rain_requires_rate(self):
        result = await system_evaluate(**ARRAY, **RADAR, clutter_type="rain")
        assert result["status"] == "failed"
        assert "rain_rate_mm_hr" in result["error"]

    async def test_rain_clutter_engages(self):
        result = await system_evaluate(
            **ARRAY, **RADAR, clutter_type="rain", rain_rate_mm_hr=10.0
        )
        assert "error" not in result, result.get("error")
        assert result["rain_loss_db"] > 0
        assert result["clutter_rcs_dbsm"] > -90

    async def test_partial_timeline_rejected(self):
        result = await system_evaluate(
            **ARRAY, **RADAR, search_az_extent_deg=90.0, search_el_extent_deg=30.0
        )
        assert result["status"] == "failed"
        assert "prf_hz" in result["error"]

    async def test_frame_time_alone_rejected(self):
        result = await system_evaluate(**ARRAY, **RADAR, search_frame_time_ms=1000.0)
        assert result["status"] == "failed"
        assert "search_frame_time_ms" in result["error"]

    async def test_radar_options_rejected_for_comms(self):
        result = await system_evaluate(**ARRAY, **COMMS, swerling=1, pd_required=0.9)
        assert result["status"] == "failed"
        assert "swerling" in result["error"] and "pd_required" in result["error"]


class TestScanAngle:
    async def test_comms_scan_angle_forwarded(self):
        boresight = await system_evaluate(**ARRAY, **COMMS)
        scanned = await system_evaluate(**ARRAY, **COMMS, scan_angle_deg=45.0)
        assert "error" not in scanned, scanned.get("error")
        assert scanned["scan_loss_db"] == pytest.approx(1.505, abs=0.01)
        assert boresight["eirp_dbw"] - scanned["eirp_dbw"] == pytest.approx(
            scanned["scan_loss_db"], abs=0.01
        )

    async def test_radar_scan_loss_counted_once(self):
        """Needs phased-array-systems>=0.14.1; 0.14.0 and earlier drop 4x."""
        boresight = await system_evaluate(**ARRAY, **RADAR)
        scanned = await system_evaluate(**ARRAY, **RADAR, scan_angle_deg=60.0)
        assert "error" not in scanned, scanned.get("error")
        # Gain appears twice in the radar equation, so SNR drops by 2x scan loss.
        drop = boresight["snr_single_pulse_db"] - scanned["snr_single_pulse_db"]
        assert drop == pytest.approx(2 * scanned["scan_loss_db"], abs=0.01)


class TestToolSchema:
    async def test_scenario_type_validated_at_mcp_layer(self):
        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError):
            await get_mcp().call_tool(
                "system_evaluate",
                {
                    "nx": 4, "ny": 4, "dx_m": 0.005, "dy_m": 0.005,
                    "tx_power_w_per_elem": 0.1, **COMMS, "scenario_type": "Radar",
                },
            )

    async def test_schema_carries_enums_and_bounds(self):
        tools = {t.name: t for t in await get_mcp().list_tools()}
        props = tools["system_evaluate"].inputSchema["properties"]
        assert props["scenario_type"]["enum"] == ["comms", "radar"]
        scan = next(s for s in props["scan_angle_deg"]["anyOf"] if s.get("type") == "number")
        assert scan["exclusiveMaximum"] == 90

    def test_radar_options_match_pas(self):
        """Every forwarded option exists on the PAS scenario with the same choices."""
        fields = RadarDetectionScenario.model_fields
        hints = typing.get_type_hints(system_evaluate, include_extras=True)
        for name in (*tools_system._RADAR_OPTIONS, "scan_angle_deg"):
            assert name in fields, name
            pas_choices = _literal_values(fields[name].annotation)
            if pas_choices:
                assert _literal_values(hints[name]) == pas_choices, name

    def test_both_tools_expose_same_radar_options(self):
        ev = typing.get_type_hints(system_evaluate, include_extras=True)
        ts = typing.get_type_hints(system_trade_study, include_extras=True)
        for name in tools_system._RADAR_OPTIONS:
            assert ev[name] == ts[name], name


def _literal_values(tp: object) -> set[object]:
    """Collect Literal choices from a (possibly Annotated / Optional) type."""
    if typing.get_origin(tp) is typing.Literal:
        return set(typing.get_args(tp))
    out: set[object] = set()
    for arg in typing.get_args(tp):
        out |= _literal_values(arg)
    return out


class TestSystemTradeStudy:
    async def test_small_study(self):
        result = await system_trade_study(
            **COMMS,
            n_samples=5,
            seed=42,
            variables=[
                {"name": "array.nx", "type": "int", "low": 4, "high": 8},
                {"name": "array.ny", "type": "int", "low": 4, "high": 8},
                {"name": "rf.tx_power_w_per_elem", "type": "float", "low": 0.05, "high": 0.5},
                {"name": "array.enforce_subarray_constraint", "type": "categorical",
                 "values": [False]},
            ],
        )
        assert result["status"] == "completed", result.get("error")
        assert result["n_total"] == 5
        assert result["n_failed"] == 0, result["first_error"]

    async def test_failed_cases_not_counted_feasible(self):
        """Cases that raise (here: ny missing) are reported, not counted feasible."""
        result = await system_trade_study(
            **COMMS,
            n_samples=3,
            seed=0,
            variables=[{"name": "array.nx", "type": "int", "low": 4, "high": 8}],
        )
        assert result["status"] == "completed"
        assert result["n_failed"] == 3
        assert result["n_feasible"] == 0
        assert "ny" in result["first_error"]

    async def test_pareto_filters_dominated_designs(self):
        result = await system_trade_study(
            **COMMS,
            n_samples=20,
            seed=0,
            variables=[
                {"name": "array.nx", "type": "int", "low": 4, "high": 16},
                {"name": "array.ny", "type": "int", "low": 4, "high": 16},
                {"name": "rf.tx_power_w_per_elem", "type": "float", "low": 0.05, "high": 2.0},
                {"name": "array.enforce_subarray_constraint", "type": "categorical",
                 "values": [False]},
            ],
        )
        assert result["status"] == "completed", result.get("error")
        assert result["n_failed"] == 0, result["first_error"]
        assert result["pareto_objectives"] == ["cost_usd", "eirp_dbw"]
        assert 0 < result["pareto_count"] < result["n_feasible"]

    async def test_radar_study_forwards_options(self):
        result = await system_trade_study(
            **RADAR,
            n_samples=4,
            seed=1,
            pd_required=0.8,
            swerling=1,
            clutter_type="sea",
            variables=[
                {"name": "array.nx", "type": "int", "low": 8, "high": 16},
                {"name": "array.ny", "type": "int", "low": 8, "high": 16},
                {"name": "rf.tx_power_w_per_elem", "type": "float", "low": 0.5, "high": 2.0},
                {"name": "array.enforce_subarray_constraint", "type": "categorical",
                 "values": [False]},
            ],
        )
        assert result["status"] == "completed", result.get("error")
        assert result["n_total"] == 4
        assert result["n_failed"] == 0, result["first_error"]
        assert result["pareto_objectives"] == ["cost_usd", "snr_margin_db"]

    async def test_comms_study_rejects_radar_options(self):
        result = await system_trade_study(**COMMS, swerling=1, n_samples=2)
        assert result["status"] == "failed"
        assert "swerling" in result["error"]
