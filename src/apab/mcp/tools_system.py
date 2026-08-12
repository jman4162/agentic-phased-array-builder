"""MCP tools for system-level analysis (phased-array-systems)."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from pydantic import Field

from apab.mcp.server import get_mcp

logger = logging.getLogger(__name__)
mcp = get_mcp()


@mcp.tool()
async def system_evaluate(
    nx: Annotated[int, Field(description="Number of elements in x")],
    ny: Annotated[int, Field(description="Number of elements in y")],
    dx_m: Annotated[float, Field(description="Element spacing in x (metres)")],
    dy_m: Annotated[float, Field(description="Element spacing in y (metres)")],
    freq_hz: Annotated[float, Field(description="Operating frequency (Hz)")],
    bandwidth_hz: Annotated[float, Field(description="System bandwidth (Hz)")],
    range_m: Annotated[float, Field(description="Link range (metres)")],
    tx_power_w_per_elem: Annotated[float, Field(description="Tx power per element (W)")],
    scenario_type: Annotated[
        str, Field(description="Scenario type: 'comms' or 'radar'")
    ] = "comms",
    required_snr_db: Annotated[float, Field(description="Required SNR (dB, comms)")] = 10.0,
    target_rcs_dbsm: Annotated[float, Field(description="Target RCS (dBsm, radar)")] = 0.0,
    pd_required: Annotated[
        float | None, Field(description="Required detection probability (radar)")
    ] = None,
    pfa: Annotated[
        float | None, Field(description="False-alarm probability (radar)")
    ] = None,
    n_pulses: Annotated[
        int | None, Field(description="Pulses integrated per dwell (radar)")
    ] = None,
    swerling: Annotated[
        int | None, Field(description="Swerling fluctuation model 0-4 (radar)")
    ] = None,
    integration_type: Annotated[
        str | None, Field(description="'coherent' or 'noncoherent' (radar)")
    ] = None,
    duty_cycle: Annotated[
        float | None, Field(description="Transmit duty cycle 0-1 (radar)")
    ] = None,
    scan_angle_deg: Annotated[
        float | None, Field(description="Scan angle off boresight (deg)")
    ] = None,
    clutter_type: Annotated[
        str | None, Field(description="'none', 'sea', 'ground', or 'rain' (radar)")
    ] = None,
    sea_state: Annotated[
        int | None, Field(description="Sea state 0-6 for sea clutter (radar)")
    ] = None,
    terrain_type: Annotated[
        str | None, Field(description="Terrain type for ground clutter (radar)")
    ] = None,
    cfar_type: Annotated[
        str | None, Field(description="'none', 'CA', 'OS', 'GO', or 'SO' (radar)")
    ] = None,
    cfar_ref_cells: Annotated[
        int | None, Field(description="CFAR reference cells (radar)")
    ] = None,
    prf_hz: Annotated[
        float | None, Field(description="Pulse repetition frequency (Hz, radar)")
    ] = None,
    search_az_extent_deg: Annotated[
        float | None, Field(description="Search azimuth extent (deg, radar)")
    ] = None,
    search_el_extent_deg: Annotated[
        float | None, Field(description="Search elevation extent (deg, radar)")
    ] = None,
    search_frame_time_ms: Annotated[
        float | None, Field(description="Search frame time budget (ms, radar)")
    ] = None,
    taper: Annotated[str, Field(description="Taper window name")] = "uniform",
    requirements: Annotated[
        list[dict[str, Any]] | None,
        Field(description="Optional list of requirement dicts"),
    ] = None,
) -> dict[str, Any]:
    """Evaluate a phased-array system architecture against a scenario.

    Radar scenarios accept the full detection surface: Swerling model,
    Pd/Pfa, pulse integration, clutter (sea/ground/rain), CFAR, and
    search-timeline parameters. Unset radar options use the scenario
    model's defaults.
    """
    try:
        from apab.core.schemas import ArraySpec, ScanPoint
        from apab.system.wrappers_pas import PASSystemEngine

        logger.info(
            "Evaluating system: %dx%d, %s scenario @ %.2e Hz",
            nx, ny, scenario_type, freq_hz,
        )

        spec = ArraySpec(
            size=[nx, ny],
            spacing_m=[dx_m, dy_m],
            taper=taper,
            steer=ScanPoint(theta_deg=0, phi_deg=0),
        )
        rf_spec = {
            "tx_power_w_per_elem": tx_power_w_per_elem,
            "freq_hz": freq_hz,
        }

        engine = PASSystemEngine()
        arch = engine.build_architecture(spec, rf_spec)

        scenario: Any
        if scenario_type == "radar":
            radar_kwargs = {
                k: v
                for k, v in {
                    "pd_required": pd_required,
                    "pfa": pfa,
                    "n_pulses": n_pulses,
                    "swerling": swerling,
                    "integration_type": integration_type,
                    "duty_cycle": duty_cycle,
                    "scan_angle_deg": scan_angle_deg,
                    "clutter_type": clutter_type,
                    "sea_state": sea_state,
                    "terrain_type": terrain_type,
                    "cfar_type": cfar_type,
                    "cfar_ref_cells": cfar_ref_cells,
                    "prf_hz": prf_hz,
                    "search_az_extent_deg": search_az_extent_deg,
                    "search_el_extent_deg": search_el_extent_deg,
                    "search_frame_time_ms": search_frame_time_ms,
                }.items()
                if v is not None
            }
            scenario = engine.build_radar_scenario(
                freq_hz=freq_hz,
                bandwidth_hz=bandwidth_hz,
                range_m=range_m,
                target_rcs_dbsm=target_rcs_dbsm,
                **radar_kwargs,
            )
        else:
            scenario = engine.build_comms_scenario(
                freq_hz=freq_hz,
                bandwidth_hz=bandwidth_hz,
                range_m=range_m,
                required_snr_db=required_snr_db,
            )

        metrics = engine.evaluate(arch, scenario, requirements)
        logger.info("System evaluation completed")
        return metrics
    except Exception as e:
        logger.exception("system_evaluate failed")
        return {"error": str(e), "status": "failed"}


@mcp.tool()
async def system_trade_study(
    freq_hz: Annotated[float, Field(description="Operating frequency (Hz)")],
    bandwidth_hz: Annotated[float, Field(description="System bandwidth (Hz)")],
    range_m: Annotated[float, Field(description="Link range (metres)")],
    scenario_type: Annotated[
        str, Field(description="Scenario type: 'comms' or 'radar'")
    ] = "comms",
    required_snr_db: Annotated[float, Field(description="Required SNR (dB, comms)")] = 10.0,
    target_rcs_dbsm: Annotated[float, Field(description="Target RCS (dBsm, radar)")] = 0.0,
    pd_required: Annotated[
        float | None, Field(description="Required detection probability (radar)")
    ] = None,
    pfa: Annotated[
        float | None, Field(description="False-alarm probability (radar)")
    ] = None,
    n_pulses: Annotated[
        int | None, Field(description="Pulses integrated per dwell (radar)")
    ] = None,
    swerling: Annotated[
        int | None, Field(description="Swerling fluctuation model 0-4 (radar)")
    ] = None,
    integration_type: Annotated[
        str | None, Field(description="'coherent' or 'noncoherent' (radar)")
    ] = None,
    duty_cycle: Annotated[
        float | None, Field(description="Transmit duty cycle 0-1 (radar)")
    ] = None,
    scan_angle_deg: Annotated[
        float | None, Field(description="Scan angle off boresight (deg)")
    ] = None,
    clutter_type: Annotated[
        str | None, Field(description="'none', 'sea', 'ground', or 'rain' (radar)")
    ] = None,
    sea_state: Annotated[
        int | None, Field(description="Sea state 0-6 for sea clutter (radar)")
    ] = None,
    terrain_type: Annotated[
        str | None, Field(description="Terrain type for ground clutter (radar)")
    ] = None,
    cfar_type: Annotated[
        str | None, Field(description="'none', 'CA', 'OS', 'GO', or 'SO' (radar)")
    ] = None,
    cfar_ref_cells: Annotated[
        int | None, Field(description="CFAR reference cells (radar)")
    ] = None,
    prf_hz: Annotated[
        float | None, Field(description="Pulse repetition frequency (Hz, radar)")
    ] = None,
    search_az_extent_deg: Annotated[
        float | None, Field(description="Search azimuth extent (deg, radar)")
    ] = None,
    search_el_extent_deg: Annotated[
        float | None, Field(description="Search elevation extent (deg, radar)")
    ] = None,
    search_frame_time_ms: Annotated[
        float | None, Field(description="Search frame time budget (ms, radar)")
    ] = None,
    variables: Annotated[
        list[dict[str, Any]] | None,
        Field(description="Design variables: [{name, type, low, high}]"),
    ] = None,
    requirements: Annotated[
        list[dict[str, Any]] | None,
        Field(description="Requirement dicts: [{id, name, metric_key, op, value}]"),
    ] = None,
    n_samples: Annotated[int, Field(description="Number of DOE samples")] = 50,
    method: Annotated[str, Field(description="DOE method: 'lhs', 'random', 'grid'")] = "lhs",
    seed: Annotated[int | None, Field(description="Random seed")] = None,
) -> dict[str, Any]:
    """Run a design-of-experiments trade study with Pareto analysis."""
    try:
        from apab.system.wrappers_pas import PASSystemEngine

        logger.info(
            "Running trade study: %s scenario, %d samples, method=%s",
            scenario_type, n_samples, method,
        )

        engine = PASSystemEngine()

        scenario: Any
        if scenario_type == "radar":
            radar_kwargs = {
                k: v
                for k, v in {
                    "pd_required": pd_required,
                    "pfa": pfa,
                    "n_pulses": n_pulses,
                    "swerling": swerling,
                    "integration_type": integration_type,
                    "duty_cycle": duty_cycle,
                    "scan_angle_deg": scan_angle_deg,
                    "clutter_type": clutter_type,
                    "sea_state": sea_state,
                    "terrain_type": terrain_type,
                    "cfar_type": cfar_type,
                    "cfar_ref_cells": cfar_ref_cells,
                    "prf_hz": prf_hz,
                    "search_az_extent_deg": search_az_extent_deg,
                    "search_el_extent_deg": search_el_extent_deg,
                    "search_frame_time_ms": search_frame_time_ms,
                }.items()
                if v is not None
            }
            scenario = engine.build_radar_scenario(
                freq_hz=freq_hz,
                bandwidth_hz=bandwidth_hz,
                range_m=range_m,
                target_rcs_dbsm=target_rcs_dbsm,
                **radar_kwargs,
            )
        else:
            scenario = engine.build_comms_scenario(
                freq_hz=freq_hz,
                bandwidth_hz=bandwidth_hz,
                range_m=range_m,
                required_snr_db=required_snr_db,
            )

        result = engine.run_trade_study(
            scenario=scenario,
            requirements=requirements,
            variables=variables,
            n_samples=n_samples,
            method=method,
            seed=seed,
        )

        logger.info("Trade study completed: %d feasible designs", result["n_feasible"])
        return {
            "n_feasible": result["n_feasible"],
            "n_total": len(result.get("results", {})),
            "pareto_count": len(result.get("pareto", {})),
            "status": "completed",
        }
    except Exception as e:
        logger.exception("system_trade_study failed")
        return {"error": str(e), "status": "failed"}
