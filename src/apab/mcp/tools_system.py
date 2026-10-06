"""MCP tools for system-level analysis (phased-array-systems)."""

from __future__ import annotations

import logging
import math
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field

from apab.mcp.server import get_mcp

logger = logging.getLogger(__name__)
mcp = get_mcp()

# ── shared parameter aliases ──────────────────────────────────────────────
# Allowed values and bounds mirror phased-array-systems' RadarDetectionScenario
# so they reach the LLM in the tool schema; tests check the two do not drift.

ScenarioType = Annotated[
    Literal["comms", "radar"], Field(description="Scenario type: 'comms' or 'radar'")
]
RequiredSnrDb = Annotated[float, Field(description="Required SNR (dB, comms)")]
TargetRcsDbsm = Annotated[float, Field(description="Target RCS (dBsm, radar)")]
ScanAngleDeg = Annotated[
    float | None, Field(ge=0, lt=90, description="Beam scan angle off boresight (deg)")
]
PdRequired = Annotated[
    float | None, Field(gt=0, lt=1, description="Required detection probability (radar)")
]
Pfa = Annotated[float | None, Field(gt=0, lt=1, description="False-alarm probability (radar)")]
NPulses = Annotated[int | None, Field(ge=1, description="Pulses integrated per dwell (radar)")]


def _int_from_digits(value: Any) -> Any:
    """Accept "1" for 1: LLMs often send integer choices as strings, and a
    Literal of ints does not coerce them (an int field would)."""
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return value


Swerling = Annotated[
    Literal[0, 1, 2, 3, 4] | None,
    BeforeValidator(_int_from_digits),
    Field(description="Swerling fluctuation model (radar)"),
]
IntegrationType = Annotated[
    Literal["coherent", "noncoherent"] | None, Field(description="Pulse integration (radar)")
]
DutyCycle = Annotated[float | None, Field(gt=0, le=1, description="Transmit duty cycle (radar)")]
ClutterType = Annotated[
    Literal["none", "sea", "ground", "rain"] | None,
    Field(description="Clutter environment (radar); 'rain' needs rain_rate_mm_hr"),
]
SeaState = Annotated[int | None, Field(ge=0, le=6, description="Douglas sea state (radar)")]
TerrainType = Annotated[
    Literal["rural", "urban", "forest", "desert", "wetland"] | None,
    Field(description="Terrain for ground clutter (radar)"),
]
RainRateMmHr = Annotated[
    float | None, Field(gt=0, description="Rain rate for rain clutter/attenuation (mm/hr, radar)")
]
AntennaHeightM = Annotated[
    float | None, Field(ge=0, description="Antenna height above surface (m, radar)")
]
TargetHeightM = Annotated[
    float | None, Field(ge=0, description="Target height above surface (m, radar)")
]
Polarization = Annotated[
    Literal["HH", "VV", "HV"] | None, Field(description="Polarization for clutter (radar)")
]
CfarType = Annotated[
    Literal["none", "CA", "OS", "GO", "SO"] | None, Field(description="CFAR detector (radar)")
]
CfarRefCells = Annotated[int | None, Field(ge=2, description="CFAR reference cells (radar)")]
PrfHz = Annotated[
    float | None, Field(gt=0, description="PRF (Hz, radar search timeline)")
]
SearchAzExtentDeg = Annotated[
    float | None, Field(gt=0, le=360, description="Search azimuth extent (deg, radar timeline)")
]
SearchElExtentDeg = Annotated[
    float | None, Field(gt=0, le=90, description="Search elevation extent (deg, radar timeline)")
]
SearchFrameTimeMs = Annotated[
    float | None, Field(gt=0, description="Search frame budget (ms, radar timeline)")
]

# Radar-only options forwarded to RadarDetectionScenario when set.
_RADAR_OPTIONS: tuple[str, ...] = (
    "pd_required",
    "pfa",
    "n_pulses",
    "swerling",
    "integration_type",
    "duty_cycle",
    "clutter_type",
    "sea_state",
    "terrain_type",
    "rain_rate_mm_hr",
    "antenna_height_m",
    "target_height_m",
    "polarization",
    "cfar_type",
    "cfar_ref_cells",
    "prf_hz",
    "search_az_extent_deg",
    "search_el_extent_deg",
    "search_frame_time_ms",
)
_TIMELINE_KEYS = ("prf_hz", "search_az_extent_deg", "search_el_extent_deg")


def _radar_options(params: dict[str, Any]) -> dict[str, Any]:
    """Pick the radar options that were set out of a tool's ``locals()``."""
    return {k: params[k] for k in _RADAR_OPTIONS if params[k] is not None}


def _build_scenario(
    engine: Any,
    scenario_type: str,
    freq_hz: float,
    bandwidth_hz: float,
    range_m: float,
    required_snr_db: float,
    target_rcs_dbsm: float,
    scan_angle_deg: float | None,
    radar_options: dict[str, Any],
) -> Any:
    """Build a comms or radar scenario, refusing options that would be ignored."""
    common: dict[str, Any] = {}
    if scan_angle_deg is not None:
        common["scan_angle_deg"] = scan_angle_deg

    if scenario_type == "comms":
        if radar_options:
            raise ValueError(
                "radar-only options set for a comms scenario: " + ", ".join(radar_options)
            )
        return engine.build_comms_scenario(
            freq_hz=freq_hz,
            bandwidth_hz=bandwidth_hz,
            range_m=range_m,
            required_snr_db=required_snr_db,
            **common,
        )
    if scenario_type != "radar":
        raise ValueError(f"scenario_type must be 'comms' or 'radar', got {scenario_type!r}")

    if radar_options.get("clutter_type") == "rain" and "rain_rate_mm_hr" not in radar_options:
        raise ValueError("clutter_type='rain' requires rain_rate_mm_hr")
    timeline_set = [k for k in _TIMELINE_KEYS if k in radar_options]
    if timeline_set and len(timeline_set) < len(_TIMELINE_KEYS):
        missing = [k for k in _TIMELINE_KEYS if k not in radar_options]
        raise ValueError("search timeline needs all of prf_hz, search_az_extent_deg, "
                         "search_el_extent_deg; missing: " + ", ".join(missing))
    if "search_frame_time_ms" in radar_options and not timeline_set:
        raise ValueError(
            "search_frame_time_ms requires prf_hz, search_az_extent_deg, search_el_extent_deg"
        )

    return engine.build_radar_scenario(
        freq_hz=freq_hz,
        bandwidth_hz=bandwidth_hz,
        range_m=range_m,
        target_rcs_dbsm=target_rcs_dbsm,
        **common,
        **radar_options,
    )


def _json_safe(metrics: dict[str, Any]) -> dict[str, Any]:
    """Replace inf/NaN with None, recording the original under ``nonfinite_metrics``.

    PAS uses infinities as sentinels (``imd3_dbc = inf`` when no nonlinearity
    is modeled, ``sll_db = -inf`` when no sidelobe exists); JSON cannot carry them.
    """
    nonfinite = {
        k: str(v) for k, v in metrics.items() if isinstance(v, float) and not math.isfinite(v)
    }
    if not nonfinite:
        return metrics
    out = {k: (None if k in nonfinite else v) for k, v in metrics.items()}
    out["nonfinite_metrics"] = nonfinite
    return out


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
    scenario_type: ScenarioType = "comms",
    required_snr_db: RequiredSnrDb = 10.0,
    target_rcs_dbsm: TargetRcsDbsm = 0.0,
    scan_angle_deg: ScanAngleDeg = None,
    pd_required: PdRequired = None,
    pfa: Pfa = None,
    n_pulses: NPulses = None,
    swerling: Swerling = None,
    integration_type: IntegrationType = None,
    duty_cycle: DutyCycle = None,
    clutter_type: ClutterType = None,
    sea_state: SeaState = None,
    terrain_type: TerrainType = None,
    rain_rate_mm_hr: RainRateMmHr = None,
    antenna_height_m: AntennaHeightM = None,
    target_height_m: TargetHeightM = None,
    polarization: Polarization = None,
    cfar_type: CfarType = None,
    cfar_ref_cells: CfarRefCells = None,
    prf_hz: PrfHz = None,
    search_az_extent_deg: SearchAzExtentDeg = None,
    search_el_extent_deg: SearchElExtentDeg = None,
    search_frame_time_ms: SearchFrameTimeMs = None,
    taper: Annotated[str, Field(description="Taper window name")] = "uniform",
    requirements: Annotated[
        list[dict[str, Any]] | None,
        Field(description="Optional list of requirement dicts"),
    ] = None,
) -> dict[str, Any]:
    """Evaluate a phased-array architecture against a comms link or radar detection scenario."""
    radar_options = _radar_options(locals())
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
        scenario = _build_scenario(
            engine, scenario_type, freq_hz, bandwidth_hz, range_m,
            required_snr_db, target_rcs_dbsm, scan_angle_deg, radar_options,
        )

        metrics = _json_safe(engine.evaluate(arch, scenario, requirements))
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
    scenario_type: ScenarioType = "comms",
    required_snr_db: RequiredSnrDb = 10.0,
    target_rcs_dbsm: TargetRcsDbsm = 0.0,
    scan_angle_deg: ScanAngleDeg = None,
    pd_required: PdRequired = None,
    pfa: Pfa = None,
    n_pulses: NPulses = None,
    swerling: Swerling = None,
    integration_type: IntegrationType = None,
    duty_cycle: DutyCycle = None,
    clutter_type: ClutterType = None,
    sea_state: SeaState = None,
    terrain_type: TerrainType = None,
    rain_rate_mm_hr: RainRateMmHr = None,
    antenna_height_m: AntennaHeightM = None,
    target_height_m: TargetHeightM = None,
    polarization: Polarization = None,
    cfar_type: CfarType = None,
    cfar_ref_cells: CfarRefCells = None,
    prf_hz: PrfHz = None,
    search_az_extent_deg: SearchAzExtentDeg = None,
    search_el_extent_deg: SearchElExtentDeg = None,
    search_frame_time_ms: SearchFrameTimeMs = None,
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
    radar_options = _radar_options(locals())
    try:
        from apab.system.wrappers_pas import PASSystemEngine

        logger.info(
            "Running trade study: %s scenario, %d samples, method=%s",
            scenario_type, n_samples, method,
        )

        engine = PASSystemEngine()
        scenario = _build_scenario(
            engine, scenario_type, freq_hz, bandwidth_hz, range_m,
            required_snr_db, target_rcs_dbsm, scan_angle_deg, radar_options,
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
        n_total = len(next(iter(result["results"].values()), {})) if result["results"] else 0
        n_pareto = len(next(iter(result["pareto"].values()), {})) if result["pareto"] else 0
        return {
            "n_feasible": result["n_feasible"],
            "n_failed": result["n_failed"],
            "first_error": result["first_error"],
            "n_total": n_total,
            "pareto_count": n_pareto,
            "pareto_objectives": result["pareto_objectives"],
            "status": "completed",
        }
    except Exception as e:
        logger.exception("system_trade_study failed")
        return {"error": str(e), "status": "failed"}
