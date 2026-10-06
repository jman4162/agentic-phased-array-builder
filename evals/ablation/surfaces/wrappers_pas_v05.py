"""Wrapper around the phased-array-systems (PAS) library for APAB system-level analysis."""

from __future__ import annotations

import logging
from typing import Any, cast

import pandas as pd
from phased_array_systems.architecture import Architecture, ArrayConfig, RFChainConfig
from phased_array_systems.evaluate import evaluate_case
from phased_array_systems.requirements import Requirement, RequirementSet
from phased_array_systems.scenarios import CommsLinkScenario, RadarDetectionScenario
from phased_array_systems.trades import (
    BatchRunner,
    DesignSpace,
    extract_pareto,
    filter_feasible,
    generate_doe,
)

from apab.core.schemas import ArraySpec

logger = logging.getLogger(__name__)

_C = 3e8  # speed of light (m/s)


class PASSystemEngine:
    """High-level system engine backed by *phased-array-systems*.

    Provides helpers for building architectures, scenarios, evaluating
    link/radar budgets, and running DOE trade studies with Pareto analysis.
    """

    def __init__(self) -> None:
        pass

    # ── architecture ──────────────────────────────────────────────────

    def build_architecture(
        self,
        array_spec: ArraySpec,
        rf_spec: dict[str, Any],
    ) -> Architecture:
        """Build a PAS :class:`Architecture` from APAB schemas.

        Parameters
        ----------
        array_spec:
            APAB :class:`ArraySpec` (size, spacing, taper, steer).
        rf_spec:
            Dictionary of RF-chain parameters.  Must contain
            ``tx_power_w_per_elem``; may optionally contain ``freq_hz``
            (used to convert physical spacing to wavelengths),
            ``pa_efficiency``, ``noise_figure_db``, ``n_tx_beams``,
            ``feed_loss_db``, ``system_loss_db``.

        Returns
        -------
        Architecture
            The PAS architecture object.
        """
        nx = array_spec.size[0]
        ny = array_spec.size[1]

        # Convert physical spacing to wavelengths if frequency is available.
        freq_hz = rf_spec.get("freq_hz")
        if freq_hz is not None:
            wavelength = _C / freq_hz
            dx_lambda = array_spec.spacing_m[0] / wavelength
            dy_lambda = array_spec.spacing_m[1] / wavelength
        else:
            dx_lambda = 0.5
            dy_lambda = 0.5

        array_cfg = ArrayConfig(
            nx=nx,
            ny=ny,
            dx_lambda=dx_lambda,
            dy_lambda=dy_lambda,
            enforce_subarray_constraint=False,
        )

        # Build RFChainConfig from rf_spec, excluding keys not in the model.
        rf_keys = {
            "tx_power_w_per_elem",
            "pa_efficiency",
            "noise_figure_db",
            "n_tx_beams",
            "feed_loss_db",
            "system_loss_db",
        }
        rf_kwargs = {k: v for k, v in rf_spec.items() if k in rf_keys}
        rf_cfg = RFChainConfig(**rf_kwargs)

        return Architecture(array=array_cfg, rf=rf_cfg)

    # ── scenarios ─────────────────────────────────────────────────────

    def build_comms_scenario(
        self,
        freq_hz: float,
        bandwidth_hz: float,
        range_m: float,
        required_snr_db: float,
        **kwargs: Any,
    ) -> CommsLinkScenario:
        """Create a communications link scenario.

        All positional arguments are required by the PAS
        :class:`CommsLinkScenario`; additional keyword arguments are
        forwarded directly to the constructor.
        """
        return CommsLinkScenario(
            freq_hz=freq_hz,
            bandwidth_hz=bandwidth_hz,
            range_m=range_m,
            required_snr_db=required_snr_db,
            **kwargs,
        )

    def build_radar_scenario(
        self,
        freq_hz: float,
        bandwidth_hz: float,
        range_m: float,
        target_rcs_dbsm: float,
        **kwargs: Any,
    ) -> RadarDetectionScenario:
        """Create a radar detection scenario.

        All positional arguments are required by the PAS
        :class:`RadarDetectionScenario`; additional keyword arguments are
        forwarded directly to the constructor.
        """
        return RadarDetectionScenario(
            freq_hz=freq_hz,
            bandwidth_hz=bandwidth_hz,
            range_m=range_m,
            target_rcs_dbsm=target_rcs_dbsm,
            **kwargs,
        )

    # ── evaluation ────────────────────────────────────────────────────

    def evaluate(
        self,
        arch: Architecture,
        scenario: CommsLinkScenario | RadarDetectionScenario,
        requirements: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Evaluate an architecture/scenario case and return metrics.

        Parameters
        ----------
        arch:
            PAS :class:`Architecture`.
        scenario:
            A comms or radar scenario.
        requirements:
            Optional list of requirement dicts, each with keys
            ``id``, ``name``, ``metric_key``, ``op``, ``value``, and
            optionally ``severity``.

        Returns
        -------
        dict
            Flat metrics dictionary produced by ``evaluate_case``.
        """
        req_set = _build_requirement_set(requirements)
        metrics = evaluate_case(arch, scenario, req_set)
        return dict(metrics)

    # ── trade studies ─────────────────────────────────────────────────

    def run_trade_study(
        self,
        scenario: CommsLinkScenario | RadarDetectionScenario,
        requirements: list[dict[str, Any]] | None = None,
        variables: list[dict[str, Any]] | None = None,
        n_samples: int = 50,
        method: str = "lhs",
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Run a DOE trade study with Pareto extraction.

        Parameters
        ----------
        scenario:
            A comms or radar scenario.
        requirements:
            Optional list of requirement dicts (same format as
            :meth:`evaluate`).
        variables:
            List of design-variable dicts, each with ``name``, ``type``,
            ``low``, ``high``, and/or ``values``.
        n_samples:
            Number of DOE samples (ignored for grid method).
        method:
            Sampling method (``"lhs"``, ``"random"``, or ``"grid"``).
        seed:
            Random seed for reproducibility.

        Returns
        -------
        dict
            ``"results"`` — full results DataFrame as dict,
            ``"pareto"`` — Pareto-optimal subset as dict,
            ``"n_feasible"`` — number of feasible designs.
        """
        # Build design space.
        design_space = DesignSpace()
        for var in variables or []:
            design_space.add_variable(
                name=var["name"],
                type=var.get("type", "float"),
                low=var.get("low"),
                high=var.get("high"),
                values=var.get("values"),
            )

        # Generate DOE.
        cases_df = generate_doe(
            design_space,
            method=cast(Any, method),
            n_samples=n_samples,
            seed=seed,
        )

        # Build requirement set.
        req_set = _build_requirement_set(requirements)

        # Run batch evaluation (single-threaded for safety).
        runner = BatchRunner(scenario, requirements=req_set)
        results_df = runner.run(cases_df, n_workers=1)

        # Cases that raised carry meta.error and no metrics; with no
        # requirements, filter_feasible would count them as feasible.
        if "meta.error" in results_df.columns:
            failed = results_df["meta.error"].notna()
        else:
            failed = pd.Series(False, index=results_df.index)
        n_failed = int(failed.sum())
        if n_failed:
            # Keep field names (pydantic puts them on line 2); drop the docs-URL line.
            first_error = " | ".join(
                line.strip()
                for line in str(results_df.loc[failed, "meta.error"].iloc[0]).splitlines()
                if line.strip() and "errors.pydantic.dev" not in line
            )
            logger.warning("%d of %d cases failed: %s", n_failed, len(results_df), first_error)
        else:
            first_error = None

        # Filter feasible designs.
        feasible_df = filter_feasible(results_df[~failed], requirements=req_set)

        # Extract the Pareto front: minimize cost against the scenario's
        # headline performance metric.  Without both columns, every feasible
        # design is returned and ``pareto_objectives`` is empty.
        pareto_df = feasible_df
        objectives = _pareto_objectives(feasible_df.columns)
        if len(feasible_df) > 0 and objectives:
            pareto_df = extract_pareto(feasible_df, cast(Any, objectives))
        elif len(feasible_df) > 0:
            logger.info("No Pareto objectives available; returning all feasible designs")

        return {
            "results": results_df.to_dict(),
            "pareto": pareto_df.to_dict(),
            "n_feasible": len(feasible_df),
            "n_failed": n_failed,
            "first_error": first_error,
            "pareto_objectives": [name for name, _ in objectives],
        }


# ── helpers ────────────────────────────────────────────────────────────

# Performance objective per scenario: comms results carry eirp_dbw, radar
# results carry snr_margin_db (and no EIRP).
_PERFORMANCE_OBJECTIVES = ("eirp_dbw", "snr_margin_db")


def _pareto_objectives(columns: Any) -> list[tuple[str, str]]:
    """Return ``[(cost, "minimize"), (performance, "maximize")]`` or ``[]``."""
    cols = set(columns)
    perf = next((c for c in _PERFORMANCE_OBJECTIVES if c in cols), None)
    if "cost_usd" not in cols or perf is None:
        return []
    return [("cost_usd", "minimize"), (perf, "maximize")]


def _build_requirement_set(
    requirements: list[dict[str, Any]] | None,
) -> RequirementSet | None:
    """Convert a list of plain dicts into a PAS :class:`RequirementSet`."""
    if not requirements:
        return None

    req_set = RequirementSet()
    for req_dict in requirements:
        req_set.add(
            Requirement(
                id=req_dict["id"],
                name=req_dict["name"],
                metric_key=req_dict["metric_key"],
                op=req_dict["op"],
                value=req_dict["value"],
                severity=req_dict.get("severity", "must"),
            )
        )
    return req_set
