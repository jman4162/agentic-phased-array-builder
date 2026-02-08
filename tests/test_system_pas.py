"""Tests for APAB PAS system engine wrapper."""

from __future__ import annotations

import pytest

from apab.core.schemas import ArraySpec, ScanPoint
from apab.system.wrappers_pas import PASSystemEngine


@pytest.fixture
def engine() -> PASSystemEngine:
    """Provide a PASSystemEngine instance."""
    return PASSystemEngine()


@pytest.fixture
def small_array_spec() -> ArraySpec:
    """Return a small 8x8 array spec for fast tests."""
    return ArraySpec(
        size=[8, 8],
        spacing_m=[0.015, 0.015],
        taper="uniform",
        steer=ScanPoint(theta_deg=0, phi_deg=0),
    )


@pytest.fixture
def rf_spec() -> dict:
    """Return a basic RF spec dict."""
    return {
        "tx_power_w_per_elem": 1.0,
        "freq_hz": 10e9,
    }


class TestBuildArchitecture:
    def test_creates_valid_architecture(
        self, engine: PASSystemEngine, small_array_spec: ArraySpec, rf_spec: dict,
    ) -> None:
        arch = engine.build_architecture(small_array_spec, rf_spec)
        assert arch.n_elements == 64
        assert arch.array.nx == 8
        assert arch.array.ny == 8

    def test_spacing_conversion_with_freq(
        self, engine: PASSystemEngine, rf_spec: dict,
    ) -> None:
        spec = ArraySpec(
            size=[4, 4],
            spacing_m=[0.015, 0.015],
            taper="uniform",
            steer=ScanPoint(theta_deg=0, phi_deg=0),
        )
        arch = engine.build_architecture(spec, rf_spec)
        # At 10 GHz, wavelength = 0.03 m, so 0.015 m = 0.5 lambda.
        assert abs(arch.array.dx_lambda - 0.5) < 1e-9
        assert abs(arch.array.dy_lambda - 0.5) < 1e-9

    def test_defaults_without_freq(self, engine: PASSystemEngine) -> None:
        spec = ArraySpec(
            size=[4, 4],
            spacing_m=[0.015, 0.015],
            taper="uniform",
            steer=ScanPoint(theta_deg=0, phi_deg=0),
        )
        rf = {"tx_power_w_per_elem": 1.0}
        arch = engine.build_architecture(spec, rf)
        # Without freq_hz the wrapper should fall back to 0.5 lambda.
        assert arch.array.dx_lambda == 0.5
        assert arch.array.dy_lambda == 0.5


class TestBuildCommsScenario:
    def test_creates_valid_scenario(self, engine: PASSystemEngine) -> None:
        scenario = engine.build_comms_scenario(
            freq_hz=10e9,
            bandwidth_hz=100e6,
            range_m=1000,
            required_snr_db=10,
        )
        assert scenario.freq_hz == 10e9
        assert scenario.bandwidth_hz == 100e6
        assert scenario.range_m == 1000
        assert scenario.required_snr_db == 10


class TestEvaluate:
    def test_returns_dict_with_expected_keys(
        self, engine: PASSystemEngine, small_array_spec: ArraySpec, rf_spec: dict,
    ) -> None:
        arch = engine.build_architecture(small_array_spec, rf_spec)
        scenario = engine.build_comms_scenario(
            freq_hz=10e9,
            bandwidth_hz=100e6,
            range_m=1000,
            required_snr_db=10,
        )
        metrics = engine.evaluate(arch, scenario)

        assert isinstance(metrics, dict)
        # The evaluator should produce at least these antenna/link keys.
        assert "g_peak_db" in metrics or "g_peak_dbi" in metrics or any(
            "g_peak" in k for k in metrics
        ), f"No gain key found in metrics: {sorted(metrics.keys())}"
        assert any("eirp" in k for k in metrics), (
            f"No EIRP key found in metrics: {sorted(metrics.keys())}"
        )

    def test_evaluate_with_requirements(
        self, engine: PASSystemEngine, small_array_spec: ArraySpec, rf_spec: dict,
    ) -> None:
        arch = engine.build_architecture(small_array_spec, rf_spec)
        scenario = engine.build_comms_scenario(
            freq_hz=10e9,
            bandwidth_hz=100e6,
            range_m=1000,
            required_snr_db=10,
        )
        requirements = [
            {
                "id": "REQ-001",
                "name": "Minimum EIRP",
                "metric_key": "eirp_dbw",
                "op": ">=",
                "value": -100.0,
                "severity": "must",
            },
        ]
        metrics = engine.evaluate(arch, scenario, requirements=requirements)
        assert isinstance(metrics, dict)
        # Verification keys should be present when requirements are supplied.
        assert "verification.passes" in metrics


class TestRunTradeStudy:
    def test_small_doe_completes(self, engine: PASSystemEngine) -> None:
        scenario = engine.build_comms_scenario(
            freq_hz=10e9,
            bandwidth_hz=100e6,
            range_m=1000,
            required_snr_db=10,
        )
        variables = [
            {"name": "array.nx", "type": "int", "low": 4, "high": 8},
            {"name": "array.ny", "type": "int", "low": 4, "high": 8},
            {"name": "rf.tx_power_w_per_elem", "type": "float", "low": 0.5, "high": 2.0},
        ]
        result = engine.run_trade_study(
            scenario=scenario,
            variables=variables,
            n_samples=5,
            method="lhs",
            seed=42,
        )

        assert isinstance(result, dict)
        assert "results" in result
        assert "pareto" in result
        assert "n_feasible" in result
        assert isinstance(result["results"], dict)
        assert isinstance(result["pareto"], dict)
        assert isinstance(result["n_feasible"], int)
