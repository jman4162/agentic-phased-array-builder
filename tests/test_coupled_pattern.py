"""Tests for coupled_pattern (S-matrix coupling overlay)."""

from __future__ import annotations

import numpy as np
import pytest

from apab.core.schemas import ArraySpec, PatternResult, ScanPoint
from apab.pattern.coupled_pattern import coupled_pattern
from apab.pattern.wrappers_pam import PAMPatternEngine

FREQ_HZ = 10e9
NX, NY = 4, 4
N_ELEMENTS = NX * NY


@pytest.fixture
def engine() -> PAMPatternEngine:
    return PAMPatternEngine()


@pytest.fixture
def spec() -> ArraySpec:
    wavelength = 3e8 / FREQ_HZ
    half_lambda = wavelength / 2.0
    return ArraySpec(
        size=[NX, NY],
        spacing_m=[half_lambda, half_lambda],
        taper="uniform",
        steer=ScanPoint(theta_deg=0, phi_deg=0),
    )


class TestCoupledPatternFallback:
    """When s_matrix is None, coupled_pattern should fall back to full_pattern."""

    def test_returns_pattern_result(
        self, engine: PAMPatternEngine, spec: ArraySpec,
    ) -> None:
        result = coupled_pattern(engine, spec, FREQ_HZ, theta0=0, phi0=0)
        assert isinstance(result, PatternResult)

    def test_matches_uncoupled(
        self, engine: PAMPatternEngine, spec: ArraySpec,
    ) -> None:
        uncoupled = engine.full_pattern(spec, FREQ_HZ, theta0=0, phi0=0)
        fallback = coupled_pattern(
            engine, spec, FREQ_HZ, theta0=0, phi0=0, s_matrix=None,
        )
        # Should produce identical patterns since no coupling is applied
        np.testing.assert_allclose(
            np.asarray(fallback.pattern_db),
            np.asarray(uncoupled.pattern_db),
        )
        assert fallback.directivity_dbi == uncoupled.directivity_dbi


class TestCoupledPatternIdentity:
    """With an identity S-matrix the coupled result should match uncoupled."""

    def test_identity_coupling(
        self, engine: PAMPatternEngine, spec: ArraySpec,
    ) -> None:
        # Build excitation identical to what full_pattern would use
        geom = engine.create_geometry(spec, FREQ_HZ)
        steering = engine.compute_steering_weights(geom, FREQ_HZ, 0, 0)
        taper = engine.apply_taper(spec)
        excitation = steering * taper

        identity = np.eye(N_ELEMENTS, dtype=complex)

        result_coupled = coupled_pattern(
            engine, spec, FREQ_HZ, theta0=0, phi0=0,
            s_matrix=identity, excitation=excitation,
        )
        result_uncoupled = engine.full_pattern(spec, FREQ_HZ, theta0=0, phi0=0)

        np.testing.assert_allclose(
            np.asarray(result_coupled.pattern_db),
            np.asarray(result_uncoupled.pattern_db),
            atol=1e-6,
        )

    def test_directivity_similar_with_identity(
        self, engine: PAMPatternEngine, spec: ArraySpec,
    ) -> None:
        geom = engine.create_geometry(spec, FREQ_HZ)
        steering = engine.compute_steering_weights(geom, FREQ_HZ, 0, 0)
        taper = engine.apply_taper(spec)
        excitation = steering * taper

        identity = np.eye(N_ELEMENTS, dtype=complex)

        result = coupled_pattern(
            engine, spec, FREQ_HZ, theta0=0, phi0=0,
            s_matrix=identity, excitation=excitation,
        )
        uncoupled = engine.full_pattern(spec, FREQ_HZ, theta0=0, phi0=0)

        assert result.directivity_dbi is not None
        assert uncoupled.directivity_dbi is not None
        assert abs(result.directivity_dbi - uncoupled.directivity_dbi) < 0.01


class TestCoupledPatternValidation:
    """Edge-case and validation tests."""

    def test_s_matrix_without_excitation_raises(
        self, engine: PAMPatternEngine, spec: ArraySpec,
    ) -> None:
        s = np.eye(N_ELEMENTS, dtype=complex)
        with pytest.raises(ValueError, match="excitation must be provided"):
            coupled_pattern(
                engine, spec, FREQ_HZ, theta0=0, phi0=0,
                s_matrix=s, excitation=None,
            )
