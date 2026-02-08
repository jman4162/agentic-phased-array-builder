"""Tests for PAMPatternEngine (phased-array-modeling wrapper)."""

from __future__ import annotations

import numpy as np
import pytest

from apab.core.schemas import ArraySpec, PatternResult, ScanPoint
from apab.pattern.wrappers_pam import PAMPatternEngine

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FREQ_HZ = 10e9  # 10 GHz
NX, NY = 4, 4


@pytest.fixture
def engine() -> PAMPatternEngine:
    return PAMPatternEngine()


@pytest.fixture
def spec() -> ArraySpec:
    """Small 4x4 array with half-wavelength spacing at 10 GHz."""
    wavelength = 3e8 / FREQ_HZ
    half_lambda = wavelength / 2.0
    return ArraySpec(
        size=[NX, NY],
        spacing_m=[half_lambda, half_lambda],
        taper="uniform",
        steer=ScanPoint(theta_deg=0, phi_deg=0),
    )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class TestCreateGeometry:
    def test_n_elements(self, engine: PAMPatternEngine, spec: ArraySpec) -> None:
        geom = engine.create_geometry(spec, FREQ_HZ)
        assert geom.n_elements == NX * NY

    def test_positions_are_finite(self, engine: PAMPatternEngine, spec: ArraySpec) -> None:
        geom = engine.create_geometry(spec, FREQ_HZ)
        assert np.all(np.isfinite(geom.x))
        assert np.all(np.isfinite(geom.y))


# ---------------------------------------------------------------------------
# Taper
# ---------------------------------------------------------------------------


class TestApplyTaper:
    @pytest.mark.parametrize(
        "taper_name",
        ["uniform", "taylor", "chebyshev", "hamming", "hanning", "cosine", "gaussian"],
    )
    def test_taper_length(self, engine: PAMPatternEngine, taper_name: str) -> None:
        taper_spec = ArraySpec(
            size=[NX, NY],
            spacing_m=[0.015, 0.015],
            taper=taper_name,
        )
        taper = engine.apply_taper(taper_spec)
        assert len(taper) == NX * NY

    def test_uniform_taper_all_ones(self, engine: PAMPatternEngine) -> None:
        taper_spec = ArraySpec(
            size=[NX, NY],
            spacing_m=[0.015, 0.015],
            taper="uniform",
        )
        taper = engine.apply_taper(taper_spec)
        np.testing.assert_allclose(taper, 1.0)

    def test_unknown_taper_raises(self, engine: PAMPatternEngine) -> None:
        bad_spec = ArraySpec(
            size=[NX, NY],
            spacing_m=[0.015, 0.015],
            taper="nonexistent_taper",
        )
        with pytest.raises(ValueError, match="Unknown taper"):
            engine.apply_taper(bad_spec)


# ---------------------------------------------------------------------------
# Full pattern
# ---------------------------------------------------------------------------


class TestFullPattern:
    def test_returns_pattern_result(
        self, engine: PAMPatternEngine, spec: ArraySpec,
    ) -> None:
        result = engine.full_pattern(spec, FREQ_HZ, theta0=0, phi0=0)
        assert isinstance(result, PatternResult)

    def test_theta_phi_non_empty(
        self, engine: PAMPatternEngine, spec: ArraySpec,
    ) -> None:
        result = engine.full_pattern(spec, FREQ_HZ, theta0=0, phi0=0)
        assert len(result.theta_deg) > 0
        assert len(result.phi_deg) > 0

    def test_pattern_db_non_empty(
        self, engine: PAMPatternEngine, spec: ArraySpec,
    ) -> None:
        result = engine.full_pattern(spec, FREQ_HZ, theta0=0, phi0=0)
        assert result.pattern_db is not None
        pattern = np.asarray(result.pattern_db)
        assert pattern.size > 0

    def test_directivity_positive(
        self, engine: PAMPatternEngine, spec: ArraySpec,
    ) -> None:
        result = engine.full_pattern(spec, FREQ_HZ, theta0=0, phi0=0)
        assert result.directivity_dbi is not None
        assert result.directivity_dbi > 0

    def test_metadata_contains_freq(
        self, engine: PAMPatternEngine, spec: ArraySpec,
    ) -> None:
        result = engine.full_pattern(spec, FREQ_HZ, theta0=0, phi0=0)
        assert result.metadata["freq_hz"] == FREQ_HZ
