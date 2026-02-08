"""Tests for polarization conversion and axial-ratio utilities."""

from __future__ import annotations

import math

import pytest

from apab.coupling.polarization import axial_ratio, hv_to_circular


class TestHVToCircular:
    def test_h_only(self) -> None:
        """Pure H-pol: both LHCP and RHCP have magnitude 1/sqrt(2)."""
        e_lhcp, e_rhcp = hv_to_circular(e_h=1.0, e_v=0.0)
        sqrt2 = math.sqrt(2.0)
        assert e_lhcp == pytest.approx(1.0 / sqrt2, rel=1e-12)
        assert e_rhcp == pytest.approx(1.0 / sqrt2, rel=1e-12)

    def test_v_only(self) -> None:
        """Pure V-pol: LHCP = j/sqrt(2), RHCP = -j/sqrt(2)."""
        e_lhcp, e_rhcp = hv_to_circular(e_h=0.0, e_v=1.0)
        sqrt2 = math.sqrt(2.0)
        assert e_lhcp == pytest.approx(1j / sqrt2, rel=1e-12)
        assert e_rhcp == pytest.approx(-1j / sqrt2, rel=1e-12)

    def test_lhcp_input(self) -> None:
        """If e_h=1, e_v=-j => e_lhcp = (1 + j*(-j))/sqrt(2) = (1+1)/sqrt(2) = sqrt(2)."""
        e_lhcp, e_rhcp = hv_to_circular(e_h=1.0, e_v=-1j)
        assert abs(e_lhcp) == pytest.approx(math.sqrt(2.0), rel=1e-12)
        assert abs(e_rhcp) == pytest.approx(0.0, abs=1e-12)

    def test_rhcp_input(self) -> None:
        """If e_h=1, e_v=j => e_rhcp = (1 - j*j)/sqrt(2) = (1+1)/sqrt(2) = sqrt(2)."""
        e_lhcp, e_rhcp = hv_to_circular(e_h=1.0, e_v=1j)
        assert abs(e_lhcp) == pytest.approx(0.0, abs=1e-12)
        assert abs(e_rhcp) == pytest.approx(math.sqrt(2.0), rel=1e-12)

    def test_complex_inputs(self) -> None:
        """Arbitrary complex inputs should satisfy the transform identity."""
        e_h = 0.5 + 0.3j
        e_v = -0.2 + 0.7j
        e_lhcp, e_rhcp = hv_to_circular(e_h, e_v)

        sqrt2 = math.sqrt(2.0)
        expected_lhcp = (e_h + 1j * e_v) / sqrt2
        expected_rhcp = (e_h - 1j * e_v) / sqrt2
        assert e_lhcp == pytest.approx(expected_lhcp, rel=1e-12)
        assert e_rhcp == pytest.approx(expected_rhcp, rel=1e-12)


class TestAxialRatio:
    def test_co_only_gives_one(self) -> None:
        """Pure co-pol (no cross-pol) -> AR = (|1|+0)/(|1|-0) = 1.0."""
        ar = axial_ratio(e_co=1.0, e_cx=0.0)
        assert ar == pytest.approx(1.0)

    def test_equal_co_cx_gives_inf(self) -> None:
        """Equal magnitudes -> denominator is zero -> AR = inf."""
        ar = axial_ratio(e_co=1.0, e_cx=1.0)
        assert ar == math.inf

    def test_equal_complex_magnitudes_gives_inf(self) -> None:
        """Different phases but same magnitude -> inf."""
        ar = axial_ratio(e_co=1.0 + 0j, e_cx=0.0 + 1.0j)
        assert ar == math.inf

    def test_known_ratio(self) -> None:
        """co=2, cx=1 -> AR = 3/1 = 3.0."""
        ar = axial_ratio(e_co=2.0, e_cx=1.0)
        assert ar == pytest.approx(3.0)

    def test_small_cross_pol(self) -> None:
        """co=1, cx=0.1 -> AR = 1.1/0.9 = 1.2222..."""
        ar = axial_ratio(e_co=1.0, e_cx=0.1)
        assert ar == pytest.approx(1.1 / 0.9, rel=1e-10)

    def test_complex_amplitudes(self) -> None:
        """Magnitudes are used, not the raw complex values."""
        e_co = 3.0 + 4.0j  # |e_co| = 5
        e_cx = 0.0 + 0.0j  # |e_cx| = 0
        ar = axial_ratio(e_co, e_cx)
        assert ar == pytest.approx(1.0)
