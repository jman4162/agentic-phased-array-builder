"""Tests for active impedance and scan-blindness utilities."""

from __future__ import annotations

import numpy as np
import pytest

from apab.coupling.active_impedance import (
    active_impedance,
    active_reflection_coefficient,
    detect_scan_blindness,
)


class TestActiveReflectionCoefficient:
    def test_identity_s_matrix(self) -> None:
        """With S = I and uniform excitation, Gamma_active = 1 for all ports."""
        n = 4
        s = np.eye(n, dtype=complex)
        a = np.ones(n, dtype=complex)
        gamma = active_reflection_coefficient(s, a)

        # S @ a = a, gamma = a / a = 1 for each port.
        np.testing.assert_allclose(gamma, np.ones(n))

    def test_known_2x2_matrix_uniform(self) -> None:
        """Known 2x2 S-matrix with uniform excitation [1, 1].

        S = [[0.1+0.1j, 0.9],
             [0.9,       0.1+0.1j]]
        a = [1, 1]

        gamma_0 = (S00*1 + S01*1) / 1 = (0.1+0.1j + 0.9) = 1.0+0.1j
        gamma_1 = (S10*1 + S11*1) / 1 = (0.9 + 0.1+0.1j) = 1.0+0.1j
        """
        s = np.array([[0.1 + 0.1j, 0.9 + 0j], [0.9 + 0j, 0.1 + 0.1j]], dtype=complex)
        a = np.array([1.0, 1.0], dtype=complex)
        gamma = active_reflection_coefficient(s, a)

        expected = np.array([1.0 + 0.1j, 1.0 + 0.1j], dtype=complex)
        np.testing.assert_allclose(gamma, expected)

    def test_non_uniform_excitation(self) -> None:
        """With non-uniform excitation the per-port scaling matters.

        S = [[0, 0.5],
             [0.5, 0]]
        a = [2, 1]

        gamma_0 = (0*2 + 0.5*1) / 2 = 0.25
        gamma_1 = (0.5*2 + 0*1) / 1 = 1.0
        """
        s = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
        a = np.array([2.0, 1.0], dtype=complex)
        gamma = active_reflection_coefficient(s, a)

        np.testing.assert_allclose(gamma, [0.25, 1.0])

    def test_single_port(self) -> None:
        """Single-port case: gamma = S11."""
        s = np.array([[0.3 + 0.1j]], dtype=complex)
        a = np.array([1.0], dtype=complex)
        gamma = active_reflection_coefficient(s, a)
        np.testing.assert_allclose(gamma, [0.3 + 0.1j])


class TestActiveImpedance:
    def test_gamma_zero_gives_z0(self) -> None:
        """When Gamma = 0 the active impedance equals z0."""
        gamma = np.array([0.0 + 0j], dtype=complex)
        z = active_impedance(gamma, z0=50.0)
        np.testing.assert_allclose(z, [50.0 + 0j])

    def test_gamma_zero_custom_z0(self) -> None:
        gamma = np.array([0.0], dtype=complex)
        z = active_impedance(gamma, z0=75.0)
        np.testing.assert_allclose(z, [75.0 + 0j])

    def test_known_value(self) -> None:
        """Gamma = 0.2 -> Z = 50 * 1.2 / 0.8 = 75."""
        gamma = np.array([0.2], dtype=complex)
        z = active_impedance(gamma, z0=50.0)
        np.testing.assert_allclose(z, [75.0 + 0j])

    def test_vector_input(self) -> None:
        """Multiple ports at once."""
        gamma = np.array([0.0, 0.2, -0.2], dtype=complex)
        z = active_impedance(gamma, z0=50.0)
        expected = np.array([50.0, 75.0, 50.0 * 0.8 / 1.2], dtype=complex)
        np.testing.assert_allclose(z, expected)


class TestDetectScanBlindness:
    def test_no_blindness(self) -> None:
        gamma = np.array([0.1, 0.5, 0.8], dtype=complex)
        result = detect_scan_blindness(gamma, threshold=0.95)
        assert result == []

    def test_blindness_detected(self) -> None:
        gamma = np.array([0.1, 0.96, 0.99], dtype=complex)
        result = detect_scan_blindness(gamma, threshold=0.95)
        assert len(result) == 2
        ports = {d["port"] for d in result}
        assert ports == {1, 2}

    def test_exact_threshold_not_flagged(self) -> None:
        """The threshold condition is strictly greater-than."""
        gamma = np.array([0.95 + 0j], dtype=complex)
        result = detect_scan_blindness(gamma, threshold=0.95)
        assert result == []

    def test_complex_gamma_magnitude(self) -> None:
        """|0.7+0.7j| = 0.9899... which is > 0.95."""
        gamma = np.array([0.7 + 0.7j], dtype=complex)
        result = detect_scan_blindness(gamma, threshold=0.95)
        assert len(result) == 1
        assert result[0]["port"] == 0
        assert result[0]["gamma_mag"] == pytest.approx(abs(0.7 + 0.7j), rel=1e-6)
