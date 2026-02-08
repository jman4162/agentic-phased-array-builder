"""Active impedance and scan-blindness utilities.

Pure-math functions operating on numpy arrays.  No solver or schema
dependencies -- these work on raw S-parameter matrices and excitation
vectors.
"""

from __future__ import annotations

import numpy as np


def active_reflection_coefficient(
    s_matrix: np.ndarray,
    excitation: np.ndarray,
) -> np.ndarray:
    """Compute the active reflection coefficient for each port.

    For an *N*-port network the active reflection coefficient of port *n*
    is defined as::

        Gamma_active[n] = sum_m(S[n, m] * a[m]) / a[n]

    Parameters
    ----------
    s_matrix:
        Complex S-parameter matrix of shape ``(N, N)``.
    excitation:
        Complex excitation vector of shape ``(N,)`` representing the
        incident wave amplitudes at each port.

    Returns
    -------
    np.ndarray
        Complex array of shape ``(N,)`` with the active reflection
        coefficient for every port.
    """
    s_matrix = np.asarray(s_matrix, dtype=complex)
    excitation = np.asarray(excitation, dtype=complex)

    # S @ a gives the total reflected wave at each port.
    reflected = s_matrix @ excitation  # shape (N,)

    # Gamma_active[n] = reflected[n] / a[n]
    gamma = reflected / excitation

    return gamma  # type: ignore[no-any-return]


def active_impedance(
    gamma: np.ndarray,
    z0: float = 50.0,
) -> np.ndarray:
    """Convert active reflection coefficients to active impedance.

    .. math::

        Z_{\\text{active}} = z_0 \\, \\frac{1 + \\Gamma}{1 - \\Gamma}

    Parameters
    ----------
    gamma:
        Complex active reflection coefficient array (any shape).
    z0:
        Reference impedance in ohms (default 50).

    Returns
    -------
    np.ndarray
        Complex active impedance array of the same shape as *gamma*.
    """
    gamma = np.asarray(gamma, dtype=complex)
    return z0 * (1.0 + gamma) / (1.0 - gamma)


def detect_scan_blindness(
    gamma: np.ndarray,
    threshold: float = 0.95,
) -> list[dict[str, float | int]]:
    """Identify ports where the active reflection magnitude exceeds a threshold.

    Scan blindness is indicated when |Gamma_active| approaches 1, meaning
    virtually all incident power is reflected.

    Parameters
    ----------
    gamma:
        Complex active reflection coefficient array of shape ``(N,)``.
    threshold:
        Magnitude threshold above which a port is flagged (default 0.95).

    Returns
    -------
    list[dict]
        A list of dicts ``{"port": int, "gamma_mag": float}`` for every
        port whose |Gamma| exceeds *threshold*.  The list is empty when no
        blindness is detected.
    """
    gamma = np.asarray(gamma, dtype=complex)
    mags = np.abs(gamma)
    flagged: list[dict[str, float | int]] = []
    for idx in range(mags.size):
        mag = float(mags[idx])
        if mag > threshold:
            flagged.append({"port": int(idx), "gamma_mag": mag})
    return flagged
