"""Coupled-pattern computation: overlay S-matrix coupling onto array patterns."""

from __future__ import annotations

import math

import numpy as np
import phased_array as pa

from apab.core.schemas import ArraySpec, PatternResult
from apab.pattern.wrappers_pam import PAMPatternEngine


def coupled_pattern(
    engine: PAMPatternEngine,
    spec: ArraySpec,
    freq_hz: float,
    theta0: float,
    phi0: float,
    s_matrix: np.ndarray | None = None,
    excitation: np.ndarray | None = None,
) -> PatternResult:
    """Compute a far-field pattern with optional mutual-coupling correction.

    When both *s_matrix* and *excitation* are supplied the coupled excitation
    is computed as ``s_matrix @ excitation`` and used directly as the element
    weights for the full-pattern calculation.

    If *s_matrix* is ``None`` the function falls back to the standard
    :meth:`PAMPatternEngine.full_pattern` (no coupling applied).

    Parameters
    ----------
    engine:
        A :class:`PAMPatternEngine` instance.
    spec:
        Array specification.
    freq_hz:
        Operating frequency in Hz.
    theta0, phi0:
        Steering direction in degrees.
    s_matrix:
        Complex S-parameter matrix of shape ``(N, N)`` where *N* is the
        number of elements.  ``None`` disables coupling.
    excitation:
        Complex excitation vector of length *N*.  Required when *s_matrix*
        is provided.

    Returns
    -------
    PatternResult
        The resulting far-field pattern.
    """
    if s_matrix is None:
        return engine.full_pattern(spec, freq_hz, theta0, phi0)

    if excitation is None:
        raise ValueError(
            "excitation must be provided when s_matrix is not None"
        )

    coupled_excitation: np.ndarray = s_matrix @ excitation

    geom = engine.create_geometry(spec, freq_hz)
    k = pa.frequency_to_k(freq_hz)

    theta_1d, phi_1d, pattern_db = pa.compute_full_pattern(
        geom.x, geom.y, coupled_excitation, k,
    )

    theta_grid, phi_grid = np.meshgrid(theta_1d, phi_1d, indexing="ij")
    directivity_linear = pa.compute_directivity(theta_grid, phi_grid, pattern_db)
    directivity_dbi = 10.0 * math.log10(max(directivity_linear, 1e-30))

    return PatternResult(
        theta_deg=np.rad2deg(theta_1d).tolist(),
        phi_deg=np.rad2deg(phi_1d).tolist(),
        pattern_db=pattern_db.tolist(),
        directivity_dbi=directivity_dbi,
        metadata={
            "coupled": True,
            "theta0_deg": theta0,
            "phi0_deg": phi0,
            "freq_hz": freq_hz,
        },
    )
