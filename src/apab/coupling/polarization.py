"""Polarization conversion and axial-ratio utilities."""

from __future__ import annotations

import math


def hv_to_circular(
    e_h: complex,
    e_v: complex,
) -> tuple[complex, complex]:
    """Convert horizontal / vertical basis to LHCP / RHCP.

    The conversion follows the standard definition:

    .. math::

        E_{\\text{LHCP}} = \\frac{E_H + j \\, E_V}{\\sqrt{2}}

        E_{\\text{RHCP}} = \\frac{E_H - j \\, E_V}{\\sqrt{2}}

    Parameters
    ----------
    e_h:
        Complex amplitude of the horizontal (H) component.
    e_v:
        Complex amplitude of the vertical (V) component.

    Returns
    -------
    tuple[complex, complex]
        ``(e_lhcp, e_rhcp)`` complex amplitudes.
    """
    sqrt2 = math.sqrt(2.0)
    e_lhcp = (e_h + 1j * e_v) / sqrt2
    e_rhcp = (e_h - 1j * e_v) / sqrt2
    return e_lhcp, e_rhcp


def axial_ratio(
    e_co: complex,
    e_cx: complex,
) -> float:
    """Compute the axial ratio from co-pol and cross-pol amplitudes.

    .. math::

        \\text{AR} = \\frac{|E_{\\text{co}}| + |E_{\\text{cx}}|}
                          {|E_{\\text{co}}| - |E_{\\text{cx}}|}

    The result is >= 1 for an elliptically polarised wave, exactly 1 for
    circular polarisation, and ``inf`` for linear polarisation (when the
    cross-pol component vanishes or the magnitudes are equal in such a
    way that the denominator is zero).

    Parameters
    ----------
    e_co:
        Complex co-polarised amplitude.
    e_cx:
        Complex cross-polarised amplitude.

    Returns
    -------
    float
        Axial ratio (dimensionless, >= 1).  Returns ``math.inf`` when the
        denominator is zero.
    """
    mag_co = abs(e_co)
    mag_cx = abs(e_cx)
    denom = mag_co - mag_cx
    if denom == 0.0:
        return math.inf
    return (mag_co + mag_cx) / denom
