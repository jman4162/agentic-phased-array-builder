"""Wrapper around the phased-array-modeling (PAM) library for APAB pattern computations."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import phased_array as pa

from apab.core.schemas import ArraySpec, PatternResult

logger = logging.getLogger(__name__)

_C = 3e8  # speed of light (m/s)

# Polyfill: phased-array-modeling's compute_directivity uses np.trapz which was
# removed in NumPy 2.0.  Provide a local implementation that works with both.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]


class PAMPatternEngine:
    """High-level pattern engine backed by *phased-array-modeling*."""

    def __init__(self) -> None:
        pass

    # ── geometry ────────────────────────────────────────────────────────

    def create_geometry(self, spec: ArraySpec, freq_hz: float) -> Any:
        """Build a rectangular ``ArrayGeometry`` from an :class:`ArraySpec`.

        Parameters
        ----------
        spec:
            Array specification (size, spacing, etc.).
        freq_hz:
            Operating frequency in Hz.  Used to convert physical spacing to
            wavelengths for the PAM API.

        Returns
        -------
        ArrayGeometry
            The PAM geometry dataclass.
        """
        wavelength = _C / freq_hz
        dx_wavelengths = spec.spacing_m[0] / wavelength
        dy_wavelengths = spec.spacing_m[1] / wavelength
        return pa.create_rectangular_array(
            spec.size[0],
            spec.size[1],
            dx_wavelengths,
            dy_wavelengths,
            wavelength=wavelength,
        )

    # ── weights / tapers ───────────────────────────────────────────────

    def compute_steering_weights(
        self,
        geom: Any,
        freq_hz: float,
        theta0: float,
        phi0: float,
    ) -> np.ndarray:
        """Return complex steering weights for the given scan angle.

        Parameters
        ----------
        geom:
            PAM ``ArrayGeometry`` object.
        freq_hz:
            Operating frequency in Hz.
        theta0, phi0:
            Steering direction in degrees.
        """
        k = pa.frequency_to_k(freq_hz)
        return pa.steering_vector(k, geom.x, geom.y, theta0, phi0, z=geom.z)  # type: ignore[no-any-return]

    def apply_taper(self, spec: ArraySpec) -> np.ndarray:
        """Return a real-valued amplitude taper for the given :class:`ArraySpec`.

        Supported taper names: ``uniform``, ``taylor``, ``chebyshev``,
        ``hamming``, ``hanning``, ``cosine``, ``gaussian``.
        """
        nx, ny = spec.size[0], spec.size[1]
        taper_name = spec.taper.lower()

        taper_map: dict[str, Any] = {
            "taylor": pa.taylor_taper_2d,
            "chebyshev": pa.chebyshev_taper_2d,
            "hamming": pa.hamming_taper_2d,
            "hanning": pa.hanning_taper_2d,
            "cosine": pa.cosine_taper_2d,
            "gaussian": pa.gaussian_taper_2d,
        }

        if taper_name == "uniform":
            return np.ones(nx * ny)

        func = taper_map.get(taper_name)
        if func is None:
            raise ValueError(
                f"Unknown taper '{spec.taper}'. "
                f"Choose from: uniform, {', '.join(sorted(taper_map))}."
            )
        return func(nx, ny)  # type: ignore[no-any-return]

    # ── full pattern ───────────────────────────────────────────────────

    def full_pattern(
        self,
        spec: ArraySpec,
        freq_hz: float,
        theta0: float,
        phi0: float,
    ) -> PatternResult:
        """Compute a full 2-D pattern and return a :class:`PatternResult`.

        The element weights are the product of steering weights and the
        amplitude taper specified in *spec*.
        """
        logger.debug(
            "full_pattern: %dx%d, freq=%.2e Hz, taper=%s",
            spec.size[0], spec.size[1], freq_hz, spec.taper,
        )
        geom = self.create_geometry(spec, freq_hz)
        steering = self.compute_steering_weights(geom, freq_hz, theta0, phi0)
        taper = self.apply_taper(spec)
        weights = steering * taper

        k = pa.frequency_to_k(freq_hz)

        theta_1d, phi_1d, pattern_db = pa.compute_full_pattern(
            geom.x, geom.y, weights, k,
        )

        # Directivity (linear -> dBi)
        # compute_directivity expects 2D meshgrids
        theta_grid, phi_grid = np.meshgrid(theta_1d, phi_1d, indexing="ij")
        directivity_linear = pa.compute_directivity(theta_grid, phi_grid, pattern_db)
        directivity_dbi = 10.0 * math.log10(max(directivity_linear, 1e-30))

        # Pattern cuts for HPBW / sidelobe estimation
        theta_cut, e_plane_db, h_plane_db = pa.compute_pattern_cuts(
            geom.x, geom.y, weights, k, theta0_deg=theta0, phi0_deg=phi0,
        )

        e_hpbw = pa.compute_half_power_beamwidth(theta_cut, e_plane_db)
        h_hpbw = pa.compute_half_power_beamwidth(theta_cut, h_plane_db)

        # Sidelobe level: take the lower (more negative) of E/H-plane peaks
        # outside the main beam
        sll_e = _sidelobe_level(e_plane_db)
        sll_h = _sidelobe_level(h_plane_db)
        sll = min(sll_e, sll_h) if sll_e is not None and sll_h is not None else None

        return PatternResult(
            theta_deg=np.rad2deg(theta_1d).tolist(),
            phi_deg=np.rad2deg(phi_1d).tolist(),
            pattern_db=pattern_db.tolist(),
            directivity_dbi=directivity_dbi,
            sidelobe_level_db=sll,
            metadata={
                "e_plane_hpbw_deg": e_hpbw,
                "h_plane_hpbw_deg": h_hpbw,
                "theta0_deg": theta0,
                "phi0_deg": phi0,
                "freq_hz": freq_hz,
            },
        )

    # ── multi-beam ─────────────────────────────────────────────────────

    def multi_beam(
        self,
        spec: ArraySpec,
        freq_hz: float,
        beam_directions: list[tuple[float, float]],
    ) -> PatternResult:
        """Compute a multi-beam pattern via weight superposition."""
        geom = self.create_geometry(spec, freq_hz)
        k = pa.frequency_to_k(freq_hz)
        weights = pa.multi_beam_weights_superposition(geom, k, beam_directions)

        theta_1d, phi_1d, pattern_db = pa.compute_full_pattern(
            geom.x, geom.y, weights, k,
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
                "beam_directions": beam_directions,
                "freq_hz": freq_hz,
            },
        )

    # ── null steering ──────────────────────────────────────────────────

    def null_steering(
        self,
        spec: ArraySpec,
        freq_hz: float,
        theta0: float,
        phi0: float,
        null_directions: list[tuple[float, float]],
    ) -> PatternResult:
        """Compute a pattern with nulls steered to specific directions."""
        geom = self.create_geometry(spec, freq_hz)
        k = pa.frequency_to_k(freq_hz)
        weights = pa.null_steering_projection(
            geom, k, theta0, phi0, null_directions,
        )

        theta_1d, phi_1d, pattern_db = pa.compute_full_pattern(
            geom.x, geom.y, weights, k,
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
                "theta0_deg": theta0,
                "phi0_deg": phi0,
                "null_directions": null_directions,
                "freq_hz": freq_hz,
            },
        )

    # ── impairments ────────────────────────────────────────────────────

    def with_impairments(
        self,
        weights: np.ndarray,
        phase_bits: int | None = None,
    ) -> np.ndarray:
        """Apply hardware impairments (phase quantisation) to *weights*.

        Parameters
        ----------
        weights:
            Complex element weights.
        phase_bits:
            If not ``None``, quantise the phase of each weight to
            ``2**phase_bits`` levels.
        """
        if phase_bits is not None:
            weights = pa.quantize_phase(weights, phase_bits)
        return weights


# ── helpers ────────────────────────────────────────────────────────────


def _sidelobe_level(cut_db: np.ndarray) -> float | None:
    """Estimate peak sidelobe level from a 1-D pattern cut.

    Returns the level (in dB, relative to the peak) of the highest sidelobe,
    or ``None`` if no sidelobes are detected.
    """
    peak_idx = int(np.argmax(cut_db))
    peak_val = cut_db[peak_idx]

    # Walk outward from the peak until the pattern drops below -3 dB to
    # define the "main beam" region, then find the max outside that region.
    threshold = peak_val - 3.0
    left = peak_idx
    while left > 0 and cut_db[left] >= threshold:
        left -= 1
    right = peak_idx
    while right < len(cut_db) - 1 and cut_db[right] >= threshold:
        right += 1

    outside = np.concatenate([cut_db[:left], cut_db[right + 1 :]])
    if len(outside) == 0:
        return None
    sll = float(np.max(outside) - peak_val)
    return sll
