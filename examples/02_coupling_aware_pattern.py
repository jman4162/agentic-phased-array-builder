#!/usr/bin/env python3
"""Example 02: Coupling-aware array pattern.

Demonstrates how mutual coupling (S-matrix) modifies the radiation pattern
compared to the isolated-element assumption.
"""

import numpy as np

from apab.core.schemas import ArraySpec, ScanPoint
from apab.coupling.active_impedance import (
    active_impedance,
    active_reflection_coefficient,
    detect_scan_blindness,
)
from apab.pattern.coupled_pattern import coupled_pattern
from apab.pattern.wrappers_pam import PAMPatternEngine

# ── Setup ─────────────────────────────────────────────────────────────
freq_hz = 10e9
n_elements = 16  # 4×4 array

spec = ArraySpec(
    size=[4, 4],
    spacing_m=[0.015, 0.015],  # λ/2 at 10 GHz
    taper="uniform",
    steer=ScanPoint(theta_deg=0, phi_deg=0),
)

engine = PAMPatternEngine()

# ── Create a synthetic S-matrix (coupling model) ─────────────────────
# Diagonal = S11, off-diagonal = mutual coupling
np.random.seed(42)
s11 = 0.1 * np.exp(1j * np.random.uniform(-np.pi, np.pi, n_elements))
S = np.diag(s11)
# Add nearest-neighbor coupling at -20 dB
coupling_level = 0.1  # -20 dB
for i in range(n_elements):
    for j in range(n_elements):
        if i != j and abs(i - j) <= 1:
            S[i, j] = coupling_level * np.exp(1j * np.random.uniform(-np.pi, np.pi))

# ── Coupling analysis ────────────────────────────────────────────────
excitation = np.ones(n_elements, dtype=complex)
gamma = active_reflection_coefficient(S, excitation)
z_active = active_impedance(gamma)
blindness = detect_scan_blindness(gamma)

print("=" * 50)
print("APAB Example 02: Coupling-Aware Pattern")
print("=" * 50)
print(f"Array: {spec.size[0]}×{spec.size[1]} at {freq_hz/1e9:.0f} GHz")
print(f"Active reflection (mean |Γ|): {np.mean(np.abs(gamma)):.4f}")
print(f"Active impedance (mean Re): {np.mean(np.real(z_active)):.1f} Ω")
print(f"Scan blindness points: {len(blindness)}")

# ── Pattern with coupling ────────────────────────────────────────────
result_isolated = engine.full_pattern(spec, freq_hz, 0.0, 0.0)
result_coupled = coupled_pattern(
    engine, spec, freq_hz, 0.0, 0.0,
    s_matrix=S,
    excitation=excitation,
)

print(f"\n--- Pattern comparison ---")
print(f"Isolated directivity: {result_isolated.directivity_dbi:.2f} dBi")
print(f"Coupled directivity:  {result_coupled.directivity_dbi:.2f} dBi")
print(f"Difference:           {result_coupled.directivity_dbi - result_isolated.directivity_dbi:.2f} dB")
