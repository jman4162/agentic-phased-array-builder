#!/usr/bin/env python3
"""Example 05: Import external S-parameters from Touchstone file.

Demonstrates importing a Touchstone (.sNp) file, computing coupling
metrics, and running a pattern analysis.
"""

import tempfile
from pathlib import Path

import numpy as np

from apab.core.schemas import ArraySpec, ScanPoint
from apab.coupling.active_impedance import (
    active_impedance,
    active_reflection_coefficient,
    detect_scan_blindness,
)
from apab.emtool.importers import import_touchstone
from apab.pattern.wrappers_pam import PAMPatternEngine

# ── Create a sample Touchstone file ──────────────────────────────────
# In practice, this would come from HFSS, CST, or measurement data.
sample_s2p = """\
! Sample 2-port S-parameters (simulated)
! Freq  S11_Re S11_Im S21_Re S21_Im S12_Re S12_Im S22_Re S22_Im
# GHz S RI R 50
9.0   0.08 -0.25  0.015 0.008  0.015 0.008  0.08 -0.25
9.5   0.10 -0.22  0.020 0.010  0.020 0.010  0.10 -0.22
10.0  0.12 -0.18  0.025 0.012  0.025 0.012  0.12 -0.18
10.5  0.15 -0.15  0.030 0.015  0.030 0.015  0.15 -0.15
11.0  0.18 -0.12  0.035 0.018  0.035 0.018  0.18 -0.12
"""


def main() -> None:
    print("=" * 50)
    print("APAB Example 05: Touchstone Import & Analysis")
    print("=" * 50)

    # Write sample data to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".s2p", delete=False) as f:
        f.write(sample_s2p)
        filepath = f.name

    # ── Import ────────────────────────────────────────────────────────
    data = import_touchstone(filepath)
    print(f"Imported: {filepath}")
    print(f"  Ports: {data['n_ports']}")
    print(f"  Frequencies: {len(data['freqs'])}")
    print(f"  Range: {data['freqs'][0]/1e9:.1f} – {data['freqs'][-1]/1e9:.1f} GHz")
    print(f"  Z₀: {data['z0']} Ω")

    # ── Coupling analysis at 10 GHz ──────────────────────────────────
    mid_idx = len(data["freqs"]) // 2
    s_matrix = data["s_params"][mid_idx]
    freq = data["freqs"][mid_idx]
    excitation = np.ones(data["n_ports"], dtype=complex)

    gamma = active_reflection_coefficient(s_matrix, excitation)
    z_active = active_impedance(gamma)
    blindness = detect_scan_blindness(gamma)

    print(f"\n--- Coupling at {freq/1e9:.1f} GHz ---")
    for i, (g, z) in enumerate(zip(gamma, z_active)):
        print(f"  Port {i+1}: |Γ|={abs(g):.4f}, Z_active={z.real:.1f}+j{z.imag:.1f} Ω")
    print(f"  Scan blindness: {'Yes' if blindness else 'No'}")

    # ── Array pattern ─────────────────────────────────────────────────
    spec = ArraySpec(
        size=[4, 4],
        spacing_m=[0.015, 0.015],
        taper="uniform",
        steer=ScanPoint(theta_deg=0, phi_deg=0),
    )
    engine = PAMPatternEngine()
    result = engine.full_pattern(spec, freq, 0.0, 0.0)

    print(f"\n--- Array Pattern ---")
    print(f"  Directivity: {result.directivity_dbi:.2f} dBi")
    if result.sidelobe_level_db is not None:
        print(f"  Sidelobe level: {result.sidelobe_level_db:.2f} dB")

    # Clean up
    Path(filepath).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
