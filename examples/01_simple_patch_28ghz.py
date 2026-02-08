#!/usr/bin/env python3
"""Example 01: Simple 28 GHz patch array — pattern + system evaluation.

This example demonstrates the minimal APAB workflow:
1. Define an array spec
2. Compute the full radiation pattern
3. Evaluate system-level metrics
4. Print key results
"""

from apab.core.schemas import ArraySpec, ScanPoint
from apab.pattern.wrappers_pam import PAMPatternEngine
from apab.system.wrappers_pas import PASSystemEngine

# ── Array definition ──────────────────────────────────────────────────
freq_hz = 28e9
c = 3e8
wavelength = c / freq_hz
spacing = wavelength / 2  # half-wavelength spacing

spec = ArraySpec(
    size=[8, 8],
    spacing_m=[spacing, spacing],
    taper="taylor",
    steer=ScanPoint(theta_deg=0, phi_deg=0),
)

# ── Pattern computation ───────────────────────────────────────────────
engine = PAMPatternEngine()
result = engine.full_pattern(spec, freq_hz, theta0=0.0, phi0=0.0)

print("=" * 50)
print("APAB Example 01: Simple 28 GHz Patch Array")
print("=" * 50)
print(f"Array size:        {spec.size[0]}×{spec.size[1]}")
print(f"Frequency:         {freq_hz/1e9:.1f} GHz")
print(f"Spacing:           {spacing*1e3:.2f} mm (λ/2)")
print(f"Taper:             {spec.taper}")
print(f"Directivity:       {result.directivity_dbi:.2f} dBi")
print(f"Sidelobe level:    {result.sidelobe_level_db:.2f} dB" if result.sidelobe_level_db else "Sidelobe level:    N/A")
print(f"E-plane HPBW:      {result.metadata.get('e_plane_hpbw_deg', 'N/A')}°")
print(f"H-plane HPBW:      {result.metadata.get('h_plane_hpbw_deg', 'N/A')}°")

# ── System evaluation ─────────────────────────────────────────────────
sys_engine = PASSystemEngine()
rf_spec = {"tx_power_w_per_elem": 0.1, "freq_hz": freq_hz}
arch = sys_engine.build_architecture(spec, rf_spec)
scenario = sys_engine.build_comms_scenario(
    freq_hz=freq_hz,
    bandwidth_hz=400e6,
    range_m=200.0,
    required_snr_db=10.0,
)
metrics = sys_engine.evaluate(arch, scenario)

print(f"\n--- System Metrics ---")
for key, val in sorted(metrics.items()):
    if isinstance(val, (int, float)):
        print(f"  {key}: {val:.4g}")
    elif isinstance(val, bool):
        print(f"  {key}: {val}")
