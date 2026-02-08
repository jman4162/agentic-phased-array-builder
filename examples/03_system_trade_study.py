#!/usr/bin/env python3
"""Example 03: System-level DOE trade study.

Runs a design-of-experiments study varying array size and TX power,
then identifies Pareto-optimal designs.
"""

from apab.system.wrappers_pas import PASSystemEngine

engine = PASSystemEngine()

# ── Build scenario ────────────────────────────────────────────────────
scenario = engine.build_comms_scenario(
    freq_hz=28e9,
    bandwidth_hz=400e6,
    range_m=200.0,
    required_snr_db=10.0,
)

# ── Define design variables ──────────────────────────────────────────
variables = [
    {"name": "array.nx", "type": "int", "low": 4, "high": 16},
    {"name": "rf.tx_power_w_per_elem", "type": "float", "low": 0.01, "high": 0.5},
]

# ── Run trade study ──────────────────────────────────────────────────
print("=" * 50)
print("APAB Example 03: System Trade Study")
print("=" * 50)
print("Running DOE with 20 samples (LHS)...")

result = engine.run_trade_study(
    scenario=scenario,
    variables=variables,
    n_samples=20,
    method="lhs",
    seed=42,
)

print(f"Total designs evaluated: {len(result.get('results', {}))} columns")
print(f"Feasible designs: {result['n_feasible']}")
print(f"Pareto-optimal: {len(result.get('pareto', {}))} columns")
print("\nTrade study complete.")
