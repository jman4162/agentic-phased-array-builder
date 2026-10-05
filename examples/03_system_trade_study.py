#!/usr/bin/env python3
"""Example 03: System-level DOE trade study.

Runs a design-of-experiments study varying array size and TX power,
then identifies Pareto-optimal designs.
"""

import pandas as pd

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
    {"name": "array.ny", "type": "int", "low": 4, "high": 16},
    {"name": "rf.tx_power_w_per_elem", "type": "float", "low": 0.01, "high": 0.5},
    # Allow arbitrary array sizes (disable sub-array divisibility check)
    {"name": "array.enforce_subarray_constraint", "type": "categorical", "values": [False]},
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

n_total = len(pd.DataFrame(result["results"]))
n_pareto = len(pd.DataFrame(result["pareto"]))
print(f"Total designs evaluated: {n_total}")
print(f"Failed cases: {result['n_failed']}")
print(f"Feasible designs: {result['n_feasible']}")
print(f"Pareto-optimal: {n_pareto} (objectives: {result['pareto_objectives']})")
print("\nTrade study complete.")
