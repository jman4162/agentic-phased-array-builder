"""Optimization method comparison benchmark for APAB.

Compares four design exploration methods on the same phased-array
optimization problem:

  Problem: Maximize EIRP subject to cost < $10,000 and SNR > 15 dB
  Variables: Nx (4-16), Ny (4-16), tx_power (0.01-0.5 W)
  Fixed: 28 GHz, 200 m range, 400 MHz BW, uniform taper

Methods:
  1. Random sampling (baseline)
  2. Latin Hypercube Sampling (space-filling DOE)
  3. scipy differential_evolution (evolutionary optimizer)
  4. LLM agent (results loaded from apab optimize TSV, if available)

Usage:
  python benchmarks/optimization_comparison.py [--budget 50] [--seed 42]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

# ── Shared evaluation function ────────────────────────────────────────

# Fixed scenario parameters
FREQ_HZ = 28e9
BANDWIDTH_HZ = 400e6
RANGE_M = 200.0
REQUIRED_SNR_DB = 15.0
DX_M = 0.00536
DY_M = 0.00536

# Constraints
MAX_COST_USD = 10_000
MIN_SNR_DB = 15.0

# Bounds
NX_BOUNDS = (4, 16)
NY_BOUNDS = (4, 16)
TX_POWER_BOUNDS = (0.01, 0.5)

_eval_count = 0


def evaluate(nx: int, ny: int, tx_power: float) -> dict[str, float]:
    """Evaluate a single design point using PAS engine."""
    global _eval_count
    _eval_count += 1

    from apab.core.schemas import ArraySpec, ScanPoint
    from apab.system.wrappers_pas import PASSystemEngine

    spec = ArraySpec(
        size=[int(nx), int(ny)],
        spacing_m=[DX_M, DY_M],
        taper="uniform",
        steer=ScanPoint(theta_deg=0, phi_deg=0),
    )
    rf_spec = {"tx_power_w_per_elem": float(tx_power), "freq_hz": FREQ_HZ}

    engine = PASSystemEngine()
    arch = engine.build_architecture(spec, rf_spec)
    scenario = engine.build_comms_scenario(
        freq_hz=FREQ_HZ,
        bandwidth_hz=BANDWIDTH_HZ,
        range_m=RANGE_M,
        required_snr_db=REQUIRED_SNR_DB,
    )
    return engine.evaluate(arch, scenario)


def is_feasible(metrics: dict[str, float]) -> bool:
    cost = metrics.get("cost_usd", float("inf"))
    if isinstance(cost, dict):
        cost = cost.get("total_usd", float("inf"))
    snr = metrics.get("snr_rx_db", -999)
    return cost < MAX_COST_USD and snr > MIN_SNR_DB


def get_eirp(metrics: dict[str, float]) -> float:
    return metrics.get("eirp_dbw", -999)


def get_cost(metrics: dict[str, float]) -> float:
    cost = metrics.get("cost_usd", 0)
    if isinstance(cost, dict):
        return cost.get("total_usd", 0)
    return cost


# ── Method 1: Random Sampling ─────────────────────────────────────────

def run_random(budget: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    results = []
    best_eirp = -999
    convergence = []

    t0 = time.time()
    for _ in range(budget):
        nx = rng.integers(*NX_BOUNDS, endpoint=True)
        ny = rng.integers(*NY_BOUNDS, endpoint=True)
        tx = rng.uniform(*TX_POWER_BOUNDS)
        metrics = evaluate(nx, ny, tx)
        feasible = is_feasible(metrics)
        eirp = get_eirp(metrics)
        results.append({"nx": nx, "ny": ny, "tx": tx, "eirp": eirp,
                        "feasible": feasible})
        if feasible and eirp > best_eirp:
            best_eirp = eirp
        convergence.append(best_eirp if best_eirp > -999 else None)
    wall_time = time.time() - t0

    n_feasible = sum(1 for r in results if r["feasible"])
    return {
        "method": "Random",
        "best_eirp": best_eirp,
        "n_feasible": n_feasible,
        "budget": budget,
        "wall_time_s": wall_time,
        "convergence": convergence,
    }


# ── Method 2: Latin Hypercube Sampling ─────────────────────────────────

def run_lhs(budget: int, seed: int) -> dict:
    from scipy.stats.qmc import LatinHypercube

    sampler = LatinHypercube(d=3, seed=seed)
    samples = sampler.random(n=budget)

    # Scale to bounds
    nx_vals = np.round(
        samples[:, 0] * (NX_BOUNDS[1] - NX_BOUNDS[0]) + NX_BOUNDS[0]
    ).astype(int)
    ny_vals = np.round(
        samples[:, 1] * (NY_BOUNDS[1] - NY_BOUNDS[0]) + NY_BOUNDS[0]
    ).astype(int)
    tx_vals = (
        samples[:, 2] * (TX_POWER_BOUNDS[1] - TX_POWER_BOUNDS[0])
        + TX_POWER_BOUNDS[0]
    )

    results = []
    best_eirp = -999
    convergence = []

    t0 = time.time()
    for i in range(budget):
        metrics = evaluate(nx_vals[i], ny_vals[i], tx_vals[i])
        feasible = is_feasible(metrics)
        eirp = get_eirp(metrics)
        results.append({"eirp": eirp, "feasible": feasible})
        if feasible and eirp > best_eirp:
            best_eirp = eirp
        convergence.append(best_eirp if best_eirp > -999 else None)
    wall_time = time.time() - t0

    n_feasible = sum(1 for r in results if r["feasible"])
    return {
        "method": "LHS",
        "best_eirp": best_eirp,
        "n_feasible": n_feasible,
        "budget": budget,
        "wall_time_s": wall_time,
        "convergence": convergence,
    }


# ── Method 3: scipy differential_evolution ─────────────────────────────

def run_scipy_de(budget: int, seed: int) -> dict:
    convergence = []
    best_eirp = -999
    eval_log: list[dict] = []

    def objective(x: np.ndarray) -> float:
        nonlocal best_eirp
        nx, ny, tx = int(round(x[0])), int(round(x[1])), x[2]
        metrics = evaluate(nx, ny, tx)
        eirp = get_eirp(metrics)
        feasible = is_feasible(metrics)

        eval_log.append({"eirp": eirp, "feasible": feasible})
        if feasible and eirp > best_eirp:
            best_eirp = eirp
        convergence.append(best_eirp if best_eirp > -999 else None)

        # Penalty for infeasible designs
        if not feasible:
            return 100.0  # large penalty (minimizing)
        return -eirp  # negate because DE minimizes

    bounds = [NX_BOUNDS, NY_BOUNDS, TX_POWER_BOUNDS]

    # popsize=5 → 15 evals/generation for 3 variables
    # maxiter controls generations; total evals ≈ popsize*len(bounds)*(maxiter+1)
    popsize = 5
    max_gen = max(1, budget // (popsize * len(bounds)) - 1)

    t0 = time.time()
    differential_evolution(
        objective,
        bounds=bounds,
        maxiter=max_gen,
        popsize=popsize,
        seed=seed,
        tol=0,
        atol=0,
        init="latinhypercube",
        polish=False,
    )
    wall_time = time.time() - t0

    n_feasible = sum(1 for r in eval_log if r["feasible"])
    actual_evals = len(eval_log)
    return {
        "method": "scipy DE",
        "best_eirp": best_eirp,
        "n_feasible": n_feasible,
        "budget": actual_evals,
        "wall_time_s": wall_time,
        "convergence": convergence,
    }


# ── Method 4: LLM Agent (load from results.tsv) ───────────────────────

def load_agent_results(tsv_path: Path) -> dict | None:
    """Load LLM agent results from an apab optimize TSV file."""
    if not tsv_path.exists():
        return None

    import csv

    results = []
    best_eirp = -999
    convergence = []

    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eirp = float(row.get("eirp_dbw", -999))
            status = row.get("status", "")
            feasible = status in ("keep", "baseline")
            results.append({"eirp": eirp, "feasible": feasible})
            if feasible and eirp > best_eirp:
                best_eirp = eirp
            convergence.append(best_eirp if best_eirp > -999 else None)

    if not results:
        return None

    return {
        "method": "LLM Agent",
        "best_eirp": best_eirp,
        "n_feasible": sum(1 for r in results if r["feasible"]),
        "budget": len(results),
        "wall_time_s": None,  # not available from TSV
        "convergence": convergence,
    }


# ── Main ───────────────────────────────────────────────────────────────

def print_comparison(results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("OPTIMIZATION METHOD COMPARISON")
    print(
        f"Problem: Maximize EIRP | cost < ${MAX_COST_USD:,} | "
        f"SNR > {MIN_SNR_DB} dB"
    )
    print("=" * 72)
    print(
        f"{'Method':<15} {'Best EIRP':>10} {'Feasible':>10} "
        f"{'Evals':>8} {'Time (s)':>10} {'Taper flex':>12} "
        f"{'Explains':>10}"
    )
    print("-" * 72)
    for r in results:
        eirp = f"{r['best_eirp']:.1f}" if r["best_eirp"] > -999 else "N/A"
        feas = f"{r['n_feasible']}/{r['budget']}"
        t = f"{r['wall_time_s']:.1f}" if r["wall_time_s"] else "N/A"
        taper = "Yes" if r["method"] == "LLM Agent" else "No"
        explains = "Yes" if r["method"] == "LLM Agent" else "No"
        print(
            f"{r['method']:<15} {eirp:>10} {feas:>10} "
            f"{r['budget']:>8} {t:>10} {taper:>12} {explains:>10}"
        )
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare optimization methods for phased-array design",
    )
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--agent-results",
        type=str,
        default="workspace/optimize/results.tsv",
        help="Path to LLM agent results TSV",
    )
    args = parser.parse_args()

    all_results = []

    print(f"\nBudget: {args.budget} evaluations per method, seed={args.seed}")
    print()

    global _eval_count

    _eval_count = 0
    print("Running Random sampling...")
    all_results.append(run_random(args.budget, args.seed))
    print(f"  Done ({_eval_count} evals)")

    _eval_count = 0
    print("Running LHS...")
    all_results.append(run_lhs(args.budget, args.seed))
    print(f"  Done ({_eval_count} evals)")

    _eval_count = 0
    print("Running scipy differential_evolution...")
    all_results.append(run_scipy_de(args.budget, args.seed))
    print(f"  Done ({_eval_count} evals)")

    agent = load_agent_results(Path(args.agent_results))
    if agent:
        print(f"Loaded LLM agent results from {args.agent_results}")
        all_results.append(agent)
    else:
        print(
            f"No agent results at {args.agent_results}. "
            "Run: apab optimize --protocol research.md"
        )

    print_comparison(all_results)


if __name__ == "__main__":
    main()
