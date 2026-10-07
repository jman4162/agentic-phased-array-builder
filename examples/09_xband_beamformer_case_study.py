#!/usr/bin/env python3
"""Case study: an X-band AESA built on a 4-channel beamformer that drives FEMs directly.

Prompted by Qorvo's QPX0252 announcement (October 2026): a 4-channel,
7.9-12 GHz beamformer IC whose transmit output drives the QPF5012 front-end
module without a discrete driver amplifier, with "up to 20 dB higher cascaded
input P1dB" in a high-linearity receive mode. The QPX0252 data sheet brief
(Rev A, September 2026, pre-production) gives 6-bit phase (5.625 deg LSB) and
6-bit gain (0.5 dB LSB) control but no RF numbers, so every beamformer RF
parameter below is an assumption, bracketed against public numbers for the
ADAR1000 (a conventional 4-channel X/Ku beamformer) and the QPF5012 FEM.

The script asks what has to be true for the vendor claims to hold and what
they change at element, array and mission level:

  1. Receive cascade: FEM -> beamformer, sweeping beamformer IP1dB and NF
  2. Transmit drive: does the beamformer reach the FEM's saturation drive,
     and what does dropping the driver save in parts, DC power and heat
  3. Aperture: lattice for the FEM band, phase-bit loss, beam squint,
     whole-IC (2x2 tile) failures versus random element failures
  4. Mission: detection range versus receive NF; survivable range to an
     in-band marine navigation radar versus cascaded IP1dB

Outputs go to examples/output/xband_qpx0252/ (figures + results.json).

Usage:
    python examples/09_xband_beamformer_case_study.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import phased_array as pa
from phased_array_systems.models.antenna.errors import phase_quantization_loss_db
from phased_array_systems.models.antenna.grating import check_grating_lobes
from phased_array_systems.models.rf.cascade import RFStage, cascade_analysis
from phased_array_systems.models.swapc.cooling import minimum_class_for

from apab.core.schemas import ArraySpec, ScanPoint
from apab.system.wrappers_pas import PASSystemEngine

OUT_DIR = Path(__file__).parent / "output" / "xband_qpx0252"
C = 299_792_458.0

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.fontsize": 8,
    }
)

# ── Assumptions ───────────────────────────────────────────────────────
# Sourced values carry a "src"; everything else is an assumption.
FEM = {  # Qorvo QPF5012, 8.5-10.5 GHz (product page / distributor summary)
    "band_ghz": (8.5, 10.5),
    "rx_gain_db": 23.0,
    "rx_nf_db": 2.2,
    "rx_psat_dbm": 13.8,
    "rx_oip3_dbm": 20.7,
    "tx_gain_db": 20.5,
    "tx_psat_dbm": 40.5,
    "tx_pae": 0.40,
    "src": "Qorvo QPF5012 product summary",
}
# Rx input P1dB of the FEM is not published; take Psat - 3 dB output, referred
# to the input. OIP3 - gain gives IIP3.
FEM["rx_ip1db_dbm"] = FEM["rx_psat_dbm"] - 3.0 - FEM["rx_gain_db"] + 1.0
FEM["rx_iip3_dbm"] = FEM["rx_oip3_dbm"] - FEM["rx_gain_db"]

CONVENTIONAL_BF = {  # ADAR1000 at 9.5 GHz, nominal bias (datasheet)
    "rx_gain_db": 10.0,
    "rx_ip1db_dbm": -16.0,
    "rx_iip3_dbm": -7.0,
    "tx_op1db_dbm": 10.0,
    "tx_psat_dbm": 14.0,
    "phase_lsb_deg": 2.8,
    "src": "ADAR1000 data sheet, Rev. A",
}
QPX0252 = {
    "phase_bits": 6,
    "gain_lsb_db": 0.5,
    "gain_steps": 64,
    "src": "QPX0252 data sheet brief, Rev A (pre-production)",
}
BF_RX_NF_DB = (8.0, 14.0, 20.0)  # assumption bracket; not published for QPX0252
DRIVER = {"gain_db": 12.0, "op1db_dbm": 25.0, "dc_w": 1.0}  # generic GaAs driver, assumed

F0 = 9.5e9  # FEM band centre
F_MAX = FEM["band_ghz"][1] * 1e9
SCAN_LIMIT_DEG = 60.0
NX = NY = 32  # 1024 elements -> 256 four-channel ICs in 2x2 tiles
DUTY = 0.10
RADAR = {
    "bandwidth_hz": 1e6,
    "range_m": 20_000.0,
    "target_rcs_dbsm": 0.0,
    "n_pulses": 16,
    "swerling": 1,
    "pfa": 1e-6,
    "pd_required": 0.9,
    "integration_type": "noncoherent",
}


def _save(fig, name: str) -> str:
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


# ── 1. Receive cascade ───────────────────────────────────────────────


def rx_chain(
    bf_ip1db: float, bf_nf: float, bf_gain: float = 10.0, bf_iip3: float | None = None
) -> dict:
    stages = [
        RFStage(
            "FEM (limiter+LNA)",
            FEM["rx_gain_db"],
            FEM["rx_nf_db"],
            iip3_dbm=FEM["rx_iip3_dbm"],
            p1db_dbm=FEM["rx_ip1db_dbm"],
        ),
        RFStage(
            "beamformer Rx",
            bf_gain,
            bf_nf,
            iip3_dbm=bf_iip3 if bf_iip3 is not None else bf_ip1db + 9.0,
            p1db_dbm=bf_ip1db,
        ),
    ]
    return cascade_analysis(stages, bandwidth_hz=RADAR["bandwidth_hz"])


def experiment_rx_cascade() -> dict:
    ref = rx_chain(
        CONVENTIONAL_BF["rx_ip1db_dbm"], BF_RX_NF_DB[0], bf_iip3=CONVENTIONAL_BF["rx_iip3_dbm"]
    )
    bf_ip1 = np.linspace(-20.0, 15.0, 141)
    ip1 = np.array([rx_chain(p, 10.0)["ip1db_dbm"] for p in bf_ip1])
    gain_db = ip1 - ref["ip1db_dbm"]
    # Beamformer IP1dB that buys +20 dB at the chain input
    need = float(np.interp(20.0, gain_db, bf_ip1))
    ceiling = FEM["rx_ip1db_dbm"]
    nf = {nf_bf: rx_chain(need, nf_bf)["total_nf_db"] for nf_bf in BF_RX_NF_DB}
    nf_sweep = np.linspace(4.0, 26.0, 89)
    nf_curve = np.array([rx_chain(need, n)["total_nf_db"] for n in nf_sweep])

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(bf_ip1, ip1, color="#1f77b4")
    ax[0].axhline(ref["ip1db_dbm"], ls="--", color="gray", lw=1)
    ax[0].axhline(ceiling, ls=":", color="#d62728", lw=1)
    ax[0].axvline(CONVENTIONAL_BF["rx_ip1db_dbm"], ls="--", color="gray", lw=1)
    ax[0].plot([need], [ref["ip1db_dbm"] + 20], "o", color="#ff7f0e")
    ax[0].annotate(
        f"+20 dB needs beamformer\nIP1dB ≈ {need:+.1f} dBm",
        (need, ref["ip1db_dbm"] + 20),
        xytext=(-14, -18),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    ax[0].text(-19.5, ref["ip1db_dbm"] + 1, "ADAR1000-class reference", fontsize=8)
    ax[0].text(-19.5, ceiling - 2.2, "FEM LNA alone (ceiling)", fontsize=8, color="#d62728")
    ax[0].set_xlabel("Beamformer Rx input P1dB (dBm)")
    ax[0].set_ylabel("Cascaded input P1dB at FEM port (dBm)")
    ax[0].set_title("Rx linearity: FEM (23 dB) → beamformer")
    ax[1].plot(nf_sweep, nf_curve, color="#2ca02c")
    ax[1].axhline(FEM["rx_nf_db"], ls=":", color="gray", lw=1)
    ax[1].set_xlabel("Beamformer Rx noise figure (dB)")
    ax[1].set_ylabel("Cascaded NF (dB)")
    ax[1].set_title("Noise cost of the beamformer stage")
    fig_name = _save(fig, "fig1_rx_cascade.png")

    return {
        "fem_rx_ip1db_dbm_assumed": FEM["rx_ip1db_dbm"],
        "reference_chain": {k: ref[k] for k in ("total_nf_db", "ip1db_dbm", "iip3_dbm", "sfdr_db")},
        "reference_binding_stage": ref["p1db_binding_stage"],
        "bf_ip1db_for_plus20_dbm": need,
        "chain_ip1db_ceiling_dbm": ceiling,
        "max_possible_improvement_db": ceiling - ref["ip1db_dbm"],
        "cascaded_nf_at_bf_nf": {f"{k:g}": v for k, v in nf.items()},
        "figure": fig_name,
    }


# ── 2. Transmit drive, parts, power, heat ────────────────────────────


def experiment_tx_drive() -> dict:
    drive_needed = FEM["tx_psat_dbm"] - FEM["tx_gain_db"]

    def fem_pout(pin_dbm: np.ndarray) -> np.ndarray:
        # Rapp-style soft limiter (p=2) around the FEM's saturated output
        psat = 10 ** (FEM["tx_psat_dbm"] / 10)
        lin = 10 ** ((pin_dbm + FEM["tx_gain_db"] + 1.0) / 10)
        return 10 * np.log10(lin / (1 + (lin / psat) ** 2) ** 0.5)

    pin = np.linspace(5, 26, 85)
    direct = fem_pout(pin)
    conv_bf_out = np.minimum(pin, CONVENTIONAL_BF["tx_psat_dbm"])
    conv_with_driver = fem_pout(np.minimum(conv_bf_out + DRIVER["gain_db"], DRIVER["op1db_dbm"]))

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(pin, direct, label="beamformer → FEM (direct drive)")
    ax.plot(
        pin,
        conv_with_driver,
        ls="--",
        label=f"ADAR1000-class → {DRIVER['gain_db']:.0f} dB driver → FEM",
    )
    ax.axvline(drive_needed, ls=":", color="#d62728", lw=1)
    ax.axvline(CONVENTIONAL_BF["tx_psat_dbm"], ls="--", color="gray", lw=1)
    ax.axhline(FEM["tx_psat_dbm"], ls=":", color="gray", lw=1)
    ax.text(
        drive_needed + 0.3,
        30,
        f"FEM drive for Psat\n≈ {drive_needed:.0f} dBm",
        fontsize=8,
        color="#d62728",
    )
    ax.text(
        CONVENTIONAL_BF["tx_psat_dbm"] - 0.3,
        26,
        "ADAR1000\nPsat",
        fontsize=8,
        ha="right",
        color="gray",
    )
    ax.set_xlabel("Beamformer channel output power (dBm)")
    ax.set_ylabel("FEM output power (dBm)")
    ax.set_title("Tx: beamformer channel power the FEM needs")
    ax.legend(loc="lower right")
    fig_name = _save(fig, "fig2_tx_drive.png")

    n = NX * NY
    n_ic = n // 4
    fem_dc_peak = 10 ** ((FEM["tx_psat_dbm"] - 30) / 10) / FEM["tx_pae"]
    fem_rf_w = 10 ** ((FEM["tx_psat_dbm"] - 30) / 10)
    cell_cm2 = (dx_m() * 100) ** 2
    diss_avg = (fem_dc_peak - fem_rf_w) * DUTY
    driver_avg = DRIVER["dc_w"] * DUTY
    flux = diss_avg / cell_cm2
    return {
        "fem_drive_for_psat_dbm": drive_needed,
        "conventional_bf_psat_dbm": CONVENTIONAL_BF["tx_psat_dbm"],
        "drive_shortfall_conventional_db": drive_needed - CONVENTIONAL_BF["tx_psat_dbm"],
        "fem_pout_from_conventional_bf_direct_dbm": float(fem_pout(np.array([14.0]))[0]),
        "rf_parts_conventional": n_ic + 2 * n,
        "rf_parts_direct": n_ic + n,
        "n_ics": n_ic,
        "fem_dc_peak_w_per_elem": fem_dc_peak,
        "driver_dc_share_of_tx_dc_pct": 100 * DRIVER["dc_w"] / (fem_dc_peak + DRIVER["dc_w"]),
        "heat_flux_w_per_cm2_avg": flux,
        "heat_flux_driver_removed_w_per_cm2": driver_avg / cell_cm2,
        "min_cooling_class": minimum_class_for(flux),
        "figure": fig_name,
    }


# ── 3. Aperture ──────────────────────────────────────────────────────


def dx_m() -> float:
    lam_min = C / F_MAX
    safe = check_grating_lobes(0.5, 0.5, SCAN_LIMIT_DEG)["max_safe_spacing_lambda"]
    return math.floor(safe * lam_min * 1e4) / 1e4  # round down to 0.1 mm


def _cut(weights, geom, k, theta0=0.0, n=1441):
    th = np.linspace(-90, 90, n)
    af = pa.array_factor_vectorized(np.deg2rad(th), np.zeros_like(th), geom.x, geom.y, weights, k)
    p = 20 * np.log10(np.abs(af) / np.abs(af).max() + 1e-12)
    return th, p


def _peak_sll(th, p, theta0, _unused=None):
    """Highest lobe outside the main lobe, bounded by the first nulls."""
    i = int(np.argmax(np.where(np.abs(th - theta0) < 5.0, p, -np.inf)))
    lo = i
    while lo > 0 and p[lo - 1] <= p[lo]:
        lo -= 1
    hi = i
    while hi < len(p) - 1 and p[hi + 1] <= p[hi]:
        hi += 1
    side = np.concatenate([p[:lo], p[hi + 1 :]])
    return float(side.max())


def experiment_aperture() -> dict:
    d = dx_m()
    lam0 = C / F0
    k0 = 2 * np.pi / lam0
    geom = pa.create_rectangular_array(NX, NY, dx=d, dy=d, wavelength=1.0)
    taper = pa.taylor_taper_2d(NX, NY, sidelobe_dB=-30.0).ravel()
    sv = pa.steering_vector(k0, geom.x, geom.y, 0.0, 0.0)
    w0 = taper * sv
    th, p_ideal = _cut(w0, geom, k0)
    hw = 4.0
    sll_ideal = _peak_sll(th, p_ideal, 0.0, hw)

    bits = {}
    for b in (5, 6, 7):
        worst = -np.inf
        for scan in np.arange(5.0, 55.1, 2.5):
            wq = pa.quantize_phase(taper * pa.steering_vector(k0, geom.x, geom.y, scan, 0.0), b)
            worst = max(worst, _peak_sll(*_cut(wq, geom, k0), scan))
        bits[str(b)] = {
            "pq_loss_db": phase_quantization_loss_db(b),
            "worst_sll_5_to_55deg_db": float(worst),
        }

    # QPX0252 control as published: 6-bit phase plus 0.5 dB gain steps over 31.5 dB
    att_db = -20 * np.log10(taper / taper.max())
    att_max = QPX0252["gain_lsb_db"] * (QPX0252["gain_steps"] - 1)
    att_q = np.clip(np.round(att_db / QPX0252["gain_lsb_db"]) * QPX0252["gain_lsb_db"], 0, att_max)
    taper_q = 10 ** (-att_q / 20)
    worst_q = -np.inf
    for scan in np.arange(5.0, 55.1, 2.5):
        sv_s = pa.steering_vector(k0, geom.x, geom.y, scan, 0.0)
        wq = pa.quantize_phase(taper_q * sv_s, QPX0252["phase_bits"])
        worst_q = max(worst_q, _peak_sll(*_cut(wq, geom, k0), scan))
    bits["qpx0252_phase_and_gain"] = {
        "taper_edge_att_db": float(att_db.max()),
        "worst_sll_5_to_55deg_db": float(worst_q),
    }

    # Whole-IC failures (2x2 tile dark) versus the same number of random elements
    rng = np.random.default_rng(7)
    ix = np.arange(NX * NY) % NX
    iy = np.arange(NX * NY) // NX
    tile = (iy // 2) * (NX // 2) + (ix // 2)
    n_trials = 200
    fail_ics = (1, 4, 8, 16)
    tile_sll, rand_sll = {}, {}
    for nf in fail_ics:
        t_s, r_s = [], []
        for _ in range(n_trials):
            dead = rng.choice(NX * NY // 4, nf, replace=False)
            m = ~np.isin(tile, dead)
            t_s.append(_peak_sll(*_cut(w0 * m, geom, k0, n=721), 0.0, hw))
            m2 = np.ones(NX * NY, bool)
            m2[rng.choice(NX * NY, 4 * nf, replace=False)] = False
            r_s.append(_peak_sll(*_cut(w0 * m2, geom, k0, n=721), 0.0, hw))
        tile_sll[nf] = np.array(t_s)
        rand_sll[nf] = np.array(r_s)

    # Beam squint across the FEM band at 60 deg scan, phase steering vs
    # TTD at 8x8 subarrays (16 delay lines) with phase inside
    freqs = np.linspace(8.5e9, 10.5e9, 21)
    xs, ys = geom.x, geom.y
    sq = {}
    for mode in ("phase", "hybrid", "ttd"):
        arch = (
            pa.create_rectangular_subarrays(NX, NY, 8, 8, d, d, 1.0) if mode == "hybrid" else None
        )
        r = pa.compute_beam_squint(
            xs, ys, 60.0, 0.0, F0, freqs, steering_mode=mode, architecture=arch, n_points=1441
        )
        sq[mode] = r
    # Gain toward the commanded 60 deg, relative to an ideal TTD beam
    sub_x = (geom.x // (8 * d)) * 8 * d + 3.5 * d  # 8x8 subarray centres in x
    u0 = math.sin(math.radians(60.0))
    gain_at_cmd = {}
    for mode in ("phase", "hybrid"):
        g = []
        for f in freqs:
            kf = 2 * np.pi * f / C
            if mode == "phase":
                phase = k0 * geom.x * u0
            else:
                phase = kf * sub_x * u0 + k0 * (geom.x - sub_x) * u0
            af = np.abs(np.sum(np.exp(1j * (kf * geom.x * u0 - phase)))) / geom.x.size
            g.append(20 * np.log10(af))
        gain_at_cmd[mode] = np.array(g)
    bw_deg = math.degrees(0.886 * lam0 / (NX * d) / math.cos(math.radians(60)))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    labels = {
        "phase": "phase shifters only",
        "hybrid": f"TTD per 8×8 subarray ({gain_at_cmd['hybrid'][0]:.1f} dB at band edges)",
        "ttd": "TTD per element",
    }
    styles = {"phase": "-", "hybrid": "-", "ttd": ":"}
    for mode, r in sq.items():
        ax[0].plot(
            freqs / 1e9,
            r["beam_angles"],
            styles[mode],
            lw=2 if mode == "ttd" else 1.5,
            label=labels[mode],
        )
    ax[0].axhspan(
        60 - bw_deg / 2, 60 + bw_deg / 2, color="gray", alpha=0.15, label="3 dB beamwidth at 60°"
    )
    ax[0].set_xlabel("Frequency (GHz), weights set at 9.5 GHz")
    ax[0].set_ylabel("Beam peak, commanded 60° (deg)")
    ax[0].set_title("Squint across the FEM band")
    ax[0].legend()
    pos = np.arange(len(fail_ics))
    ax[1].boxplot(
        [tile_sll[n] for n in fail_ics],
        positions=pos - 0.18,
        widths=0.3,
        patch_artist=True,
        boxprops={"facecolor": "#ff7f0e", "alpha": 0.6},
        medianprops={"color": "k"},
        showfliers=False,
    )
    ax[1].boxplot(
        [rand_sll[n] for n in fail_ics],
        positions=pos + 0.18,
        widths=0.3,
        patch_artist=True,
        boxprops={"facecolor": "#1f77b4", "alpha": 0.6},
        medianprops={"color": "k"},
        showfliers=False,
    )
    ax[1].axhline(sll_ideal, ls=":", color="gray")
    ax[1].set_xticks(pos, [f"{n} IC\n({4 * n} el.)" for n in fail_ics])
    ax[1].set_ylabel("Peak sidelobe, broadside cut (dB)")
    ax[1].set_title("Dead ICs (orange) vs same count of random elements (blue)")
    fig_name = _save(fig, "fig3_aperture.png")

    return {
        "dx_m": d,
        "dx_lambda_at_f0": d / lam0,
        "dx_lambda_at_fmax": d * F_MAX / C,
        "aperture_m": NX * d,
        "sll_ideal_db": sll_ideal,
        "phase_bits": bits,
        "beam_angle_deg_at_band_edges": {
            m: [float(r["beam_angles"][0]), float(r["beam_angles"][-1])] for m, r in sq.items()
        },
        "gain_at_60deg_db_at_band_edges": {
            m: [float(v[0]), float(v[-1])] for m, v in gain_at_cmd.items()
        },
        "half_beamwidth_at_60_deg": bw_deg / 2,
        "tile_failure_sll_median_db": {str(n): float(np.median(v)) for n, v in tile_sll.items()},
        "random_failure_sll_median_db": {str(n): float(np.median(v)) for n, v in rand_sll.items()},
        "tile_failure_sll_p90_db": {
            str(n): float(np.percentile(v, 90)) for n, v in tile_sll.items()
        },
        "random_failure_sll_p90_db": {
            str(n): float(np.percentile(v, 90)) for n, v in rand_sll.items()
        },
        "figure": fig_name,
    }


# ── 4. Mission ───────────────────────────────────────────────────────


def evaluate_radar(noise_figure_db: float, **extra) -> dict:
    d = dx_m()
    eng = PASSystemEngine()
    spec = ArraySpec(
        size=[NX, NY], spacing_m=[d, d], taper="uniform", steer=ScanPoint(theta_deg=0, phi_deg=0)
    )
    arch = eng.build_architecture(
        spec,
        {
            "tx_power_w_per_elem": 10 ** ((FEM["tx_psat_dbm"] - 30) / 10),
            "freq_hz": F0,
            "noise_figure_db": noise_figure_db,
            "pa_efficiency": FEM["tx_pae"],
        },
    )
    radar = {**RADAR, **extra}
    scen = eng.build_radar_scenario(freq_hz=F0, **radar)
    return eng.evaluate(arch, scen)


def experiment_mission(rx: dict) -> dict:
    nfs = np.linspace(2.4, 6.0, 19)
    rng_km = np.array([evaluate_radar(n)["detection_range_m"] / 1e3 for n in nfs])
    nf_lo = rx["cascaded_nf_at_bf_nf"]["8"]
    nf_hi = rx["cascaded_nf_at_bf_nf"]["20"]
    r_lo = evaluate_radar(nf_lo)["detection_range_m"] / 1e3
    r_hi = evaluate_radar(nf_hi)["detection_range_m"] / 1e3
    sea = {
        n: evaluate_radar(n, clutter_type="sea", sea_state=3, range_m=10_000.0)
        for n in (nf_lo, nf_hi)
    }

    # In-band interferer: a marine navigation radar at 9.41 GHz. Assumed 25 kW
    # peak into a 30 dBi slotted waveguide (main beam) with -25 dB sidelobes.
    lam = C / 9.41e9
    elem_gain_dbi = 5.0
    eirp = {
        "main beam": 10 * math.log10(25e3) + 30 + 30,
        "sidelobe": 10 * math.log10(25e3) + 30 + 5,
    }

    def r_min_m(eirp_dbm: float, ip1: float) -> float:
        # range at which the per-element input equals the chain IP1dB
        return lam / (4 * math.pi) * 10 ** ((eirp_dbm + elem_gain_dbi - ip1) / 20)

    ip1_ref = rx["reference_chain"]["ip1db_dbm"]
    ip1_new = ip1_ref + 20.0
    survive = {
        k: {"conventional_m": r_min_m(v, ip1_ref), "plus20_m": r_min_m(v, ip1_new)}
        for k, v in eirp.items()
    }

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(nfs, rng_km, color="#1f77b4")
    for nf, r, lbl in ((nf_lo, r_lo, "BF NF 8 dB"), (nf_hi, r_hi, "BF NF 20 dB")):
        ax[0].plot([nf], [r], "o")
        ax[0].annotate(
            f"{lbl}: {r:.1f} km", (nf, r), xytext=(6, 4), textcoords="offset points", fontsize=8
        )
    ax[0].set_xlabel("Cascaded receive NF (dB)")
    ax[0].set_ylabel("Detection range, 1 m², Pd 0.9 (km)")
    ax[0].set_title("Noise-limited search: NF sets range")
    ip1_axis = np.linspace(-45, -5, 81)
    for k, v in eirp.items():
        ax[1].semilogy(ip1_axis, [r_min_m(v, p) / 1e3 for p in ip1_axis], label=f"nav radar {k}")
    ax[1].axvline(ip1_ref, ls="--", color="gray", lw=1)
    ax[1].axvline(ip1_new, ls="--", color="#ff7f0e", lw=1)
    ax[1].text(ip1_ref + 0.5, 150, "conventional", fontsize=8, color="gray")
    ax[1].text(ip1_new + 0.5, 150, "+20 dB", fontsize=8, color="#ff7f0e")
    ax[1].set_xlabel("Cascaded input P1dB at the element (dBm)")
    ax[1].set_ylabel("Closest range before 1 dB compression (km)")
    ax[1].set_title("Large-signal case: in-band marine radar")
    ax[1].legend(loc="center right")
    fig_name = _save(fig, "fig4_mission.png")

    return {
        "range_km_bf_nf_8": r_lo,
        "range_km_bf_nf_20": r_hi,
        "range_loss_pct": 100 * (1 - r_hi / r_lo),
        "sea_state3_10km": {
            f"{k:.2f}": {m: v[m] for m in ("scnr_db", "scr_db", "snr_single_pulse_db")}
            for k, v in sea.items()
        },
        "interferer_eirp_dbm": eirp,
        "survival_range_m": survive,
        "figure": fig_name,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"
    results = {}

    results["assumptions"] = {
        "qpx0252": QPX0252,
        "fem": FEM,
        "conventional_bf": CONVENTIONAL_BF,
        "bf_rx_nf_bracket_db": BF_RX_NF_DB,
        "driver": DRIVER,
        "f0_hz": F0,
        "array": [NX, NY],
        "duty": DUTY,
        "radar": RADAR,
    }
    results["rx_cascade"] = experiment_rx_cascade()
    results["tx_drive"] = experiment_tx_drive()
    results["aperture"] = experiment_aperture()
    results["mission"] = experiment_mission(results["rx_cascade"])
    results_path.write_text(json.dumps(results, indent=2, default=float))
    print(
        json.dumps(
            {k: v for k, v in results.items() if k != "assumptions"}, indent=1, default=float
        )
    )


if __name__ == "__main__":
    main()
