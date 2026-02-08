"""Importers for common EM data file formats (Touchstone, far-field CSV)."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

# ── Touchstone import ──


_FREQ_MULTIPLIERS: dict[str, float] = {
    "HZ": 1.0,
    "KHZ": 1.0e3,
    "MHZ": 1.0e6,
    "GHZ": 1.0e9,
}


def _parse_option_line(line: str) -> tuple[float, str, str, float]:
    """Parse the Touchstone option line.

    Returns (freq_multiplier, param_type, data_format, z0).
    """
    # Strip leading '#' and normalise whitespace
    tokens = line.lstrip("#").upper().split()

    # Defaults per Touchstone spec
    freq_unit = "GHZ"
    param_type = "S"
    data_format = "MA"
    z0 = 50.0

    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if tok in _FREQ_MULTIPLIERS:
            freq_unit = tok
        elif tok in {"S", "Y", "Z", "H", "G"}:
            param_type = tok
        elif tok in {"RI", "MA", "DB"}:
            data_format = tok
        elif tok == "R":
            idx += 1
            if idx < len(tokens):
                z0 = float(tokens[idx])
        idx += 1

    return _FREQ_MULTIPLIERS[freq_unit], param_type, data_format, z0


def _values_to_complex(v1: float, v2: float, data_format: str) -> complex:
    """Convert a pair of raw values to a complex number."""
    if data_format == "RI":
        return complex(v1, v2)
    elif data_format == "MA":
        mag = v1
        angle_rad = math.radians(v2)
        return complex(mag * math.cos(angle_rad), mag * math.sin(angle_rad))
    elif data_format == "DB":
        mag = 10.0 ** (v1 / 20.0)
        angle_rad = math.radians(v2)
        return complex(mag * math.cos(angle_rad), mag * math.sin(angle_rad))
    else:
        raise ValueError(f"Unknown Touchstone data format: {data_format}")


def import_touchstone(filepath: str | Path) -> dict[str, Any]:
    """Parse a Touchstone (.s1p, .s2p, .sNp) file.

    Parameters
    ----------
    filepath:
        Path to the Touchstone file.

    Returns
    -------
    dict
        ``"freqs"`` -- list of frequencies in Hz.
        ``"s_params"`` -- list of 2-D numpy arrays (n_ports x n_ports complex)
        per frequency point.
        ``"n_ports"`` -- number of ports.
        ``"z0"`` -- reference impedance.
        ``"comments"`` -- list of comment strings.
    """
    filepath = Path(filepath)

    # Determine number of ports from extension
    ext = filepath.suffix.lower()
    match = re.match(r"\.s(\d+)p", ext)
    if match:
        n_ports = int(match.group(1))
    else:
        raise ValueError(f"Cannot determine port count from extension: {ext}")

    comments: list[str] = []
    freq_mult = 1.0e9
    data_format = "MA"
    z0 = 50.0
    option_found = False

    raw_data_lines: list[str] = []

    with open(filepath) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("!"):
                comments.append(stripped.lstrip("!").strip())
                continue
            if stripped.startswith("#"):
                freq_mult, _, data_format, z0 = _parse_option_line(stripped)
                option_found = True
                continue
            # Data line -- may be a continuation line for multi-port files
            raw_data_lines.append(stripped)

    if not option_found:
        # Touchstone spec defaults
        freq_mult = 1.0e9
        data_format = "MA"
        z0 = 50.0

    # Concatenate all numeric tokens from data lines
    all_tokens: list[float] = []
    for dl in raw_data_lines:
        # Remove inline comments (some files use '!' mid-line)
        if "!" in dl:
            dl = dl[: dl.index("!")]
        all_tokens.extend(float(t) for t in dl.split())

    # Each frequency point has: 1 freq value + n_ports*n_ports*2 data values
    values_per_point = 1 + n_ports * n_ports * 2
    n_freq = len(all_tokens) // values_per_point

    if n_freq == 0:
        raise ValueError("No data points found in Touchstone file.")

    freqs: list[float] = []
    s_params: list[np.ndarray] = []

    for i in range(n_freq):
        offset = i * values_per_point
        freq_hz = all_tokens[offset] * freq_mult
        freqs.append(freq_hz)

        s_matrix = np.zeros((n_ports, n_ports), dtype=complex)
        data_offset = offset + 1

        for row in range(n_ports):
            for col in range(n_ports):
                pair_idx = data_offset + (row * n_ports + col) * 2
                v1 = all_tokens[pair_idx]
                v2 = all_tokens[pair_idx + 1]
                s_matrix[row, col] = _values_to_complex(v1, v2, data_format)

        s_params.append(s_matrix)

    return {
        "freqs": freqs,
        "s_params": s_params,
        "n_ports": n_ports,
        "z0": z0,
        "comments": comments,
    }


# ── Far-field CSV import ──


def import_farfield_csv(filepath: str | Path) -> dict[str, Any]:
    """Parse a far-field CSV file with columns theta_deg, phi_deg, gain_dB.

    The file may contain comment lines starting with ``#`` which are captured
    in the returned metadata dictionary.

    Parameters
    ----------
    filepath:
        Path to the CSV file.

    Returns
    -------
    dict
        ``"theta_deg"`` -- list of theta values (degrees).
        ``"phi_deg"`` -- list of phi values (degrees).
        ``"gain_db"`` -- list of gain values (dB).
        ``"metadata"`` -- dict of metadata extracted from header comments.
    """
    filepath = Path(filepath)

    theta_deg: list[float] = []
    phi_deg: list[float] = []
    gain_db: list[float] = []
    metadata: dict[str, str] = {}

    with open(filepath, newline="") as fh:
        # Consume leading comment lines
        lines: list[str] = []
        for raw_line in fh:
            stripped = raw_line.strip()
            if stripped.startswith("#"):
                # Try to extract key=value or key: value metadata
                comment_body = stripped.lstrip("#").strip()
                if ":" in comment_body:
                    key, _, val = comment_body.partition(":")
                    metadata[key.strip()] = val.strip()
                elif "=" in comment_body:
                    key, _, val = comment_body.partition("=")
                    metadata[key.strip()] = val.strip()
                continue
            lines.append(stripped)

        reader = csv.DictReader(lines)
        for row in reader:
            # Support common column-name variations
            theta_key = _find_key(row, ["theta_deg", "theta", "Theta_deg", "Theta"])
            phi_key = _find_key(row, ["phi_deg", "phi", "Phi_deg", "Phi"])
            gain_key = _find_key(row, ["gain_dB", "gain_db", "Gain_dB", "gain", "Gain"])

            theta_deg.append(float(row[theta_key]))
            phi_deg.append(float(row[phi_key]))
            gain_db.append(float(row[gain_key]))

    return {
        "theta_deg": theta_deg,
        "phi_deg": phi_deg,
        "gain_db": gain_db,
        "metadata": metadata,
    }


def _find_key(row: dict[str, Any], candidates: list[str]) -> str:
    """Return the first matching key from *candidates* present in *row*."""
    for key in candidates:
        if key in row:
            return key
    raise KeyError(
        f"None of the expected column names {candidates} found in CSV header. "
        f"Available columns: {list(row.keys())}"
    )
