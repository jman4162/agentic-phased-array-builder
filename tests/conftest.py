"""Shared test fixtures for APAB tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Provide a temporary workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def sample_config_dict() -> dict:
    """Return a sample config dict matching SPEC section 7 example."""
    return {
        "project": {
            "name": "dualpol_ucell_28ghz",
            "workspace": "./workspace",
        },
        "llm": {
            "provider": "ollama",
            "model": "qwen2.5-coder:14b",
            "base_url": "http://localhost:11434",
            "redaction_mode": "none",
        },
        "mcp": {
            "mode": "local",
            "server_host": "127.0.0.1",
            "server_port": 8765,
        },
        "compute": {
            "backend": "local",
            "settings": {"max_workers": 4},
        },
        "solver": {
            "backend": "edgefem",
            "settings": {"mesh_quality": "medium", "max_ram_gb": 24, "max_runtime_s": 3600},
        },
        "unit_cell": {
            "lattice": {"type": "rect", "dx_m": 0.005, "dy_m": 0.005},
            "geometry": {
                "kind": "patch",
                "params": {
                    "patch_w_m": 0.003,
                    "patch_l_m": 0.003,
                    "substrate_h_m": 0.0005,
                    "er": 2.2,
                },
            },
            "ports": {"count": 2, "description": "dual-pol feed"},
            "polarization": {"basis": ["H", "V"]},
        },
        "sweep": {
            "freq_hz": {"start": 26.0e9, "stop": 30.0e9, "n": 41},
            "scan": {
                "theta_deg": [0, 60],
                "phi_deg": [0, 90],
                "n_theta": 13,
                "n_phi": 7,
            },
            "polarization": ["H", "V"],
        },
        "array": {
            "size": [16, 16],
            "spacing_m": [0.005, 0.005],
            "taper": "taylor",
            "steer": {"theta_deg": 45, "phi_deg": 0},
        },
        "outputs": {
            "export_touchstone": True,
            "export_hdf5": True,
            "plots": ["arc_scan", "gain_heatmap", "sparams_summary"],
        },
    }


@pytest.fixture
def sample_config_yaml(tmp_path: Path, sample_config_dict: dict) -> Path:
    """Write sample config to a YAML file and return the path."""
    import yaml

    config_path = tmp_path / "apab.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f, default_flow_style=False)
    return config_path
