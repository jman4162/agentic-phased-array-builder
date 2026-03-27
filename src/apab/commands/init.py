"""apab init command — scaffold a new project."""

from __future__ import annotations

import argparse
from pathlib import Path

_EXAMPLE_YAML = """\
project:
  name: "{name}"
  workspace: "./workspace"

llm:
  provider: ollama
  model: qwen2.5-coder:14b
  base_url: http://localhost:11434
  redaction_mode: none

mcp:
  mode: local

compute:
  backend: local

solver:
  backend: edgefem

# Array-level tools (pattern, system) work without unit_cell and sweep.
# Uncomment below for full-wave unit-cell simulation (requires EdgeFEM).
#
# unit_cell:
#   lattice:
#     type: rect
#     dx_m: 0.005
#     dy_m: 0.005
#   geometry:
#     kind: patch
#     params:
#       patch_w_m: 0.003
#       patch_l_m: 0.003
#       substrate_h_m: 0.000508
#       er: 3.5
#
# sweep:
#   freq_hz:
#     start: 26.5e9
#     stop: 29.5e9
#     n: 31
#   scan:
#     theta_deg: [0, 60]
#     phi_deg: [0, 90]
#     n_theta: 13
#     n_phi: 7
#   polarization: [H, V]

array:
  size: [16, 16]
  spacing_m: [0.005, 0.005]
  taper: uniform
  steer:
    theta_deg: 0
    phi_deg: 0

outputs:
  export_touchstone: false
  export_hdf5: true
  plots: []
"""

_QUICKSTART_YAML = """\
project:
  name: "{name}"
  workspace: "./workspace"

llm:
  provider: ollama
  model: qwen2.5-coder:14b
  base_url: http://localhost:11434
  redaction_mode: none

mcp:
  mode: local

compute:
  backend: local

array:
  size: [8, 8]
  spacing_m: [0.005, 0.005]
  taper: uniform
  steer:
    theta_deg: 15
    phi_deg: 0

outputs:
  export_touchstone: false
  export_hdf5: true
  plots: []
"""

_FULLWAVE_YAML = """\
project:
  name: "{name}"
  workspace: "./workspace"

llm:
  provider: ollama
  model: qwen2.5-coder:14b
  base_url: http://localhost:11434
  redaction_mode: none

mcp:
  mode: local

compute:
  backend: local

solver:
  backend: edgefem

unit_cell:
  lattice:
    type: rect
    dx_m: 0.005
    dy_m: 0.005
  geometry:
    kind: patch
    params:
      patch_w_m: 0.003
      patch_l_m: 0.003
      substrate_h_m: 0.000508
      er: 3.5

sweep:
  freq_hz:
    start: 27.0e9
    stop: 29.0e9
    n: 11
  scan:
    theta_deg: [0, 30]
    phi_deg: [0, 90]
    n_theta: 4
    n_phi: 3
  polarization: [H]

array:
  size: [8, 8]
  spacing_m: [0.005, 0.005]
  taper: uniform
  steer:
    theta_deg: 0
    phi_deg: 0

outputs:
  export_touchstone: false
  export_hdf5: true
  plots: []
"""


def cmd_init(args: argparse.Namespace) -> None:
    """Create a new APAB project scaffold."""
    project_dir = Path(args.dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    config_path = project_dir / "apab.yaml"
    if config_path.exists():
        print(f"[apab init] Config already exists: {config_path}")
        return

    if getattr(args, "quickstart_fullwave", False):
        template = _FULLWAVE_YAML
    elif getattr(args, "quickstart", False):
        template = _QUICKSTART_YAML
    else:
        template = _EXAMPLE_YAML
    config_path.write_text(template.format(name=args.name))

    # Create workspace directories
    workspace_dir = project_dir / "workspace"
    for subdir in ["runs", "cache", "logs"]:
        (workspace_dir / subdir).mkdir(parents=True, exist_ok=True)

    print(f"[apab init] Project '{args.name}' created in {project_dir.resolve()}")
    print(f"[apab init] Config: {config_path}")
    print(f"[apab init] Workspace: {workspace_dir}")
    if getattr(args, "quickstart_fullwave", False):
        print(
            "[apab init] Full-wave config: 8x8 array, 27-29 GHz "
            "(requires EdgeFEM: pip install apab[edgefem])"
        )
    elif getattr(args, "quickstart", False):
        print(
            "[apab init] Quickstart config: 8x8 array, "
            "15-degree steer (array-only, fast)"
        )
    print("[apab init] Next: run 'apab doctor' to check your environment")
