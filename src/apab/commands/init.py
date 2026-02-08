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


def cmd_init(args: argparse.Namespace) -> None:
    """Create a new APAB project scaffold."""
    project_dir = Path(args.dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    config_path = project_dir / "apab.yaml"
    if config_path.exists():
        print(f"[apab init] Config already exists: {config_path}")
        return

    config_path.write_text(_EXAMPLE_YAML.format(name=args.name))

    # Create workspace directories
    workspace_dir = project_dir / "workspace"
    for subdir in ["runs", "cache", "logs"]:
        (workspace_dir / subdir).mkdir(parents=True, exist_ok=True)

    print(f"[apab init] Project '{args.name}' created in {project_dir.resolve()}")
    print(f"[apab init] Config: {config_path}")
    print(f"[apab init] Workspace: {workspace_dir}")
    print("[apab init] Edit apab.yaml to configure your array, then run: apab design")
