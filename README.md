# APAB — Agentic Phased Array Builder

[![PyPI version](https://img.shields.io/pypi/v/apab)](https://pypi.org/project/apab/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/apab)](https://pypi.org/project/apab/)
[![Tests](https://github.com/jman4162/agentic-phased-array-builder/actions/workflows/tests.yml/badge.svg)](https://github.com/jman4162/agentic-phased-array-builder/actions/workflows/tests.yml)
[![Lint](https://github.com/jman4162/agentic-phased-array-builder/actions/workflows/lint.yml/badge.svg)](https://github.com/jman4162/agentic-phased-array-builder/actions/workflows/lint.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

LLM-driven phased-array antenna design and analysis via MCP tools.

APAB connects an LLM agent to engineering tools for phased-array antenna design: full-wave unit-cell simulation with mutual coupling (over frequency, scan angle, polarization) propagated into array-level patterns and system-level metrics.

## Features

- **17 MCP tools** — unit-cell simulation (EdgeFEM), array patterns, system-level trades, import/export, plotting
- **Agent orchestrator** — natural-language design sessions with automatic tool dispatch
- **Full pipeline** — unit cell → coupling → pattern → system metrics in one run
- **Trade studies** — DOE sampling with Pareto extraction for multi-objective optimization
- **Offline-first** — default Ollama provider runs fully local; remote providers opt-in
- **Extensible** — plugin entry points for LLM providers, EM adapters, and compute backends

## Installation

Requires Python 3.10+.

```bash
# Clone and install
git clone https://github.com/jman4162/agentic-phased-array-builder.git
cd agentic-phased-array-builder
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ollama]"

# For the default LLM provider (Ollama):
# Install from https://ollama.ai, then:
ollama pull qwen2.5-coder:14b
```

## Quick Start — CLI

```bash
# 1. Initialize a project
apab init --name my_array --dir ./my_project

# 2. Edit apab.yaml to define your array (see Configuration below)

# 3. Run non-interactively from config
apab run --config apab.yaml

# 4. Or run an interactive agent session
apab design --config apab.yaml

# 5. Generate a report
apab report <run_id> --config apab.yaml

# 6. Run as MCP server (for Claude Desktop, etc.)
apab mcp serve --config apab.yaml
```

## Quick Start — Python API

```python
from apab.core.schemas import ArraySpec, ScanPoint
from apab.pattern.wrappers_pam import PAMPatternEngine

spec = ArraySpec(
    size=[8, 8],
    spacing_m=[0.005, 0.005],
    taper="uniform",
    steer=ScanPoint(theta_deg=15, phi_deg=0),
)
engine = PAMPatternEngine()
result = engine.full_pattern(spec, freq_hz=28e9, theta0=0, phi0=0)
print(f"Directivity: {result.directivity_dbi:.2f} dBi")
print(f"Sidelobe level: {result.sidelobe_level_db:.2f} dB")
```

See `examples/` for more: coupling analysis, trade studies, agent sessions, and Touchstone import.

## Configuration

Edit `apab.yaml` to configure your project:

```yaml
project:
  name: my_array
  workspace: ./workspace

llm:
  provider: ollama
  model: qwen2.5-coder:14b
  base_url: http://localhost:11434

unit_cell:
  period_x_mm: 5.0
  period_y_mm: 5.0
  substrate_height_mm: 0.254
  substrate_eps_r: 2.2
  patch_length_mm: 3.0
  patch_width_mm: 3.8

sweep:
  freq_start_ghz: 27.0
  freq_stop_ghz: 29.0
  freq_points: 5
  theta_max_deg: 60
  theta_points: 7
  phi_points: 5

array:
  size: [8, 8]
  spacing_m: [0.005, 0.005]
  taper: taylor
  steer:
    theta_deg: 0
    phi_deg: 0
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Agent Orchestrator (LLM ↔ tool-calling loop)   │
├─────────────────────────────────────────────────┤
│  MCP Tool Layer (17 tools via FastMCP)          │
├─────────────────────────────────────────────────┤
│  Domain Wrappers (PAM, PAS, EdgeFEM, importers) │
├─────────────────────────────────────────────────┤
│  External Libraries (edgefem, phased-array-*)   │
└─────────────────────────────────────────────────┘
```

| Layer | Directory | Purpose |
|-------|-----------|---------|
| Agent | `src/apab/agent/` | LLM providers, tool dispatch, orchestration |
| MCP | `src/apab/mcp/` | First-party MCP server with 17 tools |
| Wrappers | `src/apab/pattern/`, `system/`, `coupling/` | Domain logic bridging tools to libraries |
| Core | `src/apab/core/` | Config, schemas, workspace management |

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `edgefem_run_unit_cell` | Run EdgeFEM unit-cell frequency sweep |
| `edgefem_surface_impedance` | Compute surface impedance at a frequency |
| `edgefem_export_touchstone` | Export S-params to Touchstone file |
| `pattern_compute` | Compute full 2-D array radiation pattern |
| `pattern_plot_cuts` | Generate E/H-plane pattern cut plots |
| `pattern_plot_3d` | Generate 3-D pattern visualization |
| `pattern_multi_beam` | Compute multi-beam pattern |
| `pattern_null_steer` | Compute pattern with null steering |
| `system_evaluate` | Evaluate comms/radar link metrics |
| `system_trade_study` | Run DOE trade study with Pareto extraction |
| `project_init` | Initialize project scaffold |
| `project_validate` | Validate apab.yaml configuration |
| `io_import_touchstone` | Import Touchstone S-parameter file |
| `io_save_hdf5` | Save data to run artifact directory |
| `plot_quicklook` | Generate quick-look summary plot |
| `emtool_list_adapters` | List external EM tool adapters |
| `emtool_import_results` | Import results from external EM tools |

## Development

```bash
# Run all tests
pytest tests/ -v

# Linting and type checking
ruff check src/ tests/
mypy src/apab/

# Run examples
python examples/01_simple_patch_28ghz.py
```

## License

[MIT](LICENSE)

See [SPEC.md](SPEC.md) for the full specification and [CHANGELOG.md](CHANGELOG.md) for release history.
