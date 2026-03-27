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

### Install from PyPI

```bash
pip install apab[ollama]            # array-level tools + Ollama (no C++ deps)
pip install apab[ollama,edgefem]    # + full-wave unit-cell simulation (EdgeFEM)
pip install apab[openai]            # + OpenAI provider
pip install apab[anthropic]         # + Anthropic provider
```

For the default LLM provider, install [Ollama](https://ollama.ai) and pull a model:

```bash
ollama pull qwen2.5-coder:14b
```

### EdgeFEM (optional)

EdgeFEM (full-wave solver) is a C++ extension that requires CMake and Eigen3. Only needed for unit-cell simulation — array pattern and system tools work without it.

- **macOS:** `brew install cmake eigen`
- **Ubuntu/Debian:** `sudo apt-get install cmake libeigen3-dev`
- **Windows:** Install CMake from [cmake.org](https://cmake.org/download/), Eigen3 via vcpkg

Then: `pip install apab[edgefem]`

### Development Installation

```bash
git clone https://github.com/jman4162/agentic-phased-array-builder.git
cd agentic-phased-array-builder
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ollama,edgefem]"
```

## Quick Start — CLI

```bash
# 1. Initialize a project with a ready-to-run config
apab init --name my_array --dir ./my_project --quickstart
cd my_project

# 2. Check your environment (Ollama running? Model pulled?)
apab doctor

# 3. Start an interactive design session
apab design
# Try prompts like:
#   "Design an 8x8 patch array at 28 GHz with Taylor tapering"
#   "Compute the array pattern and evaluate system metrics"

# 4. Or run non-interactively from config
apab run

# 5. Generate a report from a run
apab report <run_id>

# 6. Run as MCP server (for Claude Desktop, etc.)
apab mcp serve
```

For full-wave unit-cell simulation (requires EdgeFEM), use `--quickstart-fullwave` instead of `--quickstart`.

## Quick Start — Python API

```python
from apab import ArraySpec, ScanPoint, PAMPatternEngine

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

This creates an 8x8 element array with half-wavelength spacing at 28 GHz, computes the full 2-D radiation pattern, and returns directivity and sidelobe level. For a complete workflow including unit-cell simulation, mutual coupling, and link budgets, see `examples/06_full_pipeline_case_study.py`.

See `examples/` for more: coupling analysis, trade studies, agent sessions, and Touchstone import.

## Case Study

See `examples/06_full_pipeline_case_study.py` for a complete 28 GHz 5G
phased-array case study with EdgeFEM FEM simulation, array patterns,
mutual coupling, link budget, and trade study. A companion paper is in
`examples/case_study_paper.tex`.

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

## Disclaimer

**This software is provided for educational and research purposes only.**
Phased-array antenna technology may be subject to export control regulations
including the U.S. International Traffic in Arms Regulations (ITAR) and
Export Administration Regulations (EAR). Users are solely responsible for
ensuring that their use of this tool—including any designs, simulations,
or analyses performed—complies with all applicable laws and regulations.

The authors and contributors make no representations regarding the export
control status of any outputs produced by this software and assume no
liability for any misuse or for any violations of export control
regulations arising from its use.

## License

[MIT](LICENSE)

See [SPEC.md](SPEC.md) for the full specification and [CHANGELOG.md](CHANGELOG.md) for release history.
