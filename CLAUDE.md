# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

APAB (Agentic Phased Array Builder) is a Python package that connects an LLM to MCP-exposed engineering tools for phased-array antenna design and analysis. It plans and executes workflows end-to-end: full-wave unit-cell simulation with mutual coupling (over frequency, scan angle, polarization) propagated into array-level patterns and system-level metrics.

The full-wave solver is **EdgeFEM** (not VectorEM).

## Build & Development Commands

```bash
# Install in development mode (once pyproject.toml exists)
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_file.py::test_name -v

# Type checking
mypy src/apab/

# Linting
ruff check src/ tests/

# CLI entry points
apab init          # scaffold project
apab design        # interactive agent session
apab run           # non-interactive from config
apab report        # generate report from run bundle
apab mcp serve     # run as MCP server
```

## Architecture (Four Layers)

1. **Agent Orchestrator** (`src/apab/agent/`) - Talks to LLM providers via a `LLMProvider` protocol. Uses tool-calling to dispatch actions to MCP tools. Default: Ollama with `qwen2.5-coder:14b`, offline-capable.

2. **MCP Tool Layer** (`src/apab/mcp/`) - First-party MCP server exposing tools for simulation (EdgeFEM), array patterns (phased-array-modeling), system trades (phased-array-systems), export/plot/report, and optional external EM adapters.

3. **Execution/Compute Layer** (`src/apab/compute/`) - Local execution by default. `ComputeBackend` protocol designed for future cloud backends (AWS/GCP) via entry points (`apab.compute_backends`).

4. **Artifact + Provenance Layer** - Run bundles at `workspace/runs/<run_id>/` containing manifest, config, logs, and artifacts. Cache keys include config/geometry/sweep hashes and dependency versions.

## Key Required Dependencies

- **edgefem** (PyPI) - Full-wave unit-cell solver backend (`jman4162/EdgeFEM`)
- **phased-array-modeling** (PyPI) - Array factor/patterns/impairments (`jman4162/Phased-Array-Antenna-Model`)
- **phased-array-systems** (GitHub: `jman4162/phased-array-systems`) - System-level trades/scenarios

## Plugin/Extension Points (Entry Points)

| Group | Purpose |
|---|---|
| `apab.llm_providers` | LLM provider plugins (Ollama, OpenAI, Anthropic, Gemini, OpenAI-compatible) |
| `apab.em_adapters` | External EM tool adapters (HFSS, CST, FEKO) |
| `apab.compute_backends` | Compute backend plugins (local, future AWS/GCP) |

## Key Protocols

- `LLMProvider` - Provider abstraction in `src/apab/providers/`. Implementations: `OllamaProvider`, `OpenAIProvider`, `AnthropicProvider`, `GeminiProvider`, `OpenAICompatibleProvider`.
- `FullWaveUnitCellSolver` - Solver protocol. EdgeFEM integrated via `EdgeFEMAdapter`.
- `FullWaveFiniteArraySolver` - Reserved for future finite-array simulations.
- `ExternalEMToolAdapter` - For commercial EM tools (HFSS/CST/FEKO).
- `ComputeBackend` - For local and future cloud execution.

## Configuration

Project config lives in `apab.yaml`, validated by Pydantic. Key sections: `project`, `llm`, `mcp`, `compute`, `solver`, `unit_cell`, `sweep`, `array`, `outputs`. See SPEC.md section 7 for full example.

## Data Model

Core data arrays use the shape convention `S[f, scan, pol, i, j]` (complex). Derived quantities: `Z_active[f, scan, pol]`, `Gamma_active[f, scan, pol]`. Storage: HDF5 (primary), NPZ (cache), Touchstone (`.sNp` export). All schemas must be JSON/YAML-serializable.

## Security Constraints

- Workspace-only filesystem access by default
- Local-first, offline-capable defaults (safe mode ON)
- Remote LLM providers and cloud compute require explicit opt-in
- `llm.redaction_mode` levels: `none` (local default), `metadata_only`, `strict`
- All tool calls and remote LLM egress must be audit-logged
