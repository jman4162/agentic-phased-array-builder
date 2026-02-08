# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-02-07

### Added
- **Agent orchestrator** with LLM tool-calling loop (`apab design`, `apab run`)
- **17 MCP tools** covering unit-cell simulation, array patterns, system analysis, trade studies, I/O, and plotting
- **EdgeFEM integration** for full-wave unit-cell frequency sweeps and surface impedance
- **phased-array-modeling wrapper** (PAMPatternEngine) with full 2-D patterns, multi-beam, null steering, and hardware impairments
- **phased-array-systems wrapper** (PASSystemEngine) with comms/radar link budgets and DOE trade studies with Pareto extraction
- **Active impedance utilities** — reflection coefficient, impedance, scan-blindness detection
- **Touchstone and far-field CSV importers** with flexible format support
- **5 LLM providers** — Ollama (full), OpenAI/Anthropic/Gemini/OpenAI-compatible (stubs)
- **CLI commands**: `init`, `design`, `run`, `report`, `mcp serve`
- **Pydantic v2 configuration** with YAML load/save and full schema validation
- **Workspace management** with run bundles, artifact directories, and caching
- **5 working examples** demonstrating array patterns, coupling, trade studies, agent sessions, and Touchstone import
- **188 passing tests** covering all layers
- **Path traversal protection** via `is_within_workspace()` in all file-writing MCP tools
- **Error handling** in all MCP tool functions with structured error JSON responses
- **Logging** across MCP tools, domain wrappers, and CLI
- **CI/CD** with GitHub Actions for testing (Python 3.10-3.13) and linting (ruff + mypy)

### Fixed
- NumPy 2.x compatibility — polyfill for removed `np.trapz` function
- `pa.compute_directivity` now receives 2D meshgrids instead of 1D arrays

## [0.1.0] - 2024-12-01

### Added
- Initial project scaffold and specification (SPEC.md)
- Core Pydantic schemas and configuration system
- Basic CLI framework
