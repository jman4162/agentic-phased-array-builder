# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-07-09

### Added
- **OpenTelemetry observability** (`apab[observability]`) — spans for every session, turn, LLM call, and tool call (`apab.session > apab.turn > apab.llm.chat / apab.tool.<name>`) with token, latency, and cost attributes; per-run `trace.jsonl`; console and OTLP HTTP exporters; redaction-aware capture modes. See `docs/observability.md`
- **Runtime provenance** — every `run_to_completion` now writes `manifest.json` (config hash, dependency versions, status, token usage, trace ID) alongside `audit.json`; audit entries carry `trace_id`/`span_id`
- **Strands Agents adapter** (`apab[strands]`) — `apab.adapters.strands` exposes APAB's MCP tools to a Strands agent over stdio; example 07
- **Deterministic LangGraph pipeline** (`apab[langgraph]`) — `apab.adapters.langgraph_pipeline` runs validate → pattern → system → constraints → plots → report with SQLite checkpointing; example 08
- **Jaeger trace lab** — one-container `lab/docker-compose.yml` plus walkthrough for viewing agent traces
- **Golden-task eval harness** — `evals/run_evals.py` scores runs from their bundles (tool sequence, status, call budget, metric thresholds); LLM-free scorer tests
- **Real OpenAI, Anthropic, and Gemini providers** — full implementations with per-call `ProviderUsage` (tokens, latency, cost estimate) shared across all five providers, including Ollama
- **Agent-loop events** — `run_to_completion(on_event=...)` callback now drives the `apab run`/`apab design` rendering; the CLI no longer duplicates the loop
- **Prose quality checks** — `scripts/slopcheck.sh` (slopscore-lint + slopless) and an advisory prose CI workflow

### Fixed
- `build_manifest` crashed on optional config sections that serialize as `None`
- **`apab doctor` command** — environment health checks (Python, deps, EdgeFEM, Ollama server/model/ping) with rich table output
- **Rich interactive UX** — `apab design` and `apab run` now show spinner during inference, tool call names and results as they happen, and panel-formatted responses
- **Pre-flight provider check** — `design` and `run` commands verify LLM connectivity before starting, with actionable error messages
- **Ollama connection resilience** — 30-second timeout, `OllamaConnectionError` with clear message, `ping()` method for health checks
- **System prompt tool listing** — agent prompt now includes grouped tool names (by category) so smaller models can reliably select tools
- **`--quickstart` flag** — `apab init --quickstart` generates array-only config (fast, no EdgeFEM needed); `--quickstart-fullwave` for full-wave config
- **JSON fallback test coverage** — `_parse_tool_calls_from_text` and `_strip_json_blocks` fully tested (11 new tests)
- **Public API surface** — `from apab import ArraySpec, PAMPatternEngine` now works via lazy re-exports in `__init__.py`
- **OpenAI-compatible provider** — full implementation delegating to `OpenAIProvider`, enabling vLLM, LM Studio, Together.ai, and other OpenAI-compatible endpoints
- **CONTRIBUTING.md** — guide for adding LLM providers, EM adapters, and compute backends via the plugin entry point system
- **274 passing tests** (up from 188)

### Changed
- **EdgeFEM now optional** — moved from core dependency to `pip install apab[edgefem]`; array pattern and system tools work without it, eliminating C++ build requirement for most users
- **System prompt improved** — replaced contradictory "Do NOT write JSON" instruction with honest acknowledgment of the fallback parser; added error recovery guidance
- **README overhaul** — quickstart now includes `apab doctor`, EdgeFEM documented as optional, installation simplified
- **Development status** upgraded from Alpha to Beta in PyPI classifiers

## [0.2.0] - 2025-02-07

### Added
- **Full pipeline case study** (`examples/06_full_pipeline_case_study.py`) with EdgeFEM FEM simulation, array patterns, mutual coupling, link budget, and trade study
- **Companion LaTeX paper** (`examples/case_study_paper.tex`) documenting the 28 GHz phased-array design methodology
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
