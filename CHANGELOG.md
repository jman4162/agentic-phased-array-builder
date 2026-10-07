# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.2] - 2026-10-06

### Added
- `system_evaluate` takes `noise_figure_db` (0 to 30 dB). Without it the
  receiver noise figure was fixed at phased-array-systems' 3 dB default, so
  a request at any other noise figure could only be answered wrong
- Ablation harness: `--tasks-file` for task sets beside `tasks.yaml`
  (references go to `references_<x>.json`); `expect_none_on` marks the
  surfaces on which a task cannot be expressed; the 0.5.1 system-tool
  surface is vendored as the `v051` arm
- `examples/09_xband_beamformer_case_study.py` and
  `examples/xband_qpx0252_sysml/`: an X-band AESA case study on a
  4-channel beamformer that drives its FEMs directly, with a SysML v2
  requirement check through sysml2kit

### Fixed
- The phased-array-systems wrapper converted element spacing to
  wavelengths with c = 3e8 m/s. It now uses 299 792 458 m/s; spacing in
  wavelengths changes by 0.07%, array gain by about 0.006 dB

### Known issues
- With edgefem 1.0.0, `edgefem_run_unit_cell` (and example 06) fail with
  "Could not find matching slave edge for master edge N" while building
  periodic boundaries. The fault is in EdgeFEM's unit-cell meshing
- The plot-tool path issue listed under 0.5.1 is still open

## [0.5.1] - 2026-10-06

### Fixed
- The agent's tool dispatcher called tool functions directly and so did not
  run the argument validation an MCP client gets: allowed values and bounds in the
  tool schemas (e.g. `scan_angle_deg < 90`, `scenario_type` of `"comms"` or
  `"radar"`) were not enforced for APAB's own agent. Calls now go through
  the MCP tool's validating path; invalid arguments return an error the
  model can read, and numeric strings such as `"28e9"` are converted
- Ollama: reasoning models (seen with qwen3.5) can put the whole final
  reply in the `thinking` field and leave the content empty, which ended the
  agent loop with an empty answer. An empty reply now falls back to the
  thinking text
- Ollama: tool calls written as text in Llama 3.x's format
  (`{"name": ..., "parameters": {...}}`), embedded unfenced in prose, or
  with arguments encoded as a JSON string were parsed with no arguments or
  not at all. All three now parse
- `swerling` accepts `"1"` as well as `1`. Models often send integer
  choices as strings, and the allowed-value check rejected them

### Known issues
- `pattern_plot_cuts`, `pattern_plot_3d` and the EdgeFEM export reject
  `..` but accept absolute paths, and resolve relative paths against the
  process working directory instead of the workspace, so a model can write
  files outside the workspace

## [0.5.0] - 2026-10-04

### Added
- `system_evaluate` and `system_trade_study` take radar detection options:
  Pd/Pfa, pulse count and integration, Swerling model, duty cycle, sea,
  ground and rain clutter (`rain_rate_mm_hr`, `antenna_height_m`,
  `target_height_m`, `polarization`), CFAR, and search-timeline inputs.
  Allowed values and bounds appear in the tool schema
- `scan_angle_deg` now reaches comms scenarios; it was accepted and dropped
- `system_trade_study` returns `pareto_objectives`, the columns the Pareto
  front was taken over

### Changed
- Requires `phased-array-systems>=0.14.1` (was `>=0.4`). Radar Pd, SNR and
  margin for the same inputs differ from results computed with 0.4, whose
  radar model reported Pd only; sidelobe metrics also change with the 0.11
  sidelobe fix. 0.14.1 fixes scan loss being subtracted twice in the radar
  equation, which made scanned radar SNR pessimistic by twice the one-way
  scan loss (6.02 dB at 60°)
- `scenario_type` accepts only `"comms"` or `"radar"`. Radar options passed
  with a comms scenario, `clutter_type="rain"` without a rain rate, and a
  partial set of search-timeline inputs now return an error instead of being
  ignored

### Fixed
- `system_trade_study` reported column counts as `n_total` and
  `pareto_count`, and never filtered the Pareto front because it looked for a
  cost column (`cost.total_usd`) that phased-array-systems does not emit. It
  now counts designs and minimizes `cost_usd` against `eirp_dbw` (comms) or
  `snr_margin_db` (radar)
- Trade-study cases that raised (for example, a design space without
  `array.ny`) were counted as feasible when no requirements were given.
  They are now excluded and reported as `n_failed` with `first_error`.
  Example 03 hit this on every case and now varies `array.ny`
- `apab.__version__` (and so `apab --version`) still reported 0.3.0
  through the 0.4.x releases; it now matches the package version, and a
  test keeps the two in sync
- `system_evaluate` returned `inf`/`-inf` metrics (phased-array-systems
  sentinels such as `imd3_dbc` with no nonlinearity modeled), which are not
  valid JSON. They are now `null`, with the original value listed under
  `nonfinite_metrics`

## [0.4.1] - 2026-08-12

### Fixed
- Pin `mcp>=1.26,<2`: mcp 2.0.0 removed `mcp.server.fastmcp`, so a fresh
  install resolved a version APAB cannot import. 0.4.0 installs are
  affected the moment the resolver picks mcp 2.x; upgrade to 0.4.1.

## [0.4.0] - 2026-08-11

### Added
- **Server-side observability** — `apab mcp serve` finally emits spans:
  `create_server` initializes observability (env gate unchanged) and every
  registered tool gets an `apab.tool.<name>` span via a `call_tool`
  override, with the same attributes the agent orchestrator emits. A
  caller's W3C `TRACEPARENT` env var is adopted so client and server sides
  of the stdio transport share one trace; `APAB_TRACE_JSONL` names a span
  file for served processes, which have no run bundle. The Strands adapter
  forwards these across the process boundary, making example 07's tracing
  claim true, and a real (un-mocked) Strands integration test covers the
  loop end to end behind the `integration` marker
- **Measurement artifact contract** (`docs/measurement-contract.md`) — the
  `.meta.yaml` provenance sidecar every measured dataset must carry
  (instrument, date, calibration state, uncertainty, operator, synthetic),
  a `MeasurementProvenance` model, and a synthetic 28 GHz Touchstone
  fixture with hand-checkable values (S11 = -1/9 at exactly 28 GHz)
- **Imported arrays persist** — `io_import_touchstone` and
  `emtool_import_results` no longer discard the parsed S-matrices and
  far-field grids: with `run_id`/`workspace` they write HDF5 into the
  run's `artifacts/emtool/` directory
- **`compare_sim_measured` tool** — |S_ii| dB comparison of a simulated
  Touchstone against a measured one (RMSE, max deviation, worst
  frequency) written as a report artifact; refuses measured data without
  its provenance sidecar and propagates the `synthetic` flag

### Fixed
- `ConsoleSpanExporter` wrote spans to stdout, which corrupts the MCP
  stdio JSON-RPC stream; it now writes to stderr
- Lint workflow green again: environment-dependent typing (FastMCP
  decorator typing varies across mcp releases; Literal signatures in
  newer phased-array-systems) no longer flips mypy errors on and off
  between environments

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
