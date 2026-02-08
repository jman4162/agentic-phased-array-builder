# Agentic Phased Array Builder (APAB)
**SPEC v0.2 (draft)**

APAB is an open-source Python package that connects an LLM (default: **local-first via Ollama**) to a toolbox of **MCP-exposed engineering tools** for phased-array antenna design and analysis. APAB plans and executes workflows end-to-end, including **full-wave unit-cell simulation with mutual coupling** over **frequency, scan angle, and polarization**, then propagates those effects into **array-level patterns** and **system-level metrics**.

> Note: The full-wave solver is **EdgeFEM** (renamed from VectorEM due to copyright concerns).

---

## 1) Goals

### 1.1 Primary goals
1. **Agentic workflow**: Given natural-language requirements, APAB can:
   - propose a design plan,
   - create sweep plans,
   - run simulations/analyses via tools,
   - produce plots + reports + reproducible artifacts.
2. **Coupling-aware modeling**: Capture mutual coupling over:
   - frequency (`f`),
   - scan (`θ, φ` or `kx, ky`),
   - polarization (at least orthogonal basis; optional circular).
3. **Pluggable architecture**:
   - EdgeFEM is the first-class solver backend,
   - commercial EM tools (HFSS/CST/FEKO) supported via open adapters (import first; automation via plugins),
   - multiple LLM providers (local + hosted) via an open provider API.
4. **Local-first + reproducible**:
   - default is offline-capable (Ollama local),
   - every run produces a “run bundle” (config, logs, outputs, provenance).
5. **Future scale-up**:
   - future compatibility with GCP/AWS for cloud-hosted execution and inference,
   - future compatibility for **finite-array** full-wave simulations.

### 1.2 Non-goals (for early versions)
- Not a full CAD suite/GUI editor
- Not a multi-tenant hosted SaaS
- Not fabrication-ready layouts in v0.x

---

## 2) Required Dependencies

APAB MUST integrate with the following packages as dependencies:

- **EdgeFEM** (full-wave unit-cell simulator backend)
  - GitHub: `jman4162/EdgeFEM`
  - PyPI: `edgefem`
- **phased-array-systems** (system-level trades / scenarios)
  - GitHub: `jman4162/phased-array-systems`
- **Phased-Array-Antenna-Model** (array factor / patterns / impairments)
  - GitHub: `jman4162/Phased-Array-Antenna-Model`
  - PyPI package name: `phased-array-modeling`

Additional runtime deps (expected):
- `numpy`, `scipy`, `pydantic`, `pyyaml`, `matplotlib`, `rich`, `diskcache` (or similar)
- `mcp` (MCP Python SDK)
- LLM client packages (optional, provider-specific): `ollama`, `openai`, `anthropic`, `google-genai` (or equivalents)

> APAB should pin conservative minimums and avoid strict upper pins unless required.

---

## 3) LLM Compatibility (local-first, multi-provider)

### 3.1 Default local-first behavior (MUST)
- Default provider: **Ollama**
- Default model: **`qwen2.5-coder:14b`**
- Default mode: **offline-capable**, no outbound network required unless user opts in.

### 3.2 Compatible models/providers (SHOULD)
APAB SHOULD be compatible with:
- **Ollama local** models (default)
- **Kimi K2.5** via:
  - Ollama “cloud” model routes (treated as a remote provider behind the Ollama interface), OR
  - a provider plugin if a direct API is used
- **OpenAI API**, **Anthropic Claude API**, **Google Gemini API** via provider plugins
- “OpenAI-compatible” endpoints (for self-hosted or third-party providers exposing that schema)

### 3.3 Provider abstraction (MUST)
APAB defines a stable provider protocol used by the orchestrator:

```python
class LLMProvider(Protocol):
    name: str

    def chat(self, messages: list[dict], *, tools: list[dict] | None = None,
             tool_choice: str | dict | None = None,
             stream: bool = False,
             extra: dict | None = None) -> dict:
        """Return a provider-normalized response that may include tool calls."""

    def supports_tool_calling(self) -> bool: ...
    def supports_streaming(self) -> bool: ...
````

Provider implementations:

* `OllamaProvider` (default)
* `OpenAIProvider`
* `AnthropicProvider`
* `GeminiProvider`
* `OpenAICompatibleProvider` (configurable base_url + headers)

### 3.4 Provider plugin mechanism (MUST)

Providers SHOULD be discoverable via entry points:

* group: `apab.llm_providers`
* value: module path returning a `LLMProvider` implementation

### 3.5 Data egress controls (MUST)

Because some providers are remote:

* APAB MUST treat remote providers as **opt-in**
* APAB MUST support `llm.redaction_mode` with levels:

  * `none` (local-first default)
  * `metadata_only` (no raw geometry/logs; only summaries)
  * `strict` (never send file contents; only user prompt + high-level outcomes)
* APAB MUST log what content was sent to remote providers (auditability).

---

## 4) User Experience

### 4.1 CLI modes

* `apab init`
  Create scaffold with `apab.yaml`, workspace folders, and examples.
* `apab design`
  Interactive agent session: requirements → plan → execution.
* `apab run`
  Non-interactive run from config; ideal for scripts/CI.
* `apab report`
  Generate Markdown/HTML report from a run bundle.
* `apab mcp serve`
  Run APAB as an MCP server exposing tools/resources.

### 4.2 Python API modes

* Scriptable pipelines for power users and reproducibility.
* Tests call the same public APIs.

---

## 5) Architecture Overview

APAB is split into four cooperating layers:

1. **Agent Orchestrator**

   * Talks to an LLM provider (default: Ollama).
   * Uses tool-calling to request actions from MCP tools.
   * Maintains minimal context; relies on resources/artifacts.

2. **MCP Tool Layer**

   * First-party MCP server exposes:

     * simulation tools (EdgeFEM),
     * array pattern tools (Phased-Array-Antenna-Model),
     * system trade tools (phased-array-systems),
     * export/plot/report tools,
     * optional restricted python execution tool,
     * optional “external EM tool” adapters (HFSS/CST/FEKO).

3. **Execution/Compute Layer**

   * Default: local execution.
   * Future: cloud execution backends (AWS/GCP) via a stable compute backend API.

4. **Artifact + Provenance Layer**

   * Run bundles, caches, and manifests.
   * Optional remote artifact stores (S3/GCS) for cloud workflows.

---

## 6) Repository Layout (recommended)

```
agentic-phased-array-builder/
  pyproject.toml
  README.md
  SPEC.md
  src/apab/
    agent/
      orchestrator.py
      provider_registry.py
      prompts.py
      tool_dispatch.py
    providers/
      ollama.py
      openai.py
      anthropic.py
      gemini.py
      openai_compat.py
    mcp/
      server.py
      tools_edgefem.py
      tools_array.py
      tools_system.py
      tools_io.py
      tools_plot.py
      tools_pyexec.py
      tools_emtool.py
      tools_compute.py         # future cloud execution controls
      resources.py
    core/
      schemas.py
      config.py
      provenance.py
      cache.py
      workspace.py
    coupling/
      sparams.py
      active_impedance.py
      polarization.py
    pattern/
      wrappers_pam.py
      coupled_pattern.py
    system/
      wrappers_pas.py
    emtool/
      base.py
      importers.py
      registry.py
    compute/
      base.py                 # compute backend protocol
      local.py
      registry.py
    report/
      build_report.py
  examples/
  tests/
  docs/
```

---

## 7) Configuration

### 7.1 `apab.yaml` (canonical)

Validated by Pydantic.

#### Example `apab.yaml`

```yaml
project:
  name: "dualpol_ucell_28ghz"
  workspace: "./workspace"

llm:
  provider: "ollama"               # DEFAULT local-first
  model: "qwen2.5-coder:14b"       # DEFAULT
  base_url: "http://localhost:11434"
  redaction_mode: "none"           # default for local providers

# optional remote provider example:
# llm:
#   provider: "openai"
#   model: "gpt-4.1-mini"
#   api_key_env: "OPENAI_API_KEY"
#   redaction_mode: "metadata_only"

mcp:
  mode: "local"
  server_host: "127.0.0.1"
  server_port: 8765

compute:
  backend: "local"                 # DEFAULT
  settings:
    max_workers: 4

solver:
  backend: "edgefem"               # full-wave
  settings:
    mesh_quality: "medium"
    max_ram_gb: 24
    max_runtime_s: 3600

unit_cell:
  lattice:
    type: "rect"
    dx_m: 0.005
    dy_m: 0.005
  geometry:
    kind: "patch"
    params:
      patch_w_m: 0.003
      patch_l_m: 0.003
      substrate_h_m: 0.0005
      er: 2.2
  ports:
    count: 2
    description: "dual-pol feed"
  polarization:
    basis: ["H", "V"]

sweep:
  freq_hz: {start: 26.0e9, stop: 30.0e9, n: 41}
  scan: {theta_deg: [0, 60], phi_deg: [0, 90], n_theta: 13, n_phi: 7}
  polarization: ["H", "V"]

array:
  size: [16, 16]
  spacing_m: [0.005, 0.005]
  taper: "taylor"
  steer: {theta_deg: 45, phi_deg: 0}

outputs:
  export_touchstone: true
  export_hdf5: true
  plots: ["arc_scan", "gain_heatmap", "sparams_summary"]
```

---

## 8) Core Data Model (schemas)

All core objects MUST be serializable (JSON/YAML) and stable across minor versions.

Minimum schemas:

* `ProjectConfig`, `LLMSpec`, `MCPSpec`, `ComputeSpec`
* `UnitCellSpec`, `LatticeSpec`, `SweepSpec`, `ScanPoint`, `PolarizationSpec`
* `SimRequest`, `SimResult`, `CouplingResult`
* `ArraySpec`, `PatternResult`
* `SystemScenarioSpec`
* `RunBundleManifest`

Standardized arrays:

* `S[f, scan, pol, i, j]` complex
* Derived:

  * `Z_active[f, scan, pol]`
  * `Gamma_active[f, scan, pol]`
* Optional:

  * `EEP[f, scan, pol, theta_grid, phi_grid, component]`

Storage formats:

* Primary: HDF5
* Optional: NPZ cache
* Touchstone export: `.sNp`

---

## 9) Solver Integration: EdgeFEM Backend (unit-cell)

### 9.1 Solver protocol (stable)

```python
class FullWaveUnitCellSolver(Protocol):
    def version(self) -> str: ...
    def run_unit_cell(self, request: SimRequest) -> "RunHandle": ...
    def poll(self, run: "RunHandle") -> "RunStatus": ...
    def results(self, run: "RunHandle") -> SimResult: ...
```

EdgeFEM integration MUST be implemented via `EdgeFEMAdapter(FullWaveUnitCellSolver)`.

### 9.2 Required physics capabilities

EdgeFEM backend must support:

* periodic/unit-cell modeling concept (scan via phase progression / Bloch / equivalent)
* multi-port S-parameters across frequency
* scan + polarization sweeps

---

## 10) Future Compatibility: Finite-Array Full-Wave Simulation

### 10.1 Motivation

Unit-cell/periodic simulations are essential for coupling trends, but **finite arrays** exhibit edge effects,
finite truncation, element-to-element non-uniform coupling, and finite ground-plane behaviors.

### 10.2 Finite-array roadmap (MUST design-in now)

APAB MUST reserve API space for finite-array simulations without breaking changes.

#### 10.2.1 New request type

Introduce a second solver request type:

```python
class FiniteArraySimRequest(BaseModel):
    array_size: tuple[int, int]
    element_map: dict            # maps element indices -> geometry/params
    excitations: dict            # amplitude/phase per element (or subarray)
    freq_hz: list[float]
    boundary: dict               # radiation boundary / PML / open region
    outputs: dict                # near-field, far-field, port Z, etc.
```

#### 10.2.2 Finite-array solver protocol (future)

```python
class FullWaveFiniteArraySolver(Protocol):
    def version(self) -> str: ...
    def run_finite_array(self, request: FiniteArraySimRequest) -> "RunHandle": ...
    def poll(self, run: "RunHandle") -> "RunStatus": ...
    def results(self, run: "RunHandle") -> "FiniteArraySimResult": ...
```

#### 10.2.3 Interop with pattern engine

Finite-array outputs SHOULD integrate into pattern computations via:

* embedded element patterns computed from finite array excitations,
* port-based coupling matrices extracted from finite simulations,
* validation/regression comparisons to unit-cell predictions.

---

## 11) Mutual Coupling Computation and Use

v0.1+ MUST support:

* S-parameter matrices over sweep points
* derived **active impedance** and/or **active reflection coefficient** for a given excitation vector
* scan blindness / severe mismatch detection (threshold-based)

Coupled patterns MUST support:

* EEP when available
* coupling-matrix approximation applied to excitations
* fallback to isolated element + array factor

---

## 12) Array Pattern Layer (Phased-Array-Antenna-Model)

APAB MUST use `phased-array-modeling` to compute:

* array factor / steering vectors
* tapers
* pattern cuts and metrics
* impairment hooks (including mutual coupling usage where feasible)

Wrapper:

* `PAMPatternEngine`

---

## 13) System/Trade Layer (phased-array-systems)

APAB MUST use `phased-array-systems` for:

* scenario evaluation (comms/radar)
* trade studies (DOE sampling, Pareto)
* consistent metrics dictionaries and reports

Wrapper:

* `PASSystemEngine`

---

## 14) Commercial Tool Interoperability (HFSS / CST / FEKO)

### 14.1 Design intent

APAB MUST provide an open, stable adapter API for commercial EM tools. Runtime use is OPTIONAL.

### 14.2 Interop modes

A) **Results Import (license-agnostic path)** (MUST in v0.1)

* import Touchstone, far-field exports, and summary tables

B) **Live Automation (license-required path)** (future, plugin-only)

* run sweeps, export artifacts into APAB run bundles

### 14.3 External EM tool adapter contract

```python
class ExternalEMToolAdapter(Protocol):
    name: str
    def capabilities(self) -> dict: ...
    def open_project(self, project_ref: str) -> None: ...
    def run_sweep(self, request: SimRequest) -> "RunHandle": ...
    def export_results(self, run: "RunHandle", out_dir: str) -> dict: ...
```

Plugins SHOULD be separate packages:

* `apab-hfss`, `apab-cst`, `apab-feko`

Discovery via entry points:

* group: `apab.em_adapters`

---

## 15) Cloud Compatibility Roadmap (AWS/GCP)

### 15.1 Design intent

APAB MUST be cloud-friendly without being cloud-required.

### 15.2 Compute backend protocol (MUST design-in now)

```python
class ComputeBackend(Protocol):
    name: str
    def submit(self, task: dict) -> str: ...
    def status(self, task_id: str) -> dict: ...
    def fetch_artifacts(self, task_id: str) -> dict: ...
```

Backends:

* `local` (default)
* future: `aws-*` and `gcp-*` backends (Batch/K8s/VM-based)

Discovery via entry points:

* group: `apab.compute_backends`

### 15.3 Artifact stores (future)

Support local + remote artifact stores:

* local filesystem (default)
* future: S3/GCS-compatible object stores
* manifests store URIs, hashes, and provenance metadata

### 15.4 Cloud-hosted LLM inference (future)

APAB SHOULD support cloud-hosted inference endpoints via:

* provider plugins (OpenAI/Claude/Gemini)
* OpenAI-compatible endpoint plugin
* future: AWS/GCP managed inference endpoints behind a provider plugin

---

## 16) MCP Server Specification (minimum)

Core tools:

* `project.init`, `project.validate`
* `edgefem.run_unit_cell`, `edgefem.get_status`, `edgefem.fetch_results`, `edgefem.export_touchstone`
* `coupling.compute_active_impedance`, `coupling.compute_active_reflection`, `coupling.summary`
* `pattern.compute`, `pattern.plot_cuts`, `pattern.plot_3d`
* `system.evaluate`, `system.trade_study`
* `io.save_hdf5`, `report.build`, `plot.quicklook`

Optional (when plugins installed):

* `emtool.*` for external EM tools
* `compute.*` for cloud execution control
* `pyexec.run` restricted python execution

Resources:

* expose run bundle manifests, configs, logs, results, plots, reports

---

## 17) Run Bundles, Caching, and Provenance

Run bundle structure:

```
workspace/runs/<run_id>/
  manifest.json
  apab.yaml
  logs.txt
  artifacts/
    coupling/
    patterns/
    system/
    emtool/
    plots/
    report/
  cache_keys.json
```

Cache keys MUST include:

* config hash, geometry hash, sweep hash
* EdgeFEM version hash
* dependency versions (phased-array-modeling / phased-array-systems)
* provider name + model name (for reproducibility of agent behavior)

---

## 18) Security Model (non-negotiable)

APAB MUST enforce:

* workspace-only filesystem access by default
* local-first, offline-capable defaults
* explicit opt-in for remote LLM providers and cloud compute backends
* explicit enablement for dangerous tools (shell/arbitrary processes/network)
* audit logs for all tool calls and (if remote) LLM egress

Default policy: **safe mode ON**.

---

## 19) Testing Strategy

Unit tests:

* schema validation and YAML round-trips
* coupling math (active impedance/ARC)
* pattern regressions using phased-array-modeling
* system smoke tests using phased-array-systems
* provider normalization tests (mock responses for each provider)

Integration tests:

* minimal EdgeFEM unit-cell run on reduced sweeps
* end-to-end: unit cell → coupling → pattern → report
* import-only: Touchstone → coupling metrics → pattern → report
* future: finite-array stub pipeline (API stability tests)

---

## 20) Versioning and Roadmap

### v0.2 (this spec)

* Default local-first Ollama + Qwen configuration
* Multi-provider LLM architecture (plugins) with strict egress controls
* EdgeFEM unit-cell coupling pipeline
* External EM import adapters (Touchstone-first)
* API space reserved for finite-array simulation + cloud compute

### v0.3

* First example provider plugins (OpenAI/Claude/Gemini) + OpenAI-compatible
* First example compute backend plugin skeletons (aws/gcp)
* Expanded sweep planning + polarization transforms

### v1.0

* Optimization loops
* Mature finite-array backends (EdgeFEM finite-array or external solvers)
* Robust CI/golden artifacts/regression datasets

---

End of SPEC.md

