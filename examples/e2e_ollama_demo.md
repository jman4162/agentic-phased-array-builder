# APAB End-to-End Ollama Validation Demo

Validated on MacBook Pro M3 (32 GB RAM) with Ollama running locally.

**Date:** 2026-03-27
**Model:** qwen2.5-coder:14b
**APAB version:** 0.3.0

---

## Setup

```bash
pip install apab[ollama]           # no C++ deps needed for array-only
ollama pull qwen2.5-coder:14b      # ~8.5 GB download
```

## Step 1: Initialize and Check Environment

```bash
$ apab init --name demo_28ghz --dir ./demo --quickstart
[apab init] Project 'demo_28ghz' created in /path/to/demo
[apab init] Quickstart config: 8x8 array, 15-degree steer (array-only, fast)
[apab init] Next: run 'apab doctor' to check your environment

$ cd demo
$ apab doctor
```

```
                           APAB Environment Check
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Check                     ┃ Status ┃ Details                             ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Python 3.13.9             │   ✓    │                                     │
│ Core dependencies         │   ✓    │ All importable                      │
│ EdgeFEM                   │   !    │ Not installed (optional)             │
│ Ollama package            │   ✓    │ Installed                           │
│ Ollama server             │   ✓    │ Reachable at http://localhost:11434 │
│ Model 'qwen2.5-coder:14b'│   ✓    │ Available                           │
│ Model responds            │   ✓    │ OK (9.1s)                           │
└───────────────────────────┴────────┴─────────────────────────────────────┘

1 warning(s), but all required checks passed. Ready to design.
```

EdgeFEM shows as optional warning (yellow `!`) — array-level tools work without it.

## Step 2: Run Non-Interactive Analysis

```bash
$ apab run
```

```
╭──────────────────────────── APAB Run ────────────────────────────╮
│ demo_28ghz                                                       │
│ Provider: ollama / qwen2.5-coder:14b                             │
│ Prompt: Analyse a phased-array antenna for project 'demo_28ghz'. │
│ Array: 8×8, spacing 5.0mm × 5.0mm, taper 'uniform', steer       │
│ θ=15.0°... Design frequency: 30.0 GHz (half-wave spacing).      │
╰──────────────────────────────────────────────────────────────────╯
```

### Agent Tool Calls

**Turn 1** — Agent calls `pattern_compute`:
```
  → Calling pattern_compute(nx=8, ny=8, dx_m=0.005, dy_m=0.005,
    freq_hz=30e9, phi0=0.0, taper=uniform, theta0=15.0)
  ← pattern_compute: directivity=13.92 dBi, sidelobe_level=-3.79 dB,
    HPBW: 13° (E-plane), 12° (H-plane)
```

**Turn 2** — Agent calls `system_evaluate` (missing `bandwidth_hz`):
```
  → Calling system_evaluate(freq_hz=30e9, nx=8, ny=8, ...)
  ← system_evaluate: ERROR — missing required argument: 'bandwidth_hz'
```

**Turn 3** — Agent self-corrects and retries with `bandwidth_hz`:
```
  → Calling system_evaluate(freq_hz=30e9, nx=8, ny=8, bandwidth_hz=1e9, ...)
  ← system_evaluate: EIRP=40.1 dBW, link_margin=9.1 dB, cost=$6,400
```

**Turn 4** — Agent returns structured summary:

### Final Output

```
╭──────────────────────── Result ────────────────────────╮
│                                                        │
│  Array Pattern Metrics:                                │
│  - Directivity: 13.92 dBi                             │
│  - Sidelobe Level: -3.79 dB                           │
│  - E-plane HPBW: 13.0°                                │
│  - H-plane HPBW: 12.0°                                │
│                                                        │
│  System Evaluation Metrics:                            │
│  - EIRP: 40.1 dBW                                     │
│  - Link Margin: 9.08 dB                               │
│  - Beamwidth: 12.69°                                   │
│  - Total RF Power: 64 W                               │
│  - PA Efficiency: 30%                                  │
│  - Total Cost: $6,400                                  │
│                                                        │
│  Recommendations:                                      │
│  1. Increase directivity with tapering or spacing      │
│  2. Reduce sidelobes with null steering                │
│  3. Optimize PA efficiency                             │
│                                                        │
╰────────────────────────────────────────────────────────╯
```

## Key Observations

1. **Tool calling works** — qwen2.5-coder:14b correctly selects and calls `pattern_compute` and `system_evaluate` with appropriate parameters
2. **Error recovery works** — when the model omits a required parameter (`bandwidth_hz`), it reads the error message and self-corrects on the next turn
3. **Frequency derivation works** — the model uses 30 GHz (derived from 5mm half-wave spacing) instead of guessing an arbitrary frequency
4. **Rich UX works** — spinner during inference, tool call visibility, panel-formatted output
5. **Latency** — total wall time ~60 seconds on M3 Pro for 4 LLM turns + 3 tool calls

## Configuration Used

```yaml
# apab.yaml (generated by: apab init --quickstart)
project:
  name: "demo_28ghz"
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
```
