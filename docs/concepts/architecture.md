---
description: >-
  APAB's four-layer architecture: agent orchestrator, MCP tool layer,
  domain wrappers, and the artifact/provenance layer.
---

# Architecture

APAB has four layers. The MCP tool layer is the stable surface;
everything above it is replaceable.

```text
┌─────────────────────────────────────────────────┐
│ Agent frontends                                  │
│   APAB orchestrator · Strands agent · LangGraph  │
├─────────────────────────────────────────────────┤
│ MCP tool layer (FastMCP, 17 tools)               │
├─────────────────────────────────────────────────┤
│ Domain wrappers                                  │
│   EdgeFEM · phased-array-modeling ·              │
│   phased-array-systems                           │
├─────────────────────────────────────────────────┤
│ Artifacts + provenance (run bundles)             │
└─────────────────────────────────────────────────┘
```

## Agent frontends

The built-in orchestrator (`apab.agent.orchestrator`) loops an LLM over
the tool schemas: ask, execute tool calls, feed results back, stop on a
text answer or the turn budget. Providers plug in through the
`LLMProvider` protocol — Ollama by default, OpenAI, Anthropic, Gemini,
and OpenAI-compatible endpoints as opt-in extras. The
[Strands adapter](../tutorials/strands-adapter.md) and
[LangGraph pipeline](../tutorials/langgraph-pipeline.md) drive the same
tools from outside.

## MCP tool layer

A first-party FastMCP server registers 17 typed tools:

| Group | Tools |
|---|---|
| Unit cell (EdgeFEM) | `edgefem_run_unit_cell`, `edgefem_surface_impedance`, `edgefem_export_touchstone` |
| Array patterns | `pattern_compute`, `pattern_plot_cuts`, `pattern_plot_3d`, `pattern_multi_beam`, `pattern_null_steer` |
| System | `system_evaluate`, `system_trade_study` |
| Project + I/O | `project_init`, `project_validate`, `io_import_touchstone`, `io_save_hdf5` |
| Plot + external EM | `plot_quicklook`, `emtool_list_adapters`, `emtool_import_results` |

Tools validate inputs with Pydantic-backed JSON schemas and return
JSON-serializable results, so any MCP client — or any LLM with the
schemas in context — can call them.

## Domain wrappers

Thin adapters over the physics packages: **EdgeFEM** (full-wave
unit-cell solver, C++/Eigen), **phased-array-modeling** (array factors,
patterns, impairments), and **phased-array-systems** (link budgets,
scenarios, trade studies). Data follows the
`S[f, scan, pol, i, j]` convention with derived `Z_active` and
`Gamma_active`; storage is HDF5 first, with NPZ caching and Touchstone
export.

## Artifacts and provenance

Every run writes a [run bundle](run-bundles.md): audit log, provenance
manifest, optional OpenTelemetry trace, and the artifact tree. Cache
keys hash the config, geometry, and sweep together with dependency
versions, so identical requests reuse results and changed inputs never
collide.

## Extension points

Three entry-point groups accept third-party plugins without forking:
`apab.llm_providers`, `apab.em_adapters` (HFSS/CST/FEKO importers), and
`apab.compute_backends` (local today; the protocol anticipates cloud
executors). Details in [providers and plugins](providers-plugins.md).
