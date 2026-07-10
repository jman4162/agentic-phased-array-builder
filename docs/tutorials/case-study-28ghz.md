---
description: >-
  Full 28 GHz phased-array case study: EdgeFEM unit-cell simulation,
  coupling-aware patterns, link budget, and a system trade study.
---

# 28 GHz case study

`examples/06_full_pipeline_case_study.py` runs the whole chain for a
5G mmWave scenario: unit-cell full-wave simulation, coupling-aware 8x8
array patterns, a 200 m link budget at 400 MHz bandwidth, and a
40-point trade study over array size and transmit power. **EdgeFEM is
required** for this one:

```bash
pip install "apab[edgefem]"    # needs CMake + Eigen3 (see README)
python examples/06_full_pipeline_case_study.py
```

## What the pipeline computes

1. **Unit cell (EdgeFEM)** — a grounded patch on a Rogers-class
   substrate, swept 26–30 GHz with Floquet ports. EdgeFEM excites the
   cell with a plane wave, so for a grounded patch the reflection
   magnitude stays near one across the band and **resonance appears as
   a reflection-phase transition**, not a magnitude dip. The example
   detects resonance from the maximum phase derivative.
2. **Feed-port model (analytical)** — return loss and impedance
   bandwidth come from a cavity model of the probe-fed patch. The two
   models cross-validate: FEM phase resonance at 28.50 GHz vs the
   analytical 28.40 GHz, under 0.4 % apart.
3. **Scan behavior** — Floquet reflection versus scan angle
   characterizes the element in its array environment, feeding the
   coupling-aware pattern computation.
4. **Array pattern** — 8x8 with taper, steered cuts, directivity and
   sidelobe level.
5. **System trade study** — Latin-hypercube sampling over array size
   (4–16 per axis) and per-element power (10–500 mW) against a 5G NR
   comms scenario, with Pareto extraction.

The run writes six figures to `examples/output/` (reflection phase,
feed-port S11, scan behavior, link budget, trade study, array layout).

## Why the hybrid modeling approach

Floquet-port simulation answers array-environment questions (scan
impedance, phase response, blindness onset) that a feed-port model
cannot; the analytical feed model answers match-bandwidth questions
that a plane-wave excitation cannot. The case study keeps both and
checks one against the other — the same split you would use with a
commercial solver before committing to a fabricated design.

## Companion materials

The repository includes a LaTeX write-up of this study
(`examples/case_study_paper.tex`) and the agent-driven variant of the
same workflow in the [quickstart](../quickstart.md), where the LLM
sequences these steps from a natural-language request instead of a
script.
