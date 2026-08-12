# Measurement artifact contract

Simulated data in this ecosystem carries provenance by construction: run
bundles record hashes, solver versions, and configs. Measured data has no
equivalent, and a Touchstone file has no native place to put one. This
contract defines the minimum provenance a measured dataset must carry to
enter any tool in the stack, and the file convention that carries it.

Consumers validate independently — APAB (`compare_sim_measured`),
antenna-cad (`MeasuredSolver`), and opensatcom each implement their own
small check against this document rather than sharing a package.
opensatcom in particular stays dependency-independent by design: it is the
cross-check codebase.

## The sidecar

Every measured dataset `<name>.<ext>` is accompanied by `<name>.meta.yaml`
in the same directory. A loader that does not find the sidecar must refuse
the dataset (or, where a tool has an explicit escape hatch, mark the
result as unprovenanced). Required keys:

```yaml
instrument: "Keysight P5004A, S/N US12345678"   # what took the data
date: "2026-08-11"                               # when (ISO 8601)
calibration_state: "SOLT cal, 85052D kit, valid same-day"
uncertainty: "±0.1 dB |S11| below 20 GHz (cal kit spec)"
operator: "initials or name"
synthetic: false                                 # true = generated, not measured
```

Rules:

- `synthetic: true` marks generated data that exercises the *format*.
  Synthetic fixtures are legitimate for tests and demos; presenting one as
  a measurement is what the flag exists to prevent. Tools must propagate
  the flag into any report they produce.
- Free-text fields (`instrument`, `calibration_state`, `uncertainty`) are
  deliberately prose: the contract requires the operator to state them,
  not that they fit a schema. An honest "uncalibrated bench sweep" is
  compliant; an absent sidecar is not.
- Dates are ISO 8601. No other key is required; extra keys are allowed
  and preserved.

## File formats

- **S-parameters**: Touchstone (`.s1p`, `.s2p`, ... ), read with
  scikit-rf wherever measured data is consumed — no hand-rolled parsers
  for measurement paths. (Hand-rolled readers of *simulated* artifacts a
  producer owns, like the EdgeFEM contract loaders, are a different case:
  those are pinned by producer-owned golden fixtures.)
- **Far-field patterns**: the EdgeFEM full-grid CSV columns
  (`theta_deg, phi_deg, e_norm` or the NTF export shape) are the blessed
  complex-pattern format. The two-cut principal-plane format maps into it
  via the `E cos²φ + H sin²φ` reconstruction documented in opensatcom's
  `edgefem_json_loader`; state in the sidecar when a pattern was
  reconstructed from cuts rather than measured on the full grid.
- **Load-pull contours**: phased-array-systems'
  `LoadPullTable.from_csv` schema
  (`Gamma_real, Gamma_imag, pout_drop_db, pae_drop_pct, ampm_deg`), one
  row per measured load state. This is the seam where bench load-pull
  data enters the PA-array co-design loop.

## The first fixture

Until a bench capture exists, the reference fixture is synthetic and says
so: `patch_28ghz_synthetic.s2p` + `patch_28ghz_synthetic.meta.yaml`
(`synthetic: true`), vendored into each consuming repo's test fixtures.
It is a two-port ideal-resonator S-parameter sweep around 28 GHz whose
values are generated, hand-checkable, and stable. The day a real `.s2p`
is captured, it becomes the fixture and the synthetic one remains as the
format test.

## Out of scope

Instrument control (pyvisa/SCPI) stays out of scope until hardware
exists. This contract governs data at rest, not acquisition.
