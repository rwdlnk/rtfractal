# rtfractal

Consolidated replacement for the FractalParameterAI / FractalAnalyzer /
3D-FractalParameterAI analysis path. The shared rules (tools of record,
reference values) are in `../RT_ANALYSIS_TOOLS.md`.

- **Every change must keep the regression tests passing.** They require
  bit-identity with the legacy code, run in-process (`tests/test_regression.py`),
  so they pass the legacy options explicitly (`offsets='legacy', trim='legacy'`).
  A deliberate algorithm change goes behind an option with the legacy
  behaviour as default, until Rod decides otherwise.
- **Decided deviations from legacy (defaults, Rod 2026-10-30):** covering
  grid-origin offsets, largest box L/8 (2D and 3D), and the R² end-point trim
  test. `--legacy` (or `--offsets/--ladder/--trim legacy`) reproduces the old
  analyzers.
- **Kernels exist for both targets.** Each is written once as plain Python and
  compiled for CPU (njit) and CUDA (device functions passed through closures).
  Don't use builtin `min`/`max` with more than one argument inside them.
- **Archived CSVs differ from fresh runs in the 16th significant digit** when
  the numpy/scipy builds differ. Don't chase that.
- **Dev venv:** `.venv` (current numpy/numba). Run `.venv/bin/python -m pytest -q`.
- **Phases 1–3 are ported** (`pipeline/phase{1,2,3}.py`); `tests/test_phase{2,3}.py`
  hold their live bit-identity tests (`pytest` takes about 3 minutes).
- **Plots are ported** (`plots/`, `tests/test_plots.py`). They're drawn
  from files on disk only: the legacy CSVs plus `rtfractal_run.json` and
  `plotdata/` (`plotdata.py`).
  - Keep the plot-only data out of the legacy subfolders. The phase-3
    regression tests compare the full CSV listing of `spectra/` and
    `velocity/`.
  - Every D, D_q or f(α) figure must show its box-counting settings.
