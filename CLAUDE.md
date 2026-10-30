# rtfractal

Consolidated replacement for the FractalParameterAI / FractalAnalyzer /
3D-FractalParameterAI analysis path. The shared rules (tools of record,
reference values) are in `../RT_ANALYSIS_TOOLS.md`.

- **Every change must keep the regression tests passing.** They require
  bit-identity with the legacy code, run in-process (`tests/test_regression.py`).
  A deliberate algorithm change goes behind an option (like `--offsets`) with
  the legacy behaviour as default, until Rod decides otherwise.
- **Kernels exist for both targets.** Each is written once as plain Python and
  compiled for CPU (njit) and CUDA (device functions passed through closures).
  Don't use builtin `min`/`max` with more than one argument inside them.
- **Archived CSVs differ from fresh runs in the 16th significant digit** when
  the numpy/scipy builds differ. Don't chase that.
- **Dev venv:** `.venv` (current numpy/numba). Run `.venv/bin/python -m pytest -q`.
- **Not ported yet:** phase 2 (multifractal), phase 3 (spectra, velocity),
  phase-1 plots.
