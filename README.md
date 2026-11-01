# rtfractal

Fractal and mixing analysis of Rayleigh–Taylor interfaces in 2D and 3D. It
reads SOLA-VOF output, OpenFOAM cases and rtlbm output.

`rtfractal` consolidates the analysis path of three repositories:
- `FractalParameterAI` (`analyze_rt.py`, `analyze_rt_3d.py`)
- `FractalAnalyzer` (2D box counting)
- `3D-FractalParameterAI` (`fractal3d`)

It carries only the code those analyzers execute. It runs on a CUDA GPU or
on numba CPU threads, with identical results.

**Status (v0.1.0).** All three analysis phases are ported:
- **Phase 1:** mixing widths and fractal dimension, for every snapshot.
- **Phase 2:** multifractal spectrum, at selected times.
- **Phase 3:** power spectra and velocity statistics, at selected times.

With `--legacy`, each phase is bit-identical to the legacy code. The one
exception is the legacy 3D multifractal GPU path, which isn't reproducible
itself; rtfractal matches its deterministic CPU path. The plots are not
ported yet; use the legacy analyzers for those.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # add [test] for the test suite, [vtk] for binary/XML meshes
```

## Use

```bash
rtfractal analyze CASE                      # format and 2D/3D are auto-detected; phases 1,2,3
rtfractal analyze CASE --phases 1           # or any subset; phase 2 alone reads summary/temporal_results.csv
rtfractal analyze CASE --mf-times 4,10,16   # phase 2 (and 2D phase 3) target times, default 2,4,...,18
rtfractal analyze CASE --spec-times 10,16   # 3D phase 3 target times, default 2,4,...,20
rtfractal analyze CASE --backend cpu        # or cuda; default auto (also $RTFRACTAL_BACKEND)
rtfractal analyze CASE --output-dir OUT     # default CASE/analysis
rtfractal analyze CASE --contour-levels 0.1,0.5,0.9     # 2D
rtfractal analyze CASE --skip-energy                     # 3D: don't read U
rtfractal analyze CASE --legacy                          # reproduce FractalParameterAI numbers exactly
rtfractal analyze CASE --offsets legacy|covering --ladder legacy|L8 --trim legacy|r2   # individually
```

The output layout and the columns of `analysis/summary/temporal_results.csv`
match the legacy analyzers. The numbers differ by default, because three
inherited problems are fixed (see "Known issues"):
- covering grid-origin offsets
- a largest box of L/8, instead of L/2 in 2D and min(L, W)/2 in 3D
- the R² end-point trim test

Pass `--legacy` to reproduce the legacy analyzers bit for bit.

From Python:
```python
from rtfractal import open_case
from rtfractal.pipeline import phase1, phase2, phase3
case = open_case('path/to/case')        # case.dim, case.physics, case.snapshots, case.read_2d(...)
df = phase1.run_2d(case, 'out')         # or phase1.run_3d
phase2.run_2d(case, 'out', df=df)       # multifractal; or phase2.run_3d
phase3.run_2d(case, 'out')              # spectra + velocity; or phase3.run_3d
```

## Layout

```
io/        case.py (format/dimension detection, snapshots, physics), openfoam.py, solavof.py, mesh.py
boxcount/  segments2d.py (Liang-Barsky, CPU+CUDA), triangles3d.py (SAT, CPU+CUDA),
           measures2d.py, measures3d.py (per-box segment counts / triangle areas, CPU+CUDA),
           scaling.py (boundary removal + optimal window, shared by 2D/3D), offsets.py
interface.py  contours -> segments      mixing.py  widths, Theta, energy budget
fractal.py    D of contours, surfaces, projections
multifractal.py  tau(q), D_q, f(alpha)  spectra.py  Dalziel slab, radial, TKE spectra
velocity.py   velocity statistics       pipeline/phase1.py, phase2.py, phase3.py      cli.py
```

## Verification

`pytest` (71 tests; `tests/test_phase2.py` and `test_phase3.py` for phases 2–3) covers:
- **Unit tests:** exact power laws, a Koch curve, flat planes, CPU = CUDA
  counts, reader round trips.
- **Regression tests:** each runs the legacy code in-process and requires
  bit-identical D and box counts. They cover LBM 2D and SOLA-VOF 160×200 at
  t = 3, 10 and 16 s, and LBM 3D 80×100×40 at t = 3, 6, 10 and 16 s. They're
  skipped if the legacy repos or the data are absent.

The full phase-1 CSV for the LBM 2D reference case is bit-identical in all
32 columns.

**Comparing against archived CSVs.** Expect differences in the 16th
significant digit whenever the numpy or scipy build differs from the one
that wrote the CSV. Compare to about 1e-12, or rerun the legacy code
side by side.

## Known issues inherited from the legacy algorithm

Items 1, 2 and 4 are fixed by default (decided 2026-10-30). `--legacy`
reproduces all of them.

1. **Grid-origin offsets don't cover the set** (`boxcount/offsets.py`, `--offsets`).
   **Fixed: the default is now `covering`; `legacy` reproduces the old behaviour.**
   - **What happens:** at each box size the grid origin is moved to
     `min + f·δ` (f = 0 … 0.75), and the minimum count over origins is kept.
     Parts of the interface in `[min, min + f·δ)` are never counted, so
     shifted grids undercount, and that inflates D.
   - **Koch curve (true D 1.262):** FractalAnalyzer's own generator and
     counter give 1.35–1.39 with offsets and 1.26–1.29 without. Through the RT
     phase-1 pipeline the legacy offsets give 1.52–1.62, against 1.30 with
     covering offsets (`min − f·δ`).
   - **Straight line (true D 1.000):** 1.12–1.19 with legacy offsets, against
     0.98–0.99 with covering offsets.
   - **RT cases, mean D over τ > 2:** SOLA-VOF 160×200 drops from 1.732 to
     1.647, and LBM 160×200 from 1.648 to 1.534.
   - **3D:** a nearly flat early-time surface lies entirely below every
     shifted origin, which gives zero counts.
   - **Production study** (all 2D/3D Dalziel cases, mean D over τ > 2;
     `RT-Simulations/Dalziel_1999/offset_study/SUMMARY.md`): with covering
     offsets 2D D at 1920×2400 is about 1.69 (max box L/2) or 1.71 (L/8),
     against 1.76 legacy; 3D D is about 2.61 against 2.76. D3D − D2D stays
     close to +1 (+0.94 against +0.98).
2. **The 2D boundary-removal test compares r, not R², against 0.95**
   (`boxcount/scaling.py`, `--trim`). **Fixed: the default is now `r2`.** In 2D r ≈ −1, so the test always trims the first
   and last box size whenever the ladder has more than 8 sizes. In 3D r ≈ +1,
   so the test works as intended.
3. **Offset thresholds are absolute lengths** (0.005 m and 0.02 m), not
   fractions of the domain. Not changed.
4. **The largest box was L/2** (`--ladder`). The validation paper states
   extent/8. **Fixed: the default is now `L8`.** At L/2 even smooth curves
   and surfaces give D biased high. `projected_dimension` (a 3D diagnostic)
   still uses L/2. It always used covering-style offsets, in the legacy code
   too, so legacy D_proj values are not biased.
5. **The legacy multifractal analyzers use the biased offsets too**
   (`FractalParameterAI/analysis/multifractal_analyzer.py`,
   `3D-FractalParameterAI/fractal3d/multifractal_3d.py`: origin `min + f·δ`).
   **Fixed in rtfractal phase 2:** it uses the same offsets, ladder and trim
   options as phase 1, and its D0 equals the phase-1 D at the same snapshot.

## Phase 2 and 3 notes

The legacy behaviour below is reproduced under the defaults, apart from the
offsets, ladder and trim. These are things to know before quoting values.

- **Multifractal (phase 2):**
  - **The convexity projection often collapses D_q to D0 for q ≥ 0.**
    Compare `Dq` with `Dq_raw` before calling an interface monofractal.
  - **The negative-q side is unstable.** α width roughly doubles under the
    L/8 defaults, because the scaling windows are shorter.
  - **2D and 3D define degree of multifractality differently.** 2D uses
    D(−5) − D(5), which is often negative; 3D uses max − min.
  - **D0 reuses the phase-1 scaling window.** rtfractal recomputes it
    in-process rather than reading it back from disk.
  - **Which legacy `--legacy` reproduces:** the current legacy code
    (FractalParameterAI `869103d`, 3D-FractalParameterAI `0d47341`) run live,
    not older archived output. Legacy multifractal output written before
    2026-03-22 predates two legacy commits that changed the computation:
    `75a7763` (D_q monotonicity, scale sharing) and `0978034`/`7597f1f`
    (domain-based scales, boundary removal). It therefore has different D0
    and D0−D2, and a mismatch against it is expected. At 1920×2400 the
    SOLA-VOF folders hold both generations side by side: `time_*` from
    2026-03-16 and `parameter_*` from 2026-05-27.
  - **Screening D_q:** a projected D0−D2 of exactly 0 is an artefact of the
    concavity projection, not a measurement. There are three ways to get one:
    - the projection collapses τ(q) to a line for all q ≥ 0
      (`convexity_collapsed`);
    - τ(q) is flattened only over [0, 2], which the flag misses;
    - the projection fails and leaves τ non-concave.

    `RT-Simulations/Dalziel_1999/offset_study/phase2_production/screen.py`
    classifies each snapshot.
- **CUDA kernels:**
  - **Phase 2 kernels avoid fused multiply-adds**, so CPU and CUDA agree bit
    for bit.
  - **Phase 1 CUDA kernels still use fused multiply-adds.** They can assign
    an element lying exactly on a box edge differently from the CPU.
- **Spectra and velocity (phase 3):**
  - **The 2D "TKE spectrum" slope and dominant wavelength are u's alone.**
    Only total_energy averages u and v.
  - **2D turbulence_intensity divides by the domain-mean velocity**, which is
    near zero, so the values are meaningless.
  - **2D u_rms and v_rms are taken about the domain mean**, not about the
    horizontal mean at each height as in 3D.
  - **2D radial spectra:** the slope is fitted over the middle half of the
    bins, and total_energy is a trapezoid over log-bin averages.
  - **End-ramp padding repeats the first sample at the periodic wrap.**
  - **3D band fits are unweighted and have no 6·dx resolution flag.** On
    coarse grids both bands lie beyond the resolved range.
  - **3D phase 3 reads the time directory named exactly**, not the nearest
    snapshot, and skips missing times.

## Origin

Created 2026-10-30 from:
- FractalParameterAI `main` @ 869103d
- 3D-FractalParameterAI `main` @ 0d47341
- FractalAnalyzer `main` @ 0e28f78

Fresh history. The AI/ML components of FractalParameterAI were deliberately
left out.
