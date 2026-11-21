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
itself; rtfractal matches its deterministic CPU path.

The legacy figures are ported too (`rtfractal/plots/`, see "Plots"). They
are drawn after each phase and can be redrawn from a finished analysis
folder with `rtfractal plot`.

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
rtfractal analyze CASE --tke uv|legacy                   # 2D TKE spectrum: of (u'^2+v'^2)/2 (default) or u only
rtfractal analyze CASE --spectrum E|P                    # radial spectra: ring-integrated E(k) (default) or legacy P(k)
rtfractal analyze CASE --no-plots                        # data files only; --dpi N (default 300)
rtfractal analyze CASE --allow-missing-fields            # 3D: proceed on a partly reconstructed case

rtfractal dimension FILE                    # one D from a plain segment file: x1 y1 x2 y2 per line
rtfractal dimension FILE --as polyline      # 2 columns of ORDERED vertices instead
rtfractal dimension FILE --as points        # UNORDERED point cloud: 2 columns 2D, 3 columns 3D
rtfractal dimension FILE --as points --dim 3 # take 3 leading columns explicitly
rtfractal dimension FILE --as points --allow-below-spacing   # cross the saturation bound anyway
rtfractal dimension FILE --min-box 1e-4 --max-box 0.05 --factor 1.5   # the ladder (defaults below)
rtfractal dimension FILE --offsets legacy|covering --trim legacy|r2 --backend cpu|cuda
rtfractal dimension FILE --csv ladder.csv   # also write the box-size/count ladder

rtfractal plot CASE/analysis                # redraw every figure from the folder, no recomputation
rtfractal plot CASE                         # same (a case folder means CASE/analysis)
rtfractal plot ANALYSIS --phases 1,3 --output-dir FIGS   # subset; figures elsewhere (same subfolders)
rtfractal plot ANALYSIS --case CASE         # case for field figures if plotdata/ is absent (default:
                                            # rtfractal_run.json, else the folder's parent)
rtfractal plot OLD/analysis --assume-settings legacy     # label D figures of a folder that did not
                                                         # record its box-counting settings
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
from rtfractal.plots import plot_phase1, plot_phase2, plot_phase3, plot_all
plot_all(case, 'out')                   # case may be None for a folder written by rtfractal
```

## Plain segment files

`rtfractal dimension FILE` puts any curve through the same estimator as the
cases: four numeric columns per line are `x1 y1 x2 y2`, two columns with
`--as polyline` are the ordered vertices of one curve. Whitespace, comma,
semicolon and tab all work, one header line is skipped, and `#` or `%`
comments are ignored.

Two columns without `--as` is an error rather than a guess: an ordered
polyline and an unordered point cloud look identical in a file and are
counted differently. `--as polyline` differences the rows into segments;
`--as points` counts occupied cells, which is a different measure with its
own lower bound (next section).

The ladder defaults to `extent/512` to `extent/8` at factor 1.5, and the
chosen values are printed. Three things are warned about, because a bare
file carries none of the context a case does:

- **the ladder outrunning the data.** Below the shortest segments the set is
  a collection of straight lines, which the count reads as `D = 1` and which
  pulls the fit down. Measured on the Minkowski sausage at construction level
  3: `-0.214` against exact on a ladder four steps too fine, `-0.086` with
  the smallest box at the feature size.
- **units.** `offset_fractions` switches on *absolute* box sizes (0.005 and
  0.02), so the number of covering offsets, and therefore the bias, depends
  on whether the file is in metres or in Lennard-Jones sigma. A known issue,
  recorded below and not yet changed.
- **a fallback fit.** When no window reaches the one-decade minimum, the
  reported `D` is a whole-ladder regression rather than a selected window.
  That used to be silent.

## Point clouds

`boxcount/points2d.py` counts a 2D point set, reached through
`fractal.dimension_of_points()` or `rtfractal dimension FILE --as points`.
A point occupies exactly one box, so there is no clipping test as for
segments and no separating-axis test as for triangles: the cell index is
`floor((p - origin) / delta)` and the count is the number of distinct cells.
The CPU path takes the distinct count of the cell keys, costing O(n) memory
and never allocating the grid; the CUDA path marks an occupancy array as the
other two kernels do. Both count the same integer cells, so they agree
exactly, which is asserted.

**A finite point set has box dimension 0.** Below the spacing between the
points every box holds at most one point, the count saturates at `n`, and the
apparent dimension falls to 0. Taken two decades below the spacing the count
is *exactly* `n` at every size, the slope is exactly 0 and the fit returns
`NaN` — which is the right answer, and is a test. So a point ladder has a
hard lower bound that a curve's does not, and the counter **refuses** to
cross the median nearest-neighbour distance rather than warning about it.
`allow_below_spacing=True` (`--allow-below-spacing`) overrides it; `spacing=`
supplies a known value, such as a particle diameter or an extraction stride,
instead of measuring one.

`min_box` defaults to twice the spacing. Nearer than about four times it, the
reported dimension starts to feel the saturation: on a uniform cloud in the
unit square, exact dimension 2, the measured values are

| n | spacing | D at 2x spacing | D at 4x spacing |
|---|---|---|---|
| 20,000 | 0.00328 | 1.8188 | 1.8920 |
| 200,000 | 0.00105 | 1.9239 | 1.9320 |
| 1,000,000 | 0.00047 | 1.9430 | 1.9494 |

(`max_box = extent/8`, factor 1.5), approaching 2 as the sampling densifies,
which is a property of finite sampling rather than of the counter. Note the
large-box end matters too: a space-filling point count is a grid-cell count,
which quantises hard when the box approaches the domain, so `extent/4` reads
about 0.03 lower than `extent/8` throughout. Point samples of the shapes in
`shapes.py` recover their dimensions on the same ladder:

| shape | level | points | exact | measured |
|---|---|---|---|---|
| koch | 8 | 65,537 | 1.2619 | 1.2717 |
| sierpinski | 8 | 9,843 | 1.5850 | 1.5853 |
| minkowski | 6 | 218,793 | 1.5000 | 1.4676 |
| hilbert | 9 | 262,144 | 2.0000 | 1.9337 |
| dragon | 16 | 72,680 | 2.0000 | 1.7745 |

A dense point sample of the Koch curve and the curve itself agree to 0.03,
which is also a test. The plane-filling pair read low for the same
finite-level reason as in `test_shapes.py`; note that the point measure does
markedly better on Hilbert (-0.07) than the segment measure (-0.15).

### In 3D

`boxcount/points3d.py` is the same measure with the same hard bound, reached
through `fractal.dimension_of_points_3d()` or `--as points` on a 3-column
file. For a cloud sampling a *surface* the nearest-neighbour distance is the
surface sampling density, which is the scale that matters; nothing is derived
from the bounding-box volume, which would be wrong by the aspect ratio of a
thin layer.

It recovers an exact answer where one exists. A 3D Cantor dust at level 6 on
a ladder commensurate with its construction (factor 3) gives counts of
exactly 8, 64, 512 and 4096, D = 1.8928 against the exact 3·log2/log3 =
1.8928, at R² = 1.0000.

**That same set shows the ladder factor is not cosmetic.** On factor 1.5 it
reads +0.069 and on factor 2.0 it reads **+0.469**, because a strictly
self-similar set has a staircase box count: it jumps by 8 each time δ crosses
a factor of 3 and is flat between, so an incommensurate ladder aliases.
Physical interfaces are not strictly self-similar and the default 1.5 is fine
for them, but a textbook set wants a commensurate ladder.

Other sets, default ladder:

| set | points | exact | measured |
|---|---|---|---|
| Sierpinski tetrahedron, L7 | 32,770 | 2.0000 | 2.0038 |
| Menger sponge, L5 | 3,200,000 | 2.7268 | 2.7918 |
| regularly sampled plane | 160,000 | 2.0000 | 2.0087 |
| sphere surface | 200,000 | 2.0000 | 1.9402 |
| uniform cube | 300,000 | 3.0000 | 2.8981 |

**On the near-planar case, measure before assuming.** The documented legacy
D = 5 to 9 at R² ≈ 0.8 on a nearly flat early-time surface belongs to the
**legacy offsets**, not to the triangle measure: a flat surface lies below
every shifted origin and the counts come out zero, which is now a test. With
covering offsets the triangle counter handles a flat plane perfectly well,
measured at D = 2.018, R² = 0.9999. So the point counter does not rescue a
broken case; the offset correction already did.

What a point sample adds is a second, independent measure that agrees with
the corrected one where both are valid: 2.009 against 2.018 on a sampled
plane, and exactly 2.0087 for both on the same plane perturbed by 1e-3. They
separate once the surface carries structure near the sampling scale (2.139
against 2.208 at a 1e-2 perturbation), which is expected, because a mesh and
a vertex sample of it are then different sets. Agreement is a check; the two
must not be mixed in one comparison.

## Partly reconstructed 3D cases

A 3D OpenFOAM case writes `interface.vtk` live from a function object but
needs `reconstructPar` for the volume fields, so a case can hold every
interface mesh and only some time directories. The fractal dimension comes
from the mesh and is unaffected; the mixing widths, `Theta`, `Theta_hat` and
the energy budget cannot be computed, and phase 1 catches the read failure
and fills those columns with `NaN`.

`analyze` now stops before doing any work and names the affected time range.
`--allow-missing-fields` proceeds anyway, with valid `D` and `NaN` mixing
over that range. `case.snapshots_without_fields` exposes the list, and
`read_alpha_3d`/`read_U_3d` raise a message naming `reconstructPar` instead
of a bare missing-path error.

## Layout

```
io/        case.py (format/dimension detection, snapshots, physics), openfoam.py, solavof.py, mesh.py,
           segments.py (plain segment/polyline files + ladder diagnostics)
boxcount/  segments2d.py (Liang-Barsky, CPU+CUDA), triangles3d.py (SAT, CPU+CUDA),
           points2d.py, points3d.py (occupied cells + nearest-neighbour bound, CPU+CUDA),
           measures2d.py, measures3d.py (per-box segment counts / triangle areas, CPU+CUDA),
           scaling.py (boundary removal + optimal window, shared by 2D/3D), offsets.py
interface.py  contours -> segments      mixing.py  widths, Theta, energy budget
fractal.py    D of contours, surfaces, projections
multifractal.py  tau(q), D_q, f(alpha)  spectra.py  Dalziel slab, radial, TKE spectra
velocity.py   velocity statistics       shapes.py  Koch, Sierpinski, Minkowski, Hilbert, Dragon
pipeline/     phase1.py, phase2.py, phase3.py                cli.py
plotdata.py   rtfractal_run.json + plotdata/ (the small extra files the plots need)
plots/        common.py (metadata, settings labels, Agg figures), phase1.py, phase2.py, phase3.py
```

## Plots

`rtfractal analyze` draws the legacy figure set after each phase (unless
`--no-plots`), with the legacy file names and subfolders:
- **Phase 1, 2D** (`summary/`): mixing_evolution, mixing_self_similar,
  fractal_dimension_evolution, combined_evolution, D_and_dh_dtau,
  dalziel_integral_measures, interface_snapshots.
- **Phase 1, 3D** (`summary/`): mixing_evolution_3d, selfsimilar_scaling_3d,
  fractal_evolution_3d, dalziel_integrals_3d, surface_complexity_3d.
- **Phase 2, 2D** (`multifractal/`): f_alpha_evolution_overlay,
  Dq_evolution_overlay, D012_vs_tau, multifractality_measures,
  phase_portrait, plus multifractal_dimensions, multifractal_spectrum and
  multifractal_r_squared in each `parameter_<t>`/`time_<t>` folder.
- **Phase 2, 3D** (`multifractal/`): f_alpha_evolution_3d, Dq_evolution_3d,
  D012_vs_tau_3d, multifractality_measures_3d, phase_portrait_D0_h_3d.
- **Phase 3, 2D**: `velocity/` velocity_statistics_evolution,
  velocity_field_snapshots; `spectra/` dalziel_slope_evolution,
  dalziel_spectra_snapshots, spectrum_evolution, spectrum_t<t> (early, mid,
  late). The legacy 3D analyzer draws no phase-3 figures, and neither does
  rtfractal.

The SOLA-VOF and OpenFOAM plotting duplicates of `analyze_rt.py` are merged
into one path. Not ported: dimensions_evolution, multifractal_params_evolution
and Dq_evolution_3D, which some old SOLA-VOF folders hold. They came from
`RTAnalyzer.analyze_multifractal_evolution`, which `analyze_rt.py` never calls.

**Data the plots read.** All figures are drawn from files on disk. Three
small additions to the legacy layout make that possible (`plotdata.py`):
- `analysis/rtfractal_run.json`: the case metadata and, per phase, the
  settings it ran with and the code that ran it: `rtfractal_version`,
  `git_sha` and `git_dirty`. `git_dirty` is true when the package source
  had uncommitted edits. Editable installs pick those up, so the SHA alone
  doesn't identify the code. Check it before mixing outputs from different
  runs.
- `analysis/plotdata/field_t*.npz` and `velocity_t*.npz` (2D): the picture
  fields. They're strided to at most 1000 cells per axis, as float32.
- `analysis/plotdata/radial_spectra_t*.csv` (2D): the binned radial spectra.

A legacy-analyzer folder has none of these. Its field figures are then read
from the case.

**Box-counting settings are on every D, D_q and f(α) figure**, as a title
tag (e.g. `[covering offsets, L/8, R² trim]` or `[LEGACY box counting: ...]`)
and a footnote. They come from `rtfractal_run.json`. Folders without one are
labelled "settings not recorded", unless `--assume-settings` is given; the
label then says "assumed".

**Deliberate differences from the legacy figures** (content, not style):
- **Front labels:** ht is labelled as the bubble (upper) front and hb as the
  spike (lower) front. The legacy OpenFOAM 2D and 3D figures had them swapped.
- **3D degree of multifractality:** labelled max D_q − min D_q, which is
  what both codes compute. Legacy labelled it D₋₅ − D₅.
- **Per-time D(q) figures** also show the raw D_q (before the concavity
  projection). They state when the projection collapsed τ(q) for q ≥ 0.
- **spectrum_t<t>:** shows the spectrum the slope was actually fitted to
  (mean and plane removed), with the actual fit line. Legacy recomputed the
  spectrum without the plane detrend and drew the detrended slope through it.
- **2D radial spectra are drawn in the convention they were computed
  in**, read from the column names.
  - E(k) (default): reference slopes −5/3 and −3.
  - P(k) (`--spectrum P`): the reference slopes are the P(k) equivalents,
    −8/3 and −4.
  - Legacy drew −5/3 and −3 against P(k). The 1D Dalziel slab figure keeps
    −5/3.
- **TKE curves** are labelled with the `--tke` mode: "½(u²+v²) spectrum",
  or "legacy: u only".
- **Vorticity panels** use 21 symmetric levels over ±(98th percentile).
  Legacy's levels spanned the full range despite vmin/vmax.
- **D_and_dh_dtau** omits the two end points of the 3-point running mean,
  which legacy averaged against zero padding.
- **The 2D conservation panels** widen their legacy ±0.002 zoom when the
  data leave it.
- **dalziel_integral_measures** is now drawn for OpenFOAM 2D cases too.

**Reproduced as legacy, still worth knowing:**
- **2D α_RT fits** use τ > 0.3 τ_max with no wall guard (the 3D fit has
  one). The figure now says so.
- **Dalziel's τ = 2 benchmark slopes** are compared with a horizontal
  concentration spectrum taken from a 2D simulation.
- **2D u_rms and v_rms** are taken about the domain mean.
- **The VOF "total spectral energy"** is a trapezoid over log-bin averages.
  Its magnitude has no physical meaning.

## Verification

`pytest` (79 tests; `tests/test_phase2.py` and `test_phase3.py` for phases 2–3, `test_plots.py` for the figures) covers:
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
  - **Fixed (default `--tke uv`):** the 2D TKE spectrum is now the radial
    average of ½(P_u + P_v), the spectrum of TKE per unit mass ½(u′² + v′²).
    Its total_energy is the exact ½(⟨u′²⟩ + ⟨v′²⟩). Legacy (`--tke legacy`)
    took the slope, R² and dominant wavelength from u alone, and its
    total_energy was a trapezoid over binned power, off by orders of
    magnitude.
  - **Fixed (default `--spectrum E`): radial spectra are ring-integrated
    E(k).** This applies to the 2D full-domain spectra and the 3D horizontal
    slab spectra. Legacy (`--spectrum P`, included in `--legacy`) wrote mean
    power per mode, P(k), even under the 3D column names `E_u … E_tke`.
    - In a 2D wavenumber plane E(k) ∝ k·P(k), so E slopes are P slopes + 1.
      Kolmogorov −5/3 and Bolgiano–Obukhov −11/5 refer to E(k).
    - E(k) output names say so:
      - 2D: `<field>_slope_E`, `_R2_E`, `_dom_wavelength_E`.
      - 3D files: `spectrum_E_3d_t*.csv` (column `E`) and
        `tke_spectrum_E_3d_t*.csv`.
      - 3D summaries: `*_slope_E_<band>`.
    - Old scripts reading legacy names fail loudly instead of misreading.
    - `<field>_total_energy` is now the exact variance (TKE for `tke`).
    - The Dalziel slab spectrum is 1D and is unchanged.
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
