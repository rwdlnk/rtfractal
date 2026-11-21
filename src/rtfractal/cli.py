"""Command line: rtfractal analyze CASE | rtfractal dimension FILE | rtfractal plot ANALYSIS_DIR_OR_CASE."""
import argparse
import os
import sys
import traceback

from . import __version__, backend, plotdata
from .io.case import detect, open_case
from .io import segments as segio
from .pipeline import phase1, phase2, phase3


def _plot(phase, case, out, a):
    """Figures of one phase after it ran; a plotting failure never stops the analysis."""
    if a.no_plots:
        return
    from . import plots
    fn = {1: plots.plot_phase1, 2: plots.plot_phase2, 3: plots.plot_phase3}[phase]
    try:
        fn(case, out, dpi=a.dpi)
    except Exception:
        print(f"WARNING: phase {phase} plots failed (the data files are written):")
        traceback.print_exc()


def _analyze(a):
    if a.legacy:
        a.offsets, a.ladder, a.trim, a.tke, a.spectrum = 'legacy', 'legacy', 'legacy', 'legacy', 'P'
    case = open_case(a.case)
    out = os.path.abspath(a.output_dir or os.path.join(case.path, 'analysis'))
    be = backend.resolve(a.backend)
    print(f"rtfractal {__version__}: {case.path}\n  format={case.format} dim={case.dim} "
          f"snapshots={len(case.snapshots)} backend={be}\n  {case.physics.summary()}  H0={case.H0}\n"
          f"  box counting: offsets={a.offsets} ladder={a.ladder} trim={a.trim}  tke spectrum: {a.tke}  radial spectra: {a.spectrum}(k)")
    gap = getattr(case, 'snapshots_without_fields', [])
    if gap:
        ts = [g.t for g in gap]
        print(f"\n  !! {len(gap)} of {len(case.snapshots)} snapshots have an interface mesh but NO\n"
              f"  !! reconstructed time directory, t = {min(ts):g} to {max(ts):g} s.\n"
              f"  !! Fractal dimensions there are fine, they come from the mesh. Mixing widths,\n"
              f"  !! Theta, Theta_hat and the energy budget would be NaN for those {len(gap)} rows.\n"
              f"  !! Run reconstructPar over that range first if you need them.")
        if not a.allow_missing_fields:
            sys.exit("  Stopping before doing any work. Pass --allow-missing-fields to analyse "
                     "anyway (valid D, NaN mixing, over that range).")
    phases = {int(p) for p in a.phases.split(',')}
    if phases - {1, 2, 3}:
        sys.exit("--phases takes a comma-separated subset of 1,2,3")
    mf_times = [float(t) for t in a.mf_times.split(',')]
    box = dict(backend=be, offsets=a.offsets, ladder=a.ladder, trim=a.trim)
    df = None
    if case.dim == 2:
        if 1 in phases:
            levels = [float(v) for v in a.contour_levels.split(',')]
            print("\nPhase 1: mixing + fractal dimension")
            df = phase1.run_2d(case, out, contour_levels=levels, **box)
            _plot(1, case, out, a)
        if 2 in phases:
            print("\nPhase 2: multifractal spectrum")
            phase2.run_2d(case, out, times=mf_times, df=df, **box)    # df=None: reads summary/temporal_results.csv
            _plot(2, case, out, a)
        if 3 in phases:
            print("\nPhase 3: spectra + velocity statistics")
            phase3.run_2d(case, out, times=mf_times, tke=a.tke, spectrum=a.spectrum)   # legacy 2D: --mf-times
            _plot(3, case, out, a)
    else:
        if 1 in phases:
            print("\nPhase 1: mixing + fractal dimension")
            phase1.run_3d(case, out, energy=not a.skip_energy, **box)
            _plot(1, case, out, a)
        if 2 in phases:
            print("\nPhase 2: multifractal spectrum")
            phase2.run_3d(case, out, times=mf_times, **box)
            _plot(2, case, out, a)
        if 3 in phases:
            print("\nPhase 3: spectra + velocity statistics")
            phase3.run_3d(case, out, times=[float(t) for t in a.spec_times.split(',')], spectrum=a.spectrum)
            _plot(3, case, out, a)


def _try_open(path):
    try:
        return open_case(path) if path and os.path.isdir(path) and detect(path)[0] else None
    except Exception as e:
        print(f"  (could not open case {path}: {e})")
        return None


def _plot_cmd(a):
    """rtfractal plot: redraw figures from an existing analysis folder."""
    from . import plots
    path = os.path.abspath(a.path)
    phases = {int(p) for p in a.phases.split(',')}
    if phases - {1, 2, 3}:
        sys.exit("--phases takes a comma-separated subset of 1,2,3")
    if detect(path)[0]:                                   # a case folder
        case, analysis_dir = _try_open(path), os.path.join(path, 'analysis')
    else:
        analysis_dir = path
        run = plotdata.read_run(path) or {}
        case = None
        cands = [a.case] if a.case else [(run.get('case') or {}).get('path'), os.path.dirname(path)]
        for cand in cands:
            case = _try_open(os.path.abspath(cand)) if cand else None
            if case is not None:
                break
        if a.case and case is None:
            sys.exit(f"--case {a.case}: not a readable case")
    if not os.path.isdir(analysis_dir):
        sys.exit(f"{analysis_dir}: no such analysis folder")
    out = os.path.abspath(a.output_dir) if a.output_dir else analysis_dir
    print(f"rtfractal {__version__} plot: {analysis_dir}\n  case: {case.path if case else '(none)'}\n  output: {out}")
    files = plots.plot_all(case, analysis_dir, phases=phases, out_dir=out, dpi=a.dpi,
                           assume_settings=a.assume_settings)
    print(f"{len(files)} figures written")


def _dimension_points(a):
    """One box-counting dimension from a 2D point set (x y per line)."""
    import numpy as np

    from . import fractal
    from .boxcount import points2d, scaling

    try:
        pts = segio.read_points(a.file)
        d = segio.describe_points(pts)
    except (ValueError, OSError) as e:
        sys.exit(f'rtfractal dimension: {e}')
    be = backend.resolve(a.backend)
    min_box = a.min_box if a.min_box is not None else 2.0 * d['spacing']
    max_box = a.max_box if a.max_box is not None else d['extent'] / 8

    print(f"rtfractal {__version__}: {os.path.abspath(a.file)}")
    print(f"  {d['n']} points  x [{d['x_min']:.6g}, {d['x_max']:.6g}]  "
          f"y [{d['y_min']:.6g}, {d['y_max']:.6g}]  extent {d['extent']:.6g}")
    print(f"  nearest-neighbour spacing: median {d['spacing']:.4g}  "
          f"mean {d['spacing_mean']:.4g}  min {d['spacing_min']:.4g}")
    print(f"  ladder: min_box={min_box:.6g}"
          f"{'' if a.min_box is not None else ' (2x spacing, default)'} "
          f"max_box={max_box:.6g}{'' if a.max_box is not None else ' (extent/8, default)'} "
          f"factor={a.factor}  offsets={a.offsets} trim={a.trim} backend={be}")
    if d['extent'] > 1.0 or d['extent'] < 0.01:
        print(f"  WARNING: extent is {d['extent']:.4g}. The covering-offset count switches on "
              f"absolute box sizes (0.005, 0.02), so results are not scale invariant; compare "
              f"only with runs at a similar extent, or rescale to match.")
    if min_box < 4.0 * d['spacing']:
        print(f"  NOTE: min_box is {min_box / d['spacing']:.1f}x the spacing. A point set "
              f"saturates as the box approaches the spacing, where every box holds one point "
              f"and D falls toward 0. Measured on a uniform cloud, D reads about 0.1 low at "
              f"2x spacing and is best near 4x.")

    try:
        res = fractal.dimension_of_points(pts, min_box, max_box, a.factor, be, a.offsets,
                                          a.trim, d['spacing'], a.allow_below_spacing)
        sizes, counts, _ = points2d.box_counting_2d_points(
            pts, min_box, max_box, a.factor, be, a.offsets, d['spacing'], a.allow_below_spacing)
    except ValueError as e:
        sys.exit(f'rtfractal dimension: {e}')

    fit = res['box_sizes']
    fit_span = np.log10(fit.max() / fit.min()) if len(fit) > 1 else 0.0
    span = np.log10(sizes.max() / sizes.min())
    print(f"\n  D = {res['dimension']:.4f} +/- {res['error']:.4f}   R^2 = {res['r_squared']:.5f}")
    print(f"  ladder {len(sizes)} sizes spanning {span:.2f} decades; "
          f"fitted window {len(fit)} sizes [{fit.min():.4g}, {fit.max():.4g}]")
    if np.isnan(res['dimension']):
        print("  D is NaN: no window reached the R^2 and span thresholds")
    elif fit_span < scaling.SPAN_MIN:
        print(f"  WARNING: the fitted window spans {fit_span:.2f} decades, under the "
              f"{scaling.SPAN_MIN:.0f}-decade minimum, so no window qualified and this D is a "
              f"whole-ladder regression, not a selected window. Widen the ladder.")

    if a.csv:
        import pandas as pd
        pd.DataFrame({'box_size': sizes, 'count': counts}).to_csv(a.csv, index=False)
        print(f"  ladder written to {a.csv}")


def _dimension(a):
    """One box-counting dimension from a plain segment or polyline file."""
    import numpy as np

    from . import fractal
    from .boxcount import scaling, segments2d

    if a.as_kind == 'points':
        return _dimension_points(a)
    try:
        segs = segio.read_segments(a.file, kind=a.as_kind)
    except (ValueError, NotImplementedError, OSError) as e:
        sys.exit(f'rtfractal dimension: {e}')      # an input problem, not a crash
    d = segio.describe(segs)
    be = backend.resolve(a.backend)

    min_box, warn = segio.suggest_min_box(segs)
    if a.min_box is not None:
        min_box, warn = a.min_box, [w for w in warn if 'smallest box' not in w]
        if min_box < d['len_p10']:
            warn.append('smallest box %.4g is below the 10th-percentile segment length %.4g: '
                        "the ladder reaches below the data's own resolution and D will read low."
                        % (min_box, d['len_p10']))
    max_box = a.max_box if a.max_box is not None else d['extent'] / 8

    print(f"rtfractal {__version__}: {os.path.abspath(a.file)}")
    print(f"  {d['n']} segments  x [{d['x_min']:.6g}, {d['x_max']:.6g}]  "
          f"y [{d['y_min']:.6g}, {d['y_max']:.6g}]  extent {d['extent']:.6g}")
    print(f"  segment length: min {d['len_min']:.4g}  p10 {d['len_p10']:.4g}  "
          f"median {d['len_median']:.4g}  max {d['len_max']:.4g}")
    print(f"  ladder: min_box={min_box:.6g}{'' if a.min_box is not None else ' (extent/512, default)'} "
          f"max_box={max_box:.6g}{'' if a.max_box is not None else ' (extent/8, default)'} "
          f"factor={a.factor}  offsets={a.offsets} trim={a.trim} backend={be}")
    for w in warn:
        print(f"  WARNING: {w}")

    sizes, counts, bbox = segments2d.box_counting_2d(segs, min_box, max_box, a.factor, be, a.offsets)
    if len(sizes) < 2:
        sys.exit(f"only {len(sizes)} usable box sizes: widen the ladder")
    res = fractal.dimension_of_segments(segs, min_box, max_box, a.factor, be, a.offsets, a.trim)
    span = np.log10(sizes.max() / sizes.min())
    fit = res['box_sizes']
    fit_span = np.log10(fit.max() / fit.min()) if len(fit) > 1 else 0.0
    print(f"\n  D = {res['dimension']:.4f} +/- {res['error']:.4f}   R^2 = {res['r_squared']:.5f}")
    print(f"  ladder {len(sizes)} sizes spanning {span:.2f} decades; "
          f"fitted window {len(res['box_sizes'])} sizes "
          f"[{res['box_sizes'].min():.4g}, {res['box_sizes'].max():.4g}]")
    if np.isnan(res['dimension']):
        print("  D is NaN: no window reached the R^2 and span thresholds")
    elif fit_span < scaling.SPAN_MIN:
        print(f"  WARNING: the fitted window spans {fit_span:.2f} decades, under the "
              f"{scaling.SPAN_MIN:.0f}-decade minimum, so no window qualified and this D is a "
              f"whole-ladder regression, not a selected window. Widen the ladder.")

    if a.csv:
        import pandas as pd
        pd.DataFrame({'box_size': sizes, 'count': counts}).to_csv(a.csv, index=False)
        print(f"  ladder written to {a.csv}")


def main(argv=None):
    p = argparse.ArgumentParser(prog='rtfractal', description=__doc__)
    p.add_argument('--version', action='version', version=__version__)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('analyze', help='analyse a SOLA-VOF, OpenFOAM or rtlbm case (2D or 3D, auto-detected)')
    a.add_argument('case')
    a.add_argument('--phases', default='1,2,3',
                   help='comma-separated phases: 1 mixing + D, 2 multifractal, 3 spectra + velocity (default all)')
    a.add_argument('--mf-times', default='2,4,6,8,10,12,14,16,18',
                   help='target times in SECONDS (simulation time, not tau) for phase 2 (and phase 3 in 2D); nearest snapshot is used. '
                        'The default suits the Dalziel runs; set it per case elsewhere (dropped targets warn)')
    a.add_argument('--spec-times', default='2,4,6,8,10,12,14,16,18,20',
                   help='3D: target times in SECONDS (not tau) for phase 3 (time directory named exactly)')
    a.add_argument('--output-dir', help='default: CASE/analysis')
    a.add_argument('--backend', choices=['auto', 'cpu', 'cuda'], default=None,
                   help='box-counting backend (default auto, or $RTFRACTAL_BACKEND)')
    a.add_argument('--contour-levels', default='0.5', help='2D: comma-separated levels for D')
    a.add_argument('--offsets', choices=['legacy', 'covering'], default='covering',
                   help='grid-origin offsets: covering (default) or legacy (reproduces FractalParameterAI; biased high)')
    a.add_argument('--ladder', choices=['L8', 'legacy'], default='L8',
                   help='largest box: L8 = L/8 (default) or legacy = L/2 (2D), min(L,W)/2 (3D)')
    a.add_argument('--trim', choices=['r2', 'legacy'], default='r2',
                   help='end-point trim test: r2 (default) or legacy (2D always trims both ends)')
    a.add_argument('--tke', choices=['uv', 'legacy'], default='uv',
                   help="2D TKE spectrum: uv = spectrum of (u'^2+v'^2)/2 (default) or legacy (u spectrum only)")
    a.add_argument('--spectrum', choices=['E', 'P'], default='E',
                   help='radial spectra: E = ring-integrated E(k) (default; Kolmogorov -5/3 applies) or '
                        'P = legacy mean power per mode (slopes one steeper)')
    a.add_argument('--legacy', action='store_true',
                   help='reproduce FractalParameterAI exactly: --offsets/--ladder/--trim/--tke legacy --spectrum P')
    a.add_argument('--skip-energy', action='store_true', help='3D: skip the PE/BPE/KE budget (no U read)')
    a.add_argument('--allow-missing-fields', action='store_true',
                   help='3D: proceed when some snapshots have an interface mesh but no reconstructed '
                        'time directory. Their fractal dimensions are valid; their mixing widths, '
                        'Theta and energy budget come out NaN. Without this the run stops and says so.')
    a.add_argument('--no-plots', action='store_true', help='write the data files only (no figures)')
    a.add_argument('--dpi', type=int, default=None, help='figure resolution (default 300, as legacy)')
    a.set_defaults(func=_analyze)

    d = sub.add_parser('dimension', help='box-counting dimension of a plain segment or polyline file')
    d.add_argument('file', help='text/CSV, 4 columns "x1 y1 x2 y2" per line, or 2 columns of ordered '
                               'polyline vertices with --as polyline')
    d.add_argument('--as', dest='as_kind', choices=['auto', 'segments', 'polyline', 'points'],
                   default='auto',
                   help="how to read the columns: auto (default; 4 -> segments, 2 -> error saying which), "
                        "segments, polyline (ordered vertices of one curve), or points (an unordered "
                        "2D cloud, counted by occupied cell)")
    d.add_argument('--min-box', type=float, default=None,
                   help='smallest box (default extent/512; a value below the 10th-percentile segment '
                        'length is warned about, since the ladder then reaches below the data)')
    d.add_argument('--max-box', type=float, default=None, help='largest box (default extent/8)')
    d.add_argument('--factor', type=float, default=1.5, help='ladder ratio between box sizes (default 1.5)')
    d.add_argument('--offsets', choices=['legacy', 'covering'], default='covering',
                   help='grid-origin offsets: covering (default) or legacy (biased high)')
    d.add_argument('--trim', choices=['r2', 'legacy'], default='r2', help='end-point trim test (default r2)')
    d.add_argument('--backend', choices=['auto', 'cpu', 'cuda'], default=None, help='default auto')
    d.add_argument('--allow-below-spacing', action='store_true',
                   help='--as points: allow a smallest box below the nearest-neighbour spacing, '
                        'where the count saturates and D collapses toward 0. Refused otherwise.')
    d.add_argument('--csv', help='also write the box-size/count ladder here')
    d.set_defaults(func=_dimension)

    pl = sub.add_parser('plot', help='redraw the figures from an existing analysis folder (no recomputation)')
    pl.add_argument('path', metavar='ANALYSIS_DIR_OR_CASE',
                    help='an analysis folder, or a case folder (then CASE/analysis is used)')
    pl.add_argument('--phases', default='1,2,3', help='comma-separated phases to plot (default all available)')
    pl.add_argument('--case', help='the case the analysis folder belongs to (default: rtfractal_run.json, '
                                   'else the parent folder); only needed for figures whose fields '
                                   'were not saved in plotdata/')
    pl.add_argument('--output-dir', help='write the figures here (same subfolders) instead of into the analysis folder')
    pl.add_argument('--dpi', type=int, default=None, help='figure resolution (default 300, as legacy)')
    pl.add_argument('--assume-settings', choices=['legacy', 'default'], default=None,
                    help='label D / D_q figures with these box-counting settings when the folder did not '
                         'record them (e.g. legacy for legacy-analyzer output); marked "assumed"')
    pl.set_defaults(func=_plot_cmd)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
