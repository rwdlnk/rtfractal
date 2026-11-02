"""Command line: rtfractal analyze CASE [options] | rtfractal plot ANALYSIS_DIR_OR_CASE [options]."""
import argparse
import os
import sys
import traceback

from . import __version__, backend, plotdata
from .io.case import detect, open_case
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
    a.add_argument('--no-plots', action='store_true', help='write the data files only (no figures)')
    a.add_argument('--dpi', type=int, default=None, help='figure resolution (default 300, as legacy)')
    a.set_defaults(func=_analyze)

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
