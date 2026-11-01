"""Command line: rtfractal analyze CASE [options]."""
import argparse
import os
import sys

from . import __version__, backend
from .io.case import open_case
from .pipeline import phase1, phase2, phase3


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
        if 2 in phases:
            print("\nPhase 2: multifractal spectrum")
            phase2.run_2d(case, out, times=mf_times, df=df, **box)    # df=None: reads summary/temporal_results.csv
        if 3 in phases:
            print("\nPhase 3: spectra + velocity statistics")
            phase3.run_2d(case, out, times=mf_times, tke=a.tke, spectrum=a.spectrum)   # legacy 2D: --mf-times
    else:
        if 1 in phases:
            print("\nPhase 1: mixing + fractal dimension")
            phase1.run_3d(case, out, energy=not a.skip_energy, **box)
        if 2 in phases:
            print("\nPhase 2: multifractal spectrum")
            phase2.run_3d(case, out, times=mf_times, **box)
        if 3 in phases:
            print("\nPhase 3: spectra + velocity statistics")
            phase3.run_3d(case, out, times=[float(t) for t in a.spec_times.split(',')], spectrum=a.spectrum)


def main(argv=None):
    p = argparse.ArgumentParser(prog='rtfractal', description=__doc__)
    p.add_argument('--version', action='version', version=__version__)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('analyze', help='analyse a SOLA-VOF, OpenFOAM or rtlbm case (2D or 3D, auto-detected)')
    a.add_argument('case')
    a.add_argument('--phases', default='1,2,3',
                   help='comma-separated phases: 1 mixing + D, 2 multifractal, 3 spectra + velocity (default all)')
    a.add_argument('--mf-times', default='2,4,6,8,10,12,14,16,18',
                   help='target times [s] for phase 2 (and phase 3 in 2D); nearest snapshot is used')
    a.add_argument('--spec-times', default='2,4,6,8,10,12,14,16,18,20',
                   help='3D: target times [s] for phase 3 (time directory named exactly)')
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
    a.set_defaults(func=_analyze)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
