"""Command line: rtfractal analyze CASE [options]."""
import argparse
import os
import sys

from . import __version__, backend
from .io.case import open_case
from .pipeline import phase1


def _analyze(a):
    case = open_case(a.case)
    out = os.path.abspath(a.output_dir or os.path.join(case.path, 'analysis'))
    be = backend.resolve(a.backend)
    print(f"rtfractal {__version__}: {case.path}\n  format={case.format} dim={case.dim} "
          f"snapshots={len(case.snapshots)} backend={be}\n  {case.physics.summary()}  H0={case.H0}")
    phases = {int(p) for p in a.phases.split(',')}
    if phases - {1}:
        sys.exit("only phase 1 is ported so far; run the legacy analyzers for phases 2 and 3")
    if case.dim == 2:
        levels = [float(v) for v in a.contour_levels.split(',')]
        phase1.run_2d(case, out, contour_levels=levels, backend=be, offsets=a.offsets)
    else:
        phase1.run_3d(case, out, energy=not a.skip_energy, backend=be, offsets=a.offsets)


def main(argv=None):
    p = argparse.ArgumentParser(prog='rtfractal', description=__doc__)
    p.add_argument('--version', action='version', version=__version__)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('analyze', help='analyse a SOLA-VOF, OpenFOAM or rtlbm case (2D or 3D, auto-detected)')
    a.add_argument('case')
    a.add_argument('--phases', default='1', help='comma-separated phases to run (only 1 so far)')
    a.add_argument('--output-dir', help='default: CASE/analysis')
    a.add_argument('--backend', choices=['auto', 'cpu', 'cuda'], default=None,
                   help='box-counting backend (default auto, or $RTFRACTAL_BACKEND)')
    a.add_argument('--contour-levels', default='0.5', help='2D: comma-separated levels for D')
    a.add_argument('--offsets', choices=['legacy', 'covering'], default='legacy',
                   help='grid-origin offsets: legacy (reproduces FractalParameterAI; biased high) or covering')
    a.add_argument('--skip-energy', action='store_true', help='3D: skip the PE/BPE/KE budget (no U read)')
    a.set_defaults(func=_analyze)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
