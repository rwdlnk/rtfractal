"""Figures of the legacy analyzers, redrawn from an analysis folder.

    from rtfractal.plots import plot_phase1, plot_phase2, plot_phase3, plot_all
    plot_phase1(case, 'CASE/analysis')     # case may be None if the folder has rtfractal_run.json
                                           # and plotdata/ (written by rtfractal analyze)

Same file names and subfolders as the legacy analyzers; see the phase
modules for the figure list and the deliberate differences.  Rendering
uses the Agg canvas only (no pyplot).
"""
from .common import load_meta
from .phase1 import plot_phase1
from .phase2 import plot_phase2
from .phase3 import plot_phase3

__all__ = ['plot_phase1', 'plot_phase2', 'plot_phase3', 'plot_all', 'load_meta']


def plot_all(case, analysis_dir, phases=(1, 2, 3), out_dir=None, dpi=None, log=print, assume_settings=None):
    """Draw every available figure of the given phases -> list of files."""
    meta = load_meta(analysis_dir, case, assume_settings)
    fns = {1: plot_phase1, 2: plot_phase2, 3: plot_phase3}
    written = []
    for p in sorted(phases):
        written += fns[p](case, analysis_dir, out_dir=out_dir, dpi=dpi, log=log, meta=meta)
    return written
