"""Shared plumbing for the phase plots.

Figures are drawn on matplotlib.figure.Figure objects rendered by the Agg
canvas (no pyplot, so no global backend switch and no figure registry to
leak); each figure is saved and dropped.

Meta collects what the plots need to know about the run, from (in order of
preference) analysis/rtfractal_run.json, an open case, or the CSVs alone.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .. import plotdata

DPI = 300                      # legacy savefig dpi
SETTINGS_PRESETS = {
    'default': {'offsets': 'covering', 'ladder': 'L8', 'trim': 'r2'},
    'legacy': {'offsets': 'legacy', 'ladder': 'legacy', 'trim': 'legacy'},
}


@dataclass
class Meta:
    analysis_dir: str
    dim: Optional[int] = None
    format: Optional[str] = None
    grid: Optional[list] = None
    H: Optional[float] = None
    H0: Optional[float] = None
    case: object = None
    phases: dict = field(default_factory=dict)      # phase number (str) -> settings dict
    assumed: Optional[str] = None                   # preset name used when settings were not recorded

    @property
    def res_label(self):
        return 'x'.join(str(n) for n in self.grid) if self.grid else ''

    @property
    def format_label(self):
        return {'solavof': 'SOLA-VOF', 'openfoam': 'OpenFOAM/rtlbm'}.get(self.format, '')

    @property
    def run_label(self):
        """'160x200 SOLA-VOF' (legacy title suffix), or '' if unknown."""
        return ' '.join(s for s in (self.res_label, self.format_label) if s)

    def settings(self, phase):
        """(settings dict or None, assumed?)."""
        s = self.phases.get(str(phase))
        if s and all(k in s for k in ('offsets', 'ladder', 'trim')):
            return s, False
        if self.assumed:
            return SETTINGS_PRESETS[self.assumed], True
        return None, False


def load_meta(analysis_dir, case=None, assume_settings=None):
    """Meta for an analysis folder.  case: an open case or None.
    assume_settings: 'legacy' | 'default' | None, used to label box-counting
    settings for folders that did not record them (e.g. legacy-analyzer output)."""
    if assume_settings not in (None, 'legacy', 'default'):
        raise ValueError("assume_settings must be 'legacy', 'default' or None")
    m = Meta(analysis_dir=analysis_dir, case=case, assumed=assume_settings)
    run = plotdata.read_run(analysis_dir)
    if run:
        c = run.get('case', {})
        m.dim, m.format, m.grid, m.H0 = c.get('dim'), c.get('format'), c.get('grid'), c.get('H0')
        m.H = (c.get('physics') or {}).get('H')
        m.phases = run.get('phases', {})
    if case is not None:
        cm = plotdata.case_meta(case)
        m.dim, m.format, m.grid, m.H0, m.H = cm['dim'], cm['format'], cm['grid'], cm['H0'], cm['physics']['H']
    if m.dim is None:
        df = read_temporal(analysis_dir)
        if df is not None:
            m.dim = 3 if 'n_triangles' in df.columns else 2
        elif glob_any(os.path.join(analysis_dir, 'multifractal'), suffix='_multifractal_spectrum_3d.csv'):
            m.dim = 3
        else:
            m.dim = 2
    if m.H is None:
        m.H = _infer_H(analysis_dir)
    return m


def glob_any(d, prefix='', suffix=''):
    return os.path.isdir(d) and any(f.startswith(prefix) and f.endswith(suffix) for f in os.listdir(d))


def _infer_H(analysis_dir):
    """H from h / (h/H) in temporal_results.csv."""
    df = read_temporal(analysis_dir)
    if df is None:
        return None
    for c in ('h_total_int', 'h_total_stat', 'h_total_geo', 'h_11'):
        if c in df.columns and c + '_nondim' in df.columns:
            r = (df[c] / df[c + '_nondim']).replace([np.inf, -np.inf], np.nan).dropna()
            if len(r):
                return float(r.median())
    return None


def read_temporal(analysis_dir):
    p = os.path.join(analysis_dir, 'summary', 'temporal_results.csv')
    return pd.read_csv(p) if os.path.isfile(p) else None


# ─── box-counting settings labels ────────────────────────────────────────────

def _ladder_text(ladder, dim):
    if ladder == 'L8':
        return 'max box L/8'
    return 'max box L/2' if dim == 2 else 'max box min(L,W)/2'


def settings_tag(meta, phase):
    """Short title tag, e.g. '[covering offsets, L/8, R² trim]'."""
    s, assumed = meta.settings(phase)
    if s is None:
        return '[box-counting settings not recorded]'
    o, l, t = s['offsets'], s['ladder'], s['trim']
    if (o, l, t) == ('legacy', 'legacy', 'legacy'):
        tag = 'LEGACY box counting: legacy offsets, ' + ('L/2' if meta.dim == 2 else 'min(L,W)/2') + ', r trim'
    else:
        tag = (f"{'covering' if o == 'covering' else 'LEGACY'} offsets, "
               f"{'L/8' if l == 'L8' else ('L/2' if meta.dim == 2 else 'min(L,W)/2')}, "
               f"{'R²' if t == 'r2' else 'legacy r'} trim")
    return f"[{tag}{', assumed' if assumed else ''}]"


def settings_note(meta, phase):
    """One-line footnote with the full settings."""
    s, assumed = meta.settings(phase)
    if s is None:
        return ('Box-counting settings not recorded for this folder (no rtfractal_run.json): '
                'D values may come from legacy (biased-high) or corrected settings.')
    o, l, t = s['offsets'], s['ladder'], s['trim']
    kind = ('rtfractal default' if (o, l, t) == ('covering', 'L8', 'r2') else
            'legacy analyzers, biased high' if (o, l, t) == ('legacy', 'legacy', 'legacy') else 'mixed settings')
    txt = (f"Box counting: offsets={o}, {_ladder_text(l, meta.dim)}, trim={'R²' if t == 'r2' else 'legacy (r)'}"
           f" ({kind})")
    if assumed:
        txt += ' -- ASSUMED (not recorded in the folder)'
    return txt


def annotate_settings(fig, meta, phase):
    fig.text(0.995, 0.003, settings_note(meta, phase), ha='right', va='bottom', fontsize=7, color='0.35')


# ─── figure helpers ──────────────────────────────────────────────────────────

def new_figure(figsize, nrows=1, ncols=1, **kw):
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    axes = fig.subplots(nrows, ncols, **kw)
    return fig, axes


class Saver:
    """Saves figures under out_dir/<subdir>/<name> and keeps the list."""

    def __init__(self, out_dir, dpi=DPI, log=print):
        self.out_dir, self.dpi, self.log, self.written = out_dir, dpi, log, []

    def __call__(self, fig, subdir, name, tight=True, bbox_tight=False, notes_bottom=False):
        d = os.path.join(self.out_dir, subdir) if subdir else self.out_dir
        os.makedirs(d, exist_ok=True)
        if tight:
            # leave room at the bottom for the settings footnote
            fig.tight_layout(rect=(0, 0.025, 1, 1) if notes_bottom else None)
        path = os.path.join(d, name)
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight' if bbox_tight else None)
        fig.clf()
        self.written.append(path)
        return path


def combined_legend(ax1, ax2, **kw):
    l1, b1 = ax1.get_legend_handles_labels()
    l2, b2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, b1 + b2, **kw)


def time_colors(times, cmap_name='viridis'):
    """Legacy colouring: viridis over (t - t_min)/(t_max - t_min)."""
    import matplotlib
    cmap = matplotlib.colormaps[cmap_name]
    t_min, t_max = min(times), max(times)
    return [cmap((t - t_min) / (t_max - t_min)) if t_max > t_min else cmap(0.5) for t in times]
