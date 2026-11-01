"""Phase 2 figures (analysis/multifractal/), drawn from the phase-2 CSVs.

2D (legacy analyze_rt.py _plot_phase2, plus the per-time figures legacy
RTAnalyzer.compute_multifractal_spectrum drew into each time directory):
    f_alpha_evolution_overlay.png, Dq_evolution_overlay.png, D012_vs_tau.png,
    multifractality_measures.png, phase_portrait.png,
    <parameter_|time_><t>/multifractal_dimensions.png, multifractal_spectrum.png,
    multifractal_r_squared.png
3D (legacy analyze_rt_3d.py _plot_phase2):
    f_alpha_evolution_3d.png, Dq_evolution_3d.png, D012_vs_tau_3d.png,
    multifractality_measures_3d.png, phase_portrait_D0_h_3d.png

Deliberate differences from legacy (content, not style):
- Every figure carries the box-counting settings (title tag + footnote).
- The 3D degree of multifractality is max D_q - min D_q (as computed in
  both legacy and rtfractal); legacy labelled it D_{-5} - D_5.
- The per-time D(q) figure also shows the raw D_q (before the concavity
  projection) and says when the projection collapsed tau(q) for q >= 0.
"""
import glob
import os

import numpy as np
import pandas as pd

from .common import (Saver, annotate_settings, load_meta, new_figure, read_temporal, settings_tag,
                     time_colors)

PROJ_NOTE = r'$D_q$, $f(\alpha)$ after the concavity projection of $\tau(q)$'


def plot_phase2(case, analysis_dir, out_dir=None, dpi=None, log=print, meta=None, assume_settings=None):
    """Draw the phase-2 figures of analysis_dir into out_dir/multifractal
    (default analysis_dir).  Returns the list of files written."""
    meta = meta or load_meta(analysis_dir, case, assume_settings)
    save = Saver(out_dir or analysis_dir, **({'dpi': dpi} if dpi else {}), log=log)
    mf_dir = os.path.join(analysis_dir, 'multifractal')
    if meta.dim == 3:
        results = _load_3d(mf_dir)
    else:
        results = _load_2d(mf_dir, meta)
    if not results:
        log(f"  phase 2 plots: no multifractal results in {mf_dir}; skipped")
        return []
    _overview(results, meta, save, read_temporal(analysis_dir))
    if meta.dim == 2:
        for r in results:
            _per_time_2d(r, meta, save)
    log(f"  phase 2: {len(save.written)} figures")
    return save.written


# ─── loading ─────────────────────────────────────────────────────────────────

def _load_2d(mf_dir, meta):
    """One dict per multifractal_summary.csv row (legacy order), with the
    spectrum arrays from its time directory."""
    p = os.path.join(mf_dir, 'multifractal_summary.csv')
    if not os.path.isfile(p):
        return []
    summ = pd.read_csv(p)
    prefixes = {'solavof': ['parameter_'], 'openfoam': ['time_']}.get(meta.format, ['parameter_', 'time_'])
    out = []
    for _, row in summ.iterrows():
        d = _find_time_dir(mf_dir, prefixes, float(row['time']))
        if d is None:
            continue
        res = pd.read_csv(os.path.join(d, 'multifractal_results.csv'))
        prm = os.path.join(d, 'multifractal_parameters.csv')
        prm = dict(pd.read_csv(prm).values) if os.path.isfile(prm) else {}
        out.append(dict(row.to_dict(), dir=d, subdir=os.path.basename(d), q=res['q'].values,
                        Dq=res['Dq'].values, alpha=res['alpha'].values, f_alpha=res['f_alpha'].values,
                        r_squared=res['r_squared'].values,
                        Dq_raw=res['Dq_raw'].values if 'Dq_raw' in res else None,
                        collapsed=bool(float(prm.get('convexity_collapsed', 0) or 0))))
    return out


def _find_time_dir(mf_dir, prefixes, t):
    for pre in prefixes:                     # exact name first (str(float) round-trips through the CSV)
        d = os.path.join(mf_dir, f'{pre}{t}')
        if os.path.isfile(os.path.join(d, 'multifractal_results.csv')):
            return d
    for pre in prefixes:
        for d in glob.glob(os.path.join(mf_dir, f'{pre}*')):
            try:
                if np.isclose(float(os.path.basename(d)[len(pre):]), t, rtol=0, atol=1e-9) and \
                        os.path.isfile(os.path.join(d, 'multifractal_results.csv')):
                    return d
            except ValueError:
                pass
    return None


def _load_3d(mf_dir):
    p = os.path.join(mf_dir, 'multifractal_summary_3d.csv')
    if not os.path.isfile(p):
        return []
    out = []
    for _, row in pd.read_csv(p).iterrows():
        f = os.path.join(mf_dir, f"t{row['time']:.2f}__multifractal_spectrum_3d.csv")
        if not os.path.isfile(f):
            continue
        s = pd.read_csv(f)
        prm = os.path.join(mf_dir, f"t{row['time']:.2f}__multifractal_params_3d.csv")
        prm = dict(pd.read_csv(prm).values) if os.path.isfile(prm) else {}
        out.append(dict(row.to_dict(), q=s['q'].values, Dq=s['D_q'].values, alpha=s['alpha'].values,
                        f_alpha=s['f_alpha'].values, r_squared=s['R_squared'].values,
                        Dq_raw=s['D_q_raw'].values if 'D_q_raw' in s else None,
                        collapsed=bool(float(prm.get('convexity_collapsed', 0) or 0))))
    return out


# ─── figures ─────────────────────────────────────────────────────────────────

def _overview(results, meta, save, df):
    is3 = meta.dim == 3
    sfx3 = '_3d' if is3 else ''
    tag = settings_tag(meta, 2)
    order = sorted(results, key=lambda r: r['time'])
    colors = dict(zip([id(r) for r in order], time_colors([r['time'] for r in order])))

    # --- overlaid f(alpha) ---
    fig, ax = new_figure((10, 7))
    for r in order:
        valid = ~np.isnan(r['alpha']) & ~np.isnan(r['f_alpha'])
        if np.sum(valid) < 2:
            continue
        a, fa = r['alpha'][valid], r['f_alpha'][valid]
        i = np.argsort(a)
        ax.plot(a[i], fa[i], '-o', color=colors[id(r)], ms=3, lw=1.5, label=rf"$\tau$={r['tau']:.3f}")
    ax.set_xlabel(r'$\alpha$', fontsize=14)
    ax.set_ylabel(r'$f(\alpha)$', fontsize=14)
    title = r'3D Multifractal Spectrum $f(\alpha)$' if is3 else r'Multifractal Spectrum Evolution $f(\alpha)$'
    ax.set_title(f'{title}\n{tag}', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True)
    ax.text(0.01, 0.01, PROJ_NOTE, transform=ax.transAxes, fontsize=8, color='0.3')
    annotate_settings(fig, meta, 2)
    save(fig, 'multifractal', 'f_alpha_evolution_3d.png' if is3 else 'f_alpha_evolution_overlay.png',
         notes_bottom=True)

    # --- overlaid D(q) ---
    fig, ax = new_figure((10, 7))
    for r in order:
        valid = ~np.isnan(r['Dq'])
        ax.plot(r['q'][valid], r['Dq'][valid], '-o', color=colors[id(r)], ms=3, lw=1.5,
                label=rf"$\tau$={r['tau']:.3f}")
    ax.set_xlabel('q', fontsize=14)
    ax.set_ylabel('D(q)', fontsize=14)
    title = '3D Generalized Dimensions D(q)' if is3 else 'Generalized Dimensions D(q) Evolution'
    ax.set_title(f'{title}\n{tag}', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True)
    ax.text(0.01, 0.01, PROJ_NOTE, transform=ax.transAxes, fontsize=8, color='0.3')
    annotate_settings(fig, meta, 2)
    save(fig, 'multifractal', 'Dq_evolution_3d.png' if is3 else 'Dq_evolution_overlay.png', notes_bottom=True)

    # --- D0, D1, D2 vs tau (legacy order: summary rows) ---
    tau = [r['tau'] for r in results]
    D0, D1, D2 = ([r[k] for r in results] for k in ('D0', 'D1', 'D2'))
    fig, ax = new_figure((10, 6))
    ax.plot(tau, D0, 'bo-', ms=6, lw=2, label=r'$D_0$ (capacity)')
    ax.plot(tau, D1, 'rs-', ms=6, lw=2, label=r'$D_1$ (information)')
    ax.plot(tau, D2, 'gd-', ms=6, lw=2, label=r'$D_2$ (correlation)')
    if is3:
        ax.axhline(y=2.0, color='gray', ls='--', alpha=0.5, label='D=2')
    ax.set_xlabel(r'$\tau$', fontsize=14)
    ax.set_ylabel('Dimension', fontsize=14)
    title = '3D Generalized Dimensions vs. Time' if is3 else 'Generalized Dimensions vs. Nondimensional Time'
    ax.set_title(f'{title}\n{tag}', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)
    annotate_settings(fig, meta, 2)
    save(fig, 'multifractal', f'D012_vs_tau{sfx3}.png', notes_bottom=True)

    # --- multifractality measures ---
    fig, ax = new_figure((10, 6))
    ax.plot(tau, [r['alpha_width'] for r in results], 'ms-', ms=6, lw=2, label=r'$\Delta\alpha$ (spectrum width)')
    deg_label = (r'$\max D_q - \min D_q$ (multifractality degree)' if is3
                 else r'$D_{-5} - D_5$ (multifractality degree)')
    ax.plot(tau, [r['degree_multifractality'] for r in results], 'cd-', ms=6, lw=2, label=deg_label)
    ax.set_xlabel(r'$\tau$', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    title = '3D Multifractality Measures vs. Time' if is3 else 'Multifractality Measures vs. Time'
    ax.set_title(f'{title}\n{tag}', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)
    annotate_settings(fig, meta, 2)
    save(fig, 'multifractal', f'multifractality_measures{sfx3}.png', notes_bottom=True)

    # --- phase portrait: dimension vs h/H ---
    col = 'h_total_int_nondim' if is3 else 'h_total_geo_nondim'
    h = []
    for r in results:
        v = r.get('h_H', np.nan)
        if (v is None or not np.isfinite(v)) and df is not None and col in df.columns:
            v = float(df.loc[(df['time'] - r['time']).abs().idxmin(), col])
        h.append(np.nan if v is None else v)
    if np.isfinite(h).any():
        if is3:
            fig, ax = new_figure((8, 6))
            for hv, r in zip(h, results):
                ax.plot(hv, r['D0'], 'bo', ms=8)
                ax.annotate(rf"$\tau$={r['tau']:.2f}", (hv, r['D0']), textcoords='offset points', xytext=(5, 5),
                            fontsize=8)
            ax.set_xlabel(r'$h/H$ (integral)', fontsize=14)
            ax.set_ylabel(r'$D_0$', fontsize=14)
            ax.set_title(f'Phase portrait: 3D Fractal Dimension vs Mixing Extent\n{tag}', fontsize=12)
            name = 'phase_portrait_D0_h_3d.png'
        else:
            fig, ax = new_figure((10, 7))
            ax.plot(h, D0, 'bo-', ms=8, lw=2, label=r'$D_0$')
            ax.plot(h, D1, 'rs-', ms=8, lw=2, label=r'$D_1$')
            for hv, d0, tv in zip(h, D0, tau):
                ax.annotate(rf'$\tau$={tv:.2f}', (hv, d0), textcoords='offset points', xytext=(5, 5), fontsize=8)
            ax.set_xlabel(r'$h/H$ (geometric)', fontsize=14)
            ax.set_ylabel('Fractal Dimension', fontsize=14)
            ax.set_title(f'Phase Portrait: Dimension vs. Mixing Thickness\n{tag}', fontsize=12)
            ax.legend(fontsize=12)
            name = 'phase_portrait.png'
        ax.grid(True)
        annotate_settings(fig, meta, 2)
        save(fig, 'multifractal', name, notes_bottom=True)


def _per_time_2d(r, meta, save):
    """Legacy per-time figures, written into the time directory."""
    sub = os.path.join('multifractal', r['subdir'])
    q, Dq, alpha, fa, r2 = r['q'], r['Dq'], r['alpha'], r['f_alpha'], r['r_squared']
    t = r['time']
    tag = settings_tag(meta, 2)

    fig, ax = new_figure((10, 6))
    valid = ~np.isnan(Dq)
    ax.plot(q[valid], Dq[valid], 'bo-', markersize=4, label='D(q)')
    if r.get('Dq_raw') is not None:
        raw = r['Dq_raw']
        vr = ~np.isnan(raw)
        if vr.any() and not np.allclose(raw[vr], Dq[vr], equal_nan=True):
            ax.plot(q[vr], raw[vr], '--', color='0.5', lw=1, marker='.', ms=3,
                    label='raw D(q) (before concavity projection)')
    if 0 in q:
        d0 = Dq[np.searchsorted(q, 0)]
        ax.axhline(y=d0, color='r', linestyle='--', label=f'D(0) = {d0:.4f}')
    ax.set_xlabel('q')
    ax.set_ylabel('D(q)')
    coll = '\nconcavity projection collapsed tau(q) for q >= 0: D_q = D_0 there is an artefact' \
        if r.get('collapsed') else ''
    ax.set_title(f'Generalized Dimensions D(q) at t = {t:.2f}\n{tag}{coll}', fontsize=11)
    ax.grid(True)
    ax.legend()
    annotate_settings(fig, meta, 2)
    save(fig, sub, 'multifractal_dimensions.png', notes_bottom=True)

    fig, ax = new_figure((10, 6))
    valid = ~np.isnan(alpha) & ~np.isnan(fa)
    ax.plot(alpha[valid], fa[valid], 'bo-', markersize=4)
    for qv in (-5, -2, 0, 2, 5):
        if qv in q:
            i = np.searchsorted(q, qv)
            if i < len(q) and valid[i]:
                ax.annotate(f"q={q[i]}", (alpha[i], fa[i]), xytext=(5, 0), textcoords='offset points')
    ax.set_xlabel('α')
    ax.set_ylabel('f(α)')
    ax.set_title(f'Multifractal Spectrum f(α) at t = {t:.2f}\n{tag}', fontsize=11)
    ax.grid(True)
    annotate_settings(fig, meta, 2)
    save(fig, sub, 'multifractal_spectrum.png', notes_bottom=True)

    fig, ax = new_figure((10, 6))
    valid = ~np.isnan(r2)
    ax.plot(q[valid], r2[valid], 'go-', markersize=4)
    ax.set_xlabel('q')
    ax.set_ylabel('R²')
    ax.set_title(f'Fit Quality for Different q Values at t = {t:.2f}\n{tag}', fontsize=11)
    ax.grid(True)
    annotate_settings(fig, meta, 2)
    save(fig, sub, 'multifractal_r_squared.png', notes_bottom=True)
