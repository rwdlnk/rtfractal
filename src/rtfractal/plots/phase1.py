"""Phase 1 figures (analysis/summary/), drawn from temporal_results.csv.

2D (legacy analyze_rt.py _plot_phase1 and _plot_phase1_openfoam, merged):
    mixing_evolution.png, mixing_self_similar.png, fractal_dimension_evolution.png,
    combined_evolution.png, D_and_dh_dtau.png, dalziel_integral_measures.png,
    interface_snapshots.png  (F fields from plotdata/field_t*.npz, else read from the case;
                              contours from interfaces/interface_t*.dat)
3D (legacy analyze_rt_3d.py _plot_phase1):
    mixing_evolution_3d.png, selfsimilar_scaling_3d.png, fractal_evolution_3d.png,
    dalziel_integrals_3d.png, surface_complexity_3d.png

Deliberate differences from legacy (content, not style):
- ht is the upper (bubble) front and hb the lower (spike) front everywhere.
  The legacy OpenFOAM 2D path and the 3D path still had them swapped.
- Figures showing D carry the box-counting settings (title tag + footnote).
- D_and_dh_dtau: the two end points of the 3-point running mean (which
  legacy averaged against zero padding) are not drawn.
- The OpenFOAM 2D path now also gets dalziel_integral_measures.png.
"""
import os

import numpy as np
from matplotlib.collections import LineCollection

from .. import interface, plotdata
from .common import Saver, annotate_settings, combined_legend, load_meta, new_figure, read_temporal, settings_tag

TAU_LABEL = r'$\tau = t\sqrt{Ag/H}$'


def plot_phase1(case, analysis_dir, out_dir=None, dpi=None, log=print, meta=None, assume_settings=None):
    """Draw the phase-1 figures of analysis_dir into out_dir/summary (default
    analysis_dir).  case may be None (the interface snapshots then need
    plotdata/field_t*.npz).  Returns the list of files written."""
    meta = meta or load_meta(analysis_dir, case, assume_settings)
    df = read_temporal(analysis_dir)
    if df is None:
        log(f"  phase 1 plots: no {analysis_dir}/summary/temporal_results.csv; skipped")
        return []
    save = Saver(out_dir or analysis_dir, **({'dpi': dpi} if dpi else {}), log=log)
    if meta.dim == 3:
        _plot_3d(df, meta, save, log)
    else:
        _plot_2d(df, meta, save, log)
    log(f"  phase 1: {len(save.written)} figures")
    return save.written


def _suffix(meta, sep=' -- '):
    return f'{sep}{meta.run_label}' if meta.run_label else ''


def _ylim_cover(ax, lo, hi, values):
    """Legacy fixed limits, widened if the data fall outside them."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v):
        lo, hi = min(lo, v.min() - 0.02 * (hi - lo)), max(hi, v.max() + 0.02 * (hi - lo))
    ax.set_ylim(lo, hi)


# ─── 2D ──────────────────────────────────────────────────────────────────────

def _plot_2d(df, meta, save, log):
    dfv = df[df['time'] > 0.01].copy()
    if len(dfv) < 2:
        log("  phase 1 plots: not enough data points")
        return
    sfx = _suffix(meta)
    tag = settings_tag(meta, 1)

    # --- mixing thickness evolution ---
    fig, (ax1, ax2) = new_figure((14, 6), 1, 2)
    ax1.plot(dfv['time'], dfv['h_total_geo'], 'b-', lw=2, label='Geometric (total)')
    ax1.plot(dfv['time'], dfv['ht_geo'], 'r--', lw=1.5, label='Geometric (bubble)')
    ax1.plot(dfv['time'], dfv['hb_geo'], 'g--', lw=1.5, label='Geometric (spike)')
    ax1.plot(dfv['time'], dfv['h_total_stat'], 'k:', lw=2, label='Statistical (total)')
    if 'h_total_int' in dfv.columns:
        ax1.plot(dfv['time'], dfv['h_total_int'], 'm-', lw=2, label='Integral (total)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Mixing thickness (m)')
    ax1.set_title(f'Mixing Layer Evolution (dimensional){sfx}')
    ax1.legend(fontsize=9)
    ax1.grid(True)
    ax2.plot(dfv['tau'], dfv['h_total_geo_nondim'], 'b-', lw=2, label='Geometric (total)')
    ax2.plot(dfv['tau'], dfv['ht_geo_nondim'], 'r--', lw=1.5, label='Bubble')
    ax2.plot(dfv['tau'], dfv['hb_geo_nondim'], 'g--', lw=1.5, label='Spike')
    if 'h_total_int_nondim' in dfv.columns:
        ax2.plot(dfv['tau'], dfv['h_total_int_nondim'], 'm-', lw=2, label='Integral (total)')
    ax2.set_xlabel(TAU_LABEL)
    ax2.set_ylabel(r'$h/H$')
    ax2.set_title(f'Mixing Layer Evolution (nondimensional){sfx}')
    ax2.legend(fontsize=9)
    ax2.grid(True)
    save(fig, 'summary', 'mixing_evolution.png')

    # --- self-similar scaling: h/H vs tau^2 (legacy: fit over tau > 0.3 tau_max, no wall guard) ---
    fig, ax = new_figure((8, 6))
    t2 = dfv['tau'] ** 2
    ax.plot(t2, dfv['h_total_geo_nondim'], 'bo-', ms=4, lw=1.5, label=r'$h/H$ vs $\tau^2$ (geometric)')
    ax.plot(t2, dfv['h_total_stat_nondim'], 'rs-', ms=4, lw=1.5, label=r'$h/H$ vs $\tau^2$ (statistical)')
    has_int = 'h_total_int_nondim' in dfv.columns
    if has_int:
        ax.plot(t2, dfv['h_total_int_nondim'], 'm^-', ms=4, lw=1.5, label=r'$h/H$ vs $\tau^2$ (integral)')
    tau_lo = dfv['tau'].max() * 0.3
    late = dfv[dfv['tau'] > tau_lo]
    if len(late) > 3:
        tau2_fit = np.linspace(late['tau'].min() ** 2, late['tau'].max() ** 2, 50)
        for col, c, name in (('h_total_geo_nondim', 'b', 'geo'), ('h_total_stat_nondim', 'r', 'stat'),
                             ('h_total_int_nondim', 'm', 'int')):
            if col not in late.columns:
                continue
            coeffs = np.polyfit(late['tau'] ** 2, late[col], 1)
            ax.plot(tau2_fit, np.polyval(coeffs, tau2_fit), f'{c}--', lw=1,
                    label=rf'$\alpha_{{RT}} \approx {coeffs[0]:.4f}$ ({name})')
        ax.text(0.98, 0.02, rf'fit: $\tau > {tau_lo:.2f}$ (0.3 $\tau_{{max}}$), no wall guard',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='0.3')
    ax.set_xlabel(r'$\tau^2$')
    ax.set_ylabel(r'$h/H$')
    ax.set_title(rf'Self-similar scaling: $h/H = \alpha \tau^2${sfx}')
    ax.legend()
    ax.grid(True)
    save(fig, 'summary', 'mixing_self_similar.png')

    valid_fd = dfv.dropna(subset=['fractal_dim'])

    # --- fractal dimension evolution ---
    fig, ax1 = new_figure((10, 6))
    if len(valid_fd):
        ax1.errorbar(valid_fd['tau'], valid_fd['fractal_dim'], yerr=valid_fd['fd_error'],
                     fmt='ko-', capsize=3, lw=1.5, ms=4, label='Fractal dimension')
        ax1.fill_between(valid_fd['tau'], valid_fd['fractal_dim'] - valid_fd['fd_error'],
                         valid_fd['fractal_dim'] + valid_fd['fd_error'], alpha=0.2, color='gray')
    ax1.set_xlabel(TAU_LABEL)
    ax1.set_ylabel('Fractal dimension D')
    ax1.set_title(f'Fractal Dimension Evolution{sfx}\n{tag}', fontsize=11)
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(valid_fd['tau'], valid_fd['fd_r_squared'], 'g--', alpha=0.6, label='R2')
    ax2.set_ylabel('R2', color='g')
    ax2.set_ylim(0.9, 1.0)
    ax2.tick_params(axis='y', labelcolor='g')
    combined_legend(ax1, ax2, loc='lower right')
    annotate_settings(fig, meta, 1)
    save(fig, 'summary', 'fractal_dimension_evolution.png', notes_bottom=True)

    # --- combined: mixing + fractal dimension ---
    fig, ax1 = new_figure((12, 7))
    ax1.plot(dfv['tau'], dfv['h_total_geo_nondim'], 'b-', lw=2, label='Mixing thickness h/H')
    ax1.set_xlabel(r'$\tau$', fontsize=14)
    ax1.set_ylabel(r'$h/H$', color='b', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    if len(valid_fd):
        ax2.errorbar(valid_fd['tau'], valid_fd['fractal_dim'], yerr=valid_fd['fd_error'],
                     fmt='ro-', capsize=3, ms=4, label='Fractal dimension D')
    ax2.set_ylabel('Fractal dimension D', color='r', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='r')
    combined_legend(ax1, ax2, loc='upper left')
    ax1.set_title(f'Mixing Layer Growth and Fractal Dimension{sfx}\n{tag}', fontsize=13)
    annotate_settings(fig, meta, 1)
    save(fig, 'summary', 'combined_evolution.png', notes_bottom=True)

    # --- D(tau) and d(h/H)/dtau (bubble-merger signature) ---
    tau_h, h_vals = dfv['tau'].values, dfv['h_total_geo_nondim'].values
    if len(tau_h) > 2:
        dh_dtau = np.diff(h_vals) / np.diff(tau_h)
        tau_mid = 0.5 * (tau_h[:-1] + tau_h[1:])
        dh_smooth = np.convolve(dh_dtau, np.ones(3) / 3, mode='same')
        if len(dh_smooth) > 2:                    # end points average against zero padding
            tau_mid, dh_smooth = tau_mid[1:-1], dh_smooth[1:-1]
        fig, ax1 = new_figure((12, 7))
        if len(valid_fd):
            ax1.errorbar(valid_fd['tau'], valid_fd['fractal_dim'], yerr=valid_fd['fd_error'],
                         fmt='ko-', capsize=2, ms=3, lw=1, label='Fractal dimension D')
        ax1.set_xlabel(TAU_LABEL, fontsize=14)
        ax1.set_ylabel('Fractal dimension D', fontsize=14)
        ax2 = ax1.twinx()
        ax2.plot(tau_mid, dh_smooth, 'b-', lw=1.5, alpha=0.7, label=r'$d(h/H)/d\tau$ (3-point mean)')
        ax2.set_ylabel(r'$d(h/H)/d\tau$', color='b', fontsize=14)
        ax2.tick_params(axis='y', labelcolor='b')
        combined_legend(ax1, ax2, loc='upper left')
        ax1.set_title(rf'Fractal Dimension and Mixing Rate $d(h/H)/d\tau${sfx}' + f'\n{tag}', fontsize=13)
        annotate_settings(fig, meta, 1)
        save(fig, 'summary', 'D_and_dh_dtau.png', notes_bottom=True)

    # --- Dalziel integral measures h_{m,n} (fluid m: 1 heavy, 0 light; region n: 0 lower, 1 upper) ---
    if all(c in dfv.columns for c in ('h_10_nondim', 'h_01_nondim', 'h_11_nondim', 'h_00_nondim')):
        fig, ((a1, a2), (a3, a4)) = new_figure((14, 11), 2, 2)
        a1.plot(dfv['tau'], dfv['h_10_nondim'], 'r-', lw=2, label=r'$h_{1,0}$: heavy $\downarrow$ into lower')
        a1.plot(dfv['tau'], dfv['h_01_nondim'], 'b-', lw=2, label=r'$h_{0,1}$: light $\uparrow$ into upper')
        a1.plot(dfv['tau'], dfv['h_total_int_nondim'], 'k--', lw=2, label=r'$h_{1,0} + h_{0,1}$ (total)')
        a1.set_title('Penetration Depths (Dalziel Eq. 7)', fontsize=12)
        a2.plot(dfv['tau'], dfv['h_11_nondim'], 'r-', lw=2, label=r'$h_{1,1}$: heavy in own (upper)')
        a2.plot(dfv['tau'], dfv['h_00_nondim'], 'b-', lw=2, label=r'$h_{0,0}$: light in own (lower)')
        a2.axhline(0.5, color='gray', ls=':', alpha=0.5, label='H/2')
        a2.set_title('Own-Region Retention', fontsize=12)
        lower = dfv['h_10_nondim'] + dfv['h_00_nondim']
        upper = dfv['h_11_nondim'] + dfv['h_01_nondim']
        a3.plot(dfv['tau'], lower, 'k-', lw=2, label=r'$h_{1,0} + h_{0,0}$')
        a3.set_title('Conservation: Lower Half', fontsize=12)
        a4.plot(dfv['tau'], upper, 'k-', lw=2, label=r'$h_{1,1} + h_{0,1}$')
        a4.set_title('Conservation: Upper Half', fontsize=12)
        for a, vals in ((a3, lower), (a4, upper)):
            a.axhline(0.5, color='gray', ls=':', alpha=0.5, label='H/2')
            _ylim_cover(a, 0.498, 0.502, vals)        # legacy zoom, widened if the sums leave it
        for a in (a1, a2, a3, a4):
            a.set_xlabel(r'$\tau$', fontsize=12)
            a.set_ylabel(r'$h/H$', fontsize=12)
            a.legend(fontsize=9)
            a.grid(True)
        if meta.run_label:
            fig.suptitle(f'Dalziel integral measures -- {meta.run_label}', fontsize=13)
        save(fig, 'summary', 'dalziel_integral_measures.png')

    _interface_snapshots(df, meta, save, log)


def _snapshot_fields(df, meta, log):
    """F snapshots for interface_snapshots.png: plotdata npz first, else the case."""
    have = plotdata.list_npz(meta.analysis_dir, 'field')
    targets = plotdata.interface_snapshot_targets(list(df['time']))
    out = []
    if have:
        snaps = [plotdata.load_npz(p) for _, p in sorted(have.items())]
        for t in targets:
            out.append(min(snaps, key=lambda s: abs(s['t'] - t)))
        return out
    case = meta.case
    if case is None:
        log("  interface_snapshots.png: no plotdata/field_t*.npz and no case to read; skipped")
        return None
    for t in targets:
        snap = plotdata.nearest(case.snapshots, t)
        data = case.read_2d(snap, fields=('F',))
        out.append(plotdata.field_snapshot(data, snap.t, float(case.physics.nondim_time(snap.t)), case.H0,
                                           snap.time_idx))
    return out


def _read_interface_dat(path):
    segs = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            a, b = line.split()
            x1, y1 = a.split(',')
            x2, y2 = b.split(',')
            segs.append(((float(x1), float(y1)), (float(x2), float(y2))))
    return segs


def _interface_snapshots(df, meta, save, log):
    snaps = _snapshot_fields(df, meta, log)
    if not snaps:
        return
    n = len(snaps)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = new_figure((5 * ncols, 5 * nrows), nrows, ncols, squeeze=False)
    axes = axes.flatten()
    for ax, s in zip(axes, snaps):
        X, Y = np.meshgrid(s['x'], s['y'], indexing='ij')
        ax.contourf(X, Y, s['f'], levels=20, cmap='RdBu_r')
        dat = os.path.join(meta.analysis_dir, 'interfaces', f"interface_t{int(s['time_idx']):05d}.dat")
        if os.path.isfile(dat):
            ax.add_collection(LineCollection(_read_interface_dat(dat), colors='k', linewidths=0.8))
        else:                                   # contour of the stored field
            for c in interface.contours(np.asarray(s['f'], dtype=np.float64), X, Y, 0.5):
                ax.plot(c[:, 0], c[:, 1], 'k-', lw=0.8)
        if s.get('H0') is not None:
            ax.axhline(y=s['H0'], color='gray', ls='--', alpha=0.5)
        ax.set_title(rf"t={s['t']:.2f} s ($\tau$={s['tau']:.3f})", fontsize=11)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_xlim(s['x'][0], s['x'][-1])
        ax.set_ylim(s['y'][0], s['y'][-1])
        ax.set_aspect('equal')
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f'RT Interface Evolution{_suffix(meta)}', fontsize=14, y=1.02)
    save(fig, 'summary', 'interface_snapshots.png', bbox_tight=True)


# ─── 3D ──────────────────────────────────────────────────────────────────────

WALL_FRAC = 2.0 / 3.0          # legacy wall guard for the alpha_RT fit (analyze_rt_3d.py)


def _plot_3d(df, meta, save, log):
    dfv = df[df['time'] > 0.01].copy()
    if len(dfv) < 2:
        log("  phase 1 plots: not enough data points")
        return
    res = f' ({meta.res_label})' if meta.res_label else ''

    # --- mixing thickness ---
    fig, (ax1, ax2) = new_figure((14, 6), 1, 2)
    if 'h_total_stat' in dfv.columns:
        ax1.plot(dfv['time'], dfv['h_total_stat'], 'k:', lw=2, label='Statistical')
    if 'h_total_int' in dfv.columns:
        ax1.plot(dfv['time'], dfv['h_total_int'], 'm-', lw=2, label='Integral (Dalziel)')
        ax1.plot(dfv['time'], dfv['ht_int'], 'r--', lw=1.5, label='Bubble (int)')
        ax1.plot(dfv['time'], dfv['hb_int'], 'g--', lw=1.5, label='Spike (int)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Mixing thickness (m)')
    ax1.set_title(f'3D Mixing Layer Evolution{res}')
    ax1.legend(fontsize=9)
    ax1.grid(True)
    if 'h_total_int_nondim' in dfv.columns:
        ax2.plot(dfv['tau'], dfv['h_total_int_nondim'], 'm-', lw=2, label='Integral (total)')
        ax2.plot(dfv['tau'], dfv['ht_int_nondim'], 'r--', lw=1.5, label='Bubble')
        ax2.plot(dfv['tau'], dfv['hb_int_nondim'], 'g--', lw=1.5, label='Spike')
    ax2.set_xlabel(TAU_LABEL)
    ax2.set_ylabel(r'$h/H$')
    ax2.set_title('Mixing Layer (nondimensional)')
    ax2.legend(fontsize=9)
    ax2.grid(True)
    save(fig, 'summary', 'mixing_evolution_3d.png')

    # --- self-similar scaling with the legacy wall guard ---
    fig, ax = new_figure((8, 6))
    if 'h_total_int_nondim' in dfv.columns:
        ax.plot(dfv['tau'] ** 2, dfv['h_total_int_nondim'], 'm^-', ms=4, lw=1.5,
                label=r'$h/H$ vs $\tau^2$ (integral)')
        tau_wall, wall_front = np.inf, None
        travel = 0.5 * meta.H if meta.H else None      # H0 is the mid-height: each front has H/2 to cross
        if travel:
            for col, name in (('hb_stat', 'descending'), ('ht_stat', 'ascending')):
                if col not in dfv.columns:
                    continue
                hit = dfv[dfv[col] >= WALL_FRAC * travel]
                if len(hit) and hit['tau'].iloc[0] < tau_wall:
                    tau_wall, wall_front = hit['tau'].iloc[0], name
        tau_lo = dfv['tau'].max() * 0.3
        if np.isfinite(tau_wall):
            tau_hi = min(tau_wall, dfv['tau'].max())
            guard_txt = f'{wall_front} front at {WALL_FRAC:.2f} travel, $\\tau \\leq {tau_hi:.2f}$'
        else:
            tau_hi = dfv['tau'].max()
            guard_txt = 'no wall contact' if travel else 'wall guard off: H unknown'
        late = dfv[(dfv['tau'] > tau_lo) & (dfv['tau'] <= tau_hi)]
        if len(late) > 3 and not late['h_total_int_nondim'].isna().all():
            coeffs = np.polyfit(late['tau'] ** 2, late['h_total_int_nondim'], 1)
            tau2_fit = np.linspace(late['tau'].min() ** 2, late['tau'].max() ** 2, 50)
            ax.plot(tau2_fit, np.polyval(coeffs, tau2_fit), 'm--', lw=1,
                    label=rf'$\alpha_{{RT}} \approx {coeffs[0]:.4f}$' + f'\n({guard_txt})')
    ax.set_xlabel(r'$\tau^2$')
    ax.set_ylabel(r'$h/H$')
    ax.set_title('Self-Similar Scaling (3D)')
    ax.legend(fontsize=10)
    ax.grid(True)
    save(fig, 'summary', 'selfsimilar_scaling_3d.png')

    # --- fractal dimension ---
    fig, ax = new_figure((10, 6))
    valid = dfv['fractal_dim'].notna()
    if valid.any():
        ax.errorbar(dfv.loc[valid, 'tau'], dfv.loc[valid, 'fractal_dim'], yerr=dfv.loc[valid, 'fd_error'],
                    fmt='bo-', ms=5, lw=1.5, capsize=3, label='3D fractal dimension')
        ax.axhline(y=2.0, color='gray', ls='--', alpha=0.5, label='D=2 (smooth surface)')
        ax.legend(fontsize=10)
        _ylim_cover(ax, 1.8, 2.8, dfv.loc[valid, 'fractal_dim'])
    ax.set_xlabel(TAU_LABEL)
    ax.set_ylabel('Fractal Dimension D')
    ax.set_title(f'3D Fractal Dimension Evolution{res}\n{settings_tag(meta, 1)}', fontsize=11)
    ax.grid(True)
    annotate_settings(fig, meta, 1)
    save(fig, 'summary', 'fractal_evolution_3d.png', notes_bottom=True)

    # --- Dalziel integral measures ---
    if all(c in dfv.columns for c in ('h_10_nondim', 'h_11_nondim', 'h_00_nondim', 'h_01_nondim')):
        fig, axes = new_figure((14, 10), 2, 2)
        a = axes[0, 0]
        a.plot(dfv['tau'], dfv['h_10_nondim'], 'b-', lw=2, label=r'$h_{1,0}/H$ (heavy in lower)')
        a.plot(dfv['tau'], dfv['h_01_nondim'], 'r-', lw=2, label=r'$h_{0,1}/H$ (light in upper)')
        a.set_title('Penetration depths')
        a = axes[0, 1]
        a.plot(dfv['tau'], dfv['h_11_nondim'], 'b-', lw=2, label=r'$h_{1,1}/H$')
        a.plot(dfv['tau'], dfv['h_00_nondim'], 'r-', lw=2, label=r'$h_{0,0}/H$')
        a.set_title('Retention in own region')
        for a in (axes[0, 0], axes[0, 1]):
            a.set_ylabel(r'$h/H$')
            a.legend()
            a.grid(True)
        a = axes[1, 0]                     # (h/H)/(1/2) == h/(H/2)
        a.plot(dfv['tau'], (dfv['h_10_nondim'] + dfv['h_00_nondim']) / 0.5, 'b-', lw=2,
               label=r'$(h_{1,0}+h_{0,0})/(H/2)$')
        a.plot(dfv['tau'], (dfv['h_11_nondim'] + dfv['h_01_nondim']) / 0.5, 'r-', lw=2,
               label=r'$(h_{1,1}+h_{0,1})/(H/2)$')
        a.axhline(y=1.0, color='k', ls='--', alpha=0.5)
        a.set_title('Conservation check (should = 1)')
        a.set_ylabel('Ratio')
        a.set_xlabel(r'$\tau$')
        a.legend()
        a.grid(True)
        a = axes[1, 1]
        if 'youngs_W' in dfv.columns:
            a.plot(dfv['tau'], dfv['youngs_W'], 'g-', lw=2, label="Young's W")
        ax_eff = a.twinx()
        if 'mixing_efficiency' in dfv.columns:
            ax_eff.plot(dfv['tau'], dfv['mixing_efficiency'], 'k--', lw=1.5, label='Mixing efficiency')
        a.set_xlabel(r'$\tau$')
        a.set_ylabel("Young's W (m)")
        ax_eff.set_ylabel('Mixing efficiency')
        a.set_title("Young's integral width & efficiency")
        combined_legend(a, ax_eff)
        a.grid(True)
        fig.suptitle(f'Dalziel Integral Measures — 3D{res}', fontsize=14, y=1.01)
        save(fig, 'summary', 'dalziel_integrals_3d.png', bbox_tight=True)

    # --- surface complexity ---
    if 'n_triangles' in dfv.columns and 'surface_area' in dfv.columns:
        fig, (ax1, ax2) = new_figure((14, 5), 1, 2)
        ax1.plot(dfv['tau'], dfv['n_triangles'], 'b.-', ms=4)
        ax1.set_xlabel(r'$\tau$')
        ax1.set_ylabel('Number of triangles')
        ax1.set_title('Isosurface triangle count')
        ax1.grid(True)
        ax2.plot(dfv['tau'], dfv['surface_area'], 'r.-', ms=4)
        ax2.set_xlabel(r'$\tau$')
        ax2.set_ylabel('Surface area (m²)')
        ax2.set_title('Isosurface area evolution')
        ax2.grid(True)
        save(fig, 'summary', 'surface_complexity_3d.png')
