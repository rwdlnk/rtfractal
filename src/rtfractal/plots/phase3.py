"""Phase 3 figures, drawn from the phase-3 CSVs (2D only: the legacy 3D
analyzer draws no phase-3 figures).

2D (legacy analyze_rt.py _plot_phase3 and _plot_phase3_openfoam, merged):
    velocity/velocity_statistics_evolution.png   velocity/velocity_statistics.csv
    velocity/velocity_field_snapshots.png        plotdata/velocity_t*.npz, else the case
    spectra/dalziel_slope_evolution.png          spectra/spectrum_summary.csv
    spectra/dalziel_spectra_snapshots.png        spectra/dalziel_pk_t*.csv (band fits redone
                                                 with spectra.fit_powerlaw_weighted: same numbers)
    spectra/spectrum_evolution.png               spectra/spectrum_summary.csv
    spectra/spectrum_t<t>.png                    plotdata/radial_spectra_t*.csv, else the case

Deliberate differences from legacy (content, not style):
- spectrum_t<t>.png shows the spectrum the slope was fitted to (mean and
  least-squares plane removed, as phase 3 computes it), and the fit line is
  the actual fit over the middle half of the bins.  Legacy recomputed a
  spectrum without the plane detrend, then drew the detrended slope through
  it with an intercept matched by eye (mean offset), so line and points did
  not belong together.
- The TKE curves are labelled with the TKE-spectrum mode recorded in
  rtfractal_run.json: "½(P_u+P_v)" (--tke uv, default) or "legacy: u only"
  (--tke legacy, where slope/R² are u's alone and only the total energy
  averages u and v).
- The 2D radial spectra are mean power per mode P(k), not ring-integrated
  E(k) ~ k P(k).  Legacy drew k^-5/3 and k^-3 reference lines against them;
  here they are the P(k) equivalents k^-8/3 and k^-4.  (The Dalziel slab
  spectrum is 1D, so its -5/3 line stays.)
- Vorticity panels use 21 symmetric levels over +-(98th percentile of
  |omega|), as legacy intended; legacy's levels spanned the full range, so its
  colour bar and contrast were set by the extreme values.
"""
import os

import numpy as np
import pandas as pd

from .. import plotdata, spectra
from .common import Saver, load_meta, new_figure

DALZIEL_REFS = (           # Dalziel et al. (1999) tau = 2 slopes, legacy markers
    (-1.49, 'kv', dict(ms=10), r'Dalziel exp. ($-1.49$)'),
    (-1.79, 'k^', dict(ms=10, mfc='none'), r'Dalziel idealised sim. ($-1.79$)'),
    (-1.63, 'kD', dict(ms=8, mfc='0.7'), r'Dalziel barrier sim. ($-1.63$)'),
)


def plot_phase3(case, analysis_dir, out_dir=None, dpi=None, log=print, meta=None, assume_settings=None):
    """Draw the phase-3 figures of analysis_dir into out_dir/{spectra,velocity}
    (default analysis_dir).  Returns the list of files written."""
    meta = meta or load_meta(analysis_dir, case, assume_settings)
    if meta.dim == 3:
        log("  phase 3 plots: none for 3D (the legacy 3D analyzer draws none)")
        return []
    save = Saver(out_dir or analysis_dir, **({'dpi': dpi} if dpi else {}), log=log)
    vel = os.path.join(analysis_dir, 'velocity', 'velocity_statistics.csv')
    if os.path.isfile(vel):
        vel_df = pd.read_csv(vel)
        if len(vel_df):
            _velocity_evolution(vel_df, meta, save)
            _velocity_snapshots(vel_df, meta, save, log)
    summ = os.path.join(analysis_dir, 'spectra', 'spectrum_summary.csv')
    if os.path.isfile(summ):
        sdf = pd.read_csv(summ)
        if len(sdf):
            _dalziel(sdf, meta, save, log)
            _spectrum_evolution(sdf, meta, save)
            _spectrum_snapshots(sdf, meta, save, log)
    if not save.written:
        log(f"  phase 3 plots: no phase-3 results in {analysis_dir}; skipped")
    else:
        log(f"  phase 3: {len(save.written)} figures")
    return save.written


def _sfx(meta):
    return f' -- {meta.run_label}' if meta.run_label else ''


# ─── velocity ────────────────────────────────────────────────────────────────

def _velocity_evolution(v, meta, save):
    fig, axes = new_figure((14, 10), 2, 2)
    ax = axes[0, 0]
    ax.plot(v['tau'], v['u_rms'], 'bo-', lw=2, ms=5, label=r"$u'_{rms}$")
    ax.plot(v['tau'], v['v_rms'], 'rs-', lw=2, ms=5, label=r"$v'_{rms}$")
    ax.set_ylabel('RMS velocity (m/s)')
    ax.set_title('RMS Velocity Fluctuations (about the domain mean)')
    ax.legend()
    ax = axes[0, 1]
    ax.plot(v['tau'], v['TKE'], 'kd-', lw=2, ms=5)
    ax.set_ylabel('TKE (m$^2$/s$^2$)')
    ax.set_title('Turbulent Kinetic Energy')
    ax = axes[1, 0]
    ax.plot(v['tau'], v['reynolds_stress'], 'g^-', lw=2, ms=5)
    ax.set_ylabel(r"$\langle u'v' \rangle$ (m$^2$/s$^2$)")
    ax.set_title('Reynolds Stress')
    ax = axes[1, 1]
    ax.plot(v['tau'], v['velocity_mag_max'], 'mp-', lw=2, ms=5, label='Max |V|')
    ax.plot(v['tau'], v['velocity_mag_mean'], 'c+-', lw=2, ms=5, label='Mean |V|')
    ax.set_ylabel('Velocity magnitude (m/s)')
    ax.set_title('Velocity Magnitude')
    ax.legend()
    for ax in axes.flat:
        ax.set_xlabel(r'$\tau$')
        ax.grid(True)
    fig.suptitle(f'Velocity Statistics Evolution{_sfx(meta)}', fontsize=14)
    save(fig, 'velocity', 'velocity_statistics_evolution.png')


def _velocity_fields(vel_df, meta, log):
    targets = plotdata.velocity_snapshot_targets(list(vel_df['time']))
    have = plotdata.list_npz(meta.analysis_dir, 'velocity')
    if have:
        snaps = [plotdata.load_npz(p) for _, p in sorted(have.items())]
        return [min(snaps, key=lambda s: abs(s['t'] - t)) for t in targets]
    case = meta.case
    if case is None:
        log("  velocity_field_snapshots.png: no plotdata/velocity_t*.npz and no case to read; skipped")
        return None
    out = []
    for t in targets:
        snap = plotdata.nearest(case.snapshots, t)
        data = case.read_2d(snap, fields=('F', 'U', 'V'))
        if 'U' in data and 'V' in data:
            out.append(plotdata.velocity_snapshot(data, snap.t, float(case.physics.nondim_time(snap.t)),
                                                  snap.time_idx))
    return out


def _velocity_snapshots(vel_df, meta, save, log):
    snaps = _velocity_fields(vel_df, meta, log)
    if not snaps:
        return
    n = len(snaps)
    fig, axes = new_figure((5 * n, 10), 2, n, squeeze=False)
    for i, s in enumerate(snaps):
        X, Y = np.meshgrid(s['x'], s['y'], indexing='ij')
        ax = axes[0, i]
        cf = ax.contourf(X, Y, s['vel_mag'], levels=20, cmap='hot')
        fig.colorbar(cf, ax=ax, label='|V| (m/s)')
        ax.set_title(rf"t={s['t']:.1f} s ($\tau$={s['tau']:.3f})")
        ax = axes[1, i]
        vmax = s['vort_p98']
        # symmetric levels over +-98th percentile, values beyond drawn in the end colours
        # (legacy passed vmin/vmax with levels=20, so the levels still spanned the full range)
        levels = np.linspace(-vmax, vmax, 21) if vmax > 0 else 20
        cf = ax.contourf(X, Y, s['vorticity'], levels=levels, cmap='RdBu_r', extend='both')
        fig.colorbar(cf, ax=ax, label=r'$\omega_z$ (1/s)')
        ax.set_title(rf"Vorticity t={s['t']:.1f} s")
        for ax in axes[:, i]:
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_aspect('equal')
    fig.suptitle(f'Velocity Field & Vorticity{_sfx(meta)}', fontsize=14, y=1.02)
    save(fig, 'velocity', 'velocity_field_snapshots.png', bbox_tight=True)


# ─── Dalziel horizontal-slab spectra ─────────────────────────────────────────

def _band_tags(sdf):
    return [c[len('dalziel_beta_'):] for c in sdf.columns if c.startswith('dalziel_beta_')]


def _resolved(sdf, tag):
    c = f'dalziel_resolved_{tag}'
    if c not in sdf.columns:
        return True
    return bool(sdf[c].map(lambda v: str(v).strip().lower() in ('true', '1', '1.0')).all())


def _dalziel(sdf, meta, save, log):
    tags = _band_tags(sdf)
    if not tags:
        return
    # Fig-16 analogue: beta(tau), one curve per band
    fig, ax = new_figure((8, 6))
    colors, marks = {'10_25': 'C0', '10_50': 'C3'}, {'10_25': 'o', '10_50': 's'}
    for tag in tags:
        lo, hi = (float(v) for v in tag.split('_'))
        lbl = rf'${lo:.0f} \leq k/k_0 \leq {hi:.0f}$' + ('' if _resolved(sdf, tag) else r' (past $6\,dx$)')
        ax.plot(sdf['tau'], sdf[f'dalziel_beta_{tag}'], marker=marks.get(tag, '^'), ls='-', ms=6,
                color=colors.get(tag, 'C2'), label=lbl)
    for beta, fmt, kw, lbl in DALZIEL_REFS:
        ax.plot(2.0, beta, fmt, label=lbl, **kw)
    ax.axhline(-5 / 3, color='0.6', ls=':', alpha=0.6, label=r'$-5/3$ (Kolmogorov)')
    ax.axvline(2.0, color='0.6', lw=0.8, alpha=0.5)
    ax.set_xlabel(r'Time, $\tau$')
    ax.set_ylabel(r'Spectral slope, $\beta$')
    ax.set_ylim(-2.6, 0)
    ax.set_title(f'Dalziel horizontal concentration spectral slope{_sfx(meta)}', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save(fig, 'spectra', 'dalziel_slope_evolution.png')

    # Fig-15 analogue: P vs k/k0 at early / mid / late times, bands shaded
    n = len(sdf)
    idxs = sorted({0, n // 2, n - 1})
    fig, ax = new_figure((8, 6))
    bands = [tuple(float(v) for v in tag.split('_')) for tag in tags]
    drawn = 0
    for c, idx in zip(['C0', 'C1', 'C2'], idxs):
        row = sdf.iloc[idx]
        p = os.path.join(meta.analysis_dir, 'spectra', f"dalziel_pk_t{int(round(row['time'] * 1000))}.csv")
        if not os.path.isfile(p):
            log(f"  dalziel_spectra_snapshots.png: missing {os.path.basename(p)}")
            continue
        pk = pd.read_csv(p)
        kk, pw = pk['k_over_k0'].values, pk['power'].values
        if len(kk) == 0:
            continue
        fits = [spectra.fit_powerlaw_weighted(kk, pw, lo, hi) for lo, hi in bands]
        blab = ', '.join(rf'$\beta_{{{f["fit_k_k0_max"]:.0f}}}$={f["power_law_slope"]:.2f}' for f in fits)
        msk = pw > 0
        ax.loglog(kk[msk], pw[msk], '+', color=c, alpha=0.5, ms=4, label=rf"$\tau$={row['tau']:.2f} ({blab})")
        for f in fits:
            band = (kk >= f['fit_k_k0_min']) & (kk <= f['fit_k_k0_max']) & (pw > 0)
            if np.sum(band) > 2 and not np.isnan(f.get('power_law_intercept', np.nan)):
                kfit = kk[band]
                ax.loglog(kfit, 10 ** (f['power_law_slope'] * np.log10(kfit) + f['power_law_intercept']), '-',
                          color=c, lw=2, alpha=0.9)
        drawn += 1
    if not drawn:
        fig.clf()
        return
    for (lo, hi), a in zip(bands, [0.07, 0.04]):
        ax.axvspan(lo, hi, color='k', alpha=a)
    ax.set_xlabel(r'Wavenumber, $k/k_0$')
    ax.set_ylabel(r'Power, $P$')
    ax.set_title(f'Dalziel horizontal concentration spectrum{_sfx(meta)}', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    save(fig, 'spectra', 'dalziel_spectra_snapshots.png')


# ─── 2D radial spectra ───────────────────────────────────────────────────────

def _tke_mode(meta):
    s = meta.phases.get('3') or {}
    if s.get('tke') in ('uv', 'legacy'):
        return s['tke'], False
    if meta.assumed:
        return ('legacy' if meta.assumed == 'legacy' else 'uv'), True
    return None, False


def tke_label(meta):
    mode, assumed = _tke_mode(meta)
    txt = {'uv': r'TKE: $\frac{1}{2}(P_u+P_v)$', 'legacy': 'TKE (legacy: u only)',
           None: 'TKE (mode not recorded)'}[mode]
    return txt + (' (assumed)' if assumed else '')


def _spectrum_evolution(sdf, meta, save):
    if 'vof_slope' not in sdf.columns:
        return
    tau, tke = sdf['tau'], 'tke_slope' in sdf.columns
    tke_lbl = tke_label(meta)
    fig, axes = new_figure((14, 10), 2, 2)
    ax = axes[0, 0]
    ax.plot(tau, sdf['vof_slope'], 'bo-', lw=2, ms=5, label='VOF (F)')
    if tke:
        ax.plot(tau, sdf['tke_slope'], 'rs-', lw=2, ms=5, label=tke_lbl)
    # P(k) is mean power per mode: in a 2D wavenumber plane E(k) ~ k P(k), so
    # the E(k) slopes -5/3 and -3 appear as -8/3 and -4 here (README).
    ax.axhline(-8 / 3, color='k', ls='--', alpha=0.5, label=r'$-8/3$ ($E \propto k^{-5/3}$)')
    ax.axhline(-4, color='gray', ls=':', alpha=0.5, label=r'$-4$ ($E \propto k^{-3}$)')
    ax.set_ylabel('Slope of P(k) (mean power per mode)')
    ax.set_title('Power Law Slope Evolution')
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    ax.plot(tau, sdf['vof_dom_wavelength'], 'go-', lw=2, ms=5, label='VOF (F)')
    ax.set_ylabel('Dominant wavelength (m)')
    ax.set_title('Dominant Wavelength Evolution')
    ax.legend()
    ax = axes[1, 0]
    ax.semilogy(tau, sdf['vof_total_energy'], 'bo-', lw=2, ms=5, label='VOF (F)')
    if tke:
        ax.semilogy(tau, sdf['tke_total_energy'], 'rs-', lw=2, ms=5, label=tke_lbl)
    ax.set_ylabel('Total spectral energy')
    ax.set_title('Spectral Energy Evolution')
    ax.legend()
    ax = axes[1, 1]
    ax.plot(tau, sdf['vof_R2'], 'bo-', lw=2, ms=5, label='VOF (F)')
    if tke:
        ax.plot(tau, sdf['tke_R2'], 'rs-', lw=2, ms=5, label=tke_lbl)
    ax.set_ylabel(r'$R^2$')
    ax.set_title('Power Law Fit Quality')
    ax.legend()
    for ax in axes.flat:
        ax.set_xlabel(r'$\tau$')
        ax.grid(True)
    fig.suptitle(f'Power Spectrum Evolution (2D radial, full domain){_sfx(meta)}', fontsize=14)
    save(fig, 'spectra', 'spectrum_evolution.png')


def _radial_spectrum(meta, t, log):
    """(k, E_F) for time t: plotdata CSV, else recomputed from the case."""
    p = os.path.join(meta.analysis_dir, plotdata.PLOTDATA, f'radial_spectra_t{int(round(t * 1000))}.csv')
    if os.path.isfile(p):
        d = pd.read_csv(p)
        return d['k'].values, d['E_F'].values
    case = meta.case
    if case is None:
        return None
    snap = plotdata.nearest(case.snapshots, t)
    data = case.read_2d(snap, fields=('F',))
    s = spectra.field_spectrum_2d(data['f'], data['x'], data['y'], 'F', True, return_spectrum=True)
    return (s['k'], s['power']) if 'k' in s else None


def _spectrum_snapshots(sdf, meta, save, log):
    if 'vof_slope' not in sdf.columns:
        return
    n = len(sdf)
    for idx in (0, n // 2, n - 1):          # legacy [0, n//2, -1]
        row = sdf.iloc[idx]
        t, tau = float(row['time']), float(row['tau'])
        ks = _radial_spectrum(meta, t, log)
        if ks is None:
            log(f"  spectrum_t{t:.0f}.png: no plotdata/radial_spectra csv and no case; skipped")
            continue
        k, power = ks
        fig, ax = new_figure((10, 7))
        ax.loglog(k, power, 'b-', lw=2, alpha=0.8, label='P(k) -- VOF (mean and plane removed)')
        nk = len(k)
        lo, hi = max(1, nk // 4), min(nk - 1, 3 * nk // 4)      # spectra.spectrum_features window
        kw, pw = k[lo:hi], power[lo:hi]
        valid = pw > 0
        if hi > lo + 3 and np.sum(valid) > 2:
            slope, icpt = np.polyfit(np.log10(kw[valid]), np.log10(pw[valid]), 1)
            ax.loglog(kw, 10 ** (slope * np.log10(kw) + icpt), 'r--', lw=2,
                      label=rf'fit $k^{{{slope:.2f}}}$ (R$^2$={row["vof_R2"]:.3f}), middle half of bins')
        k_ref, p_ref = k[nk // 2], power[nk // 2]
        k53 = np.logspace(np.log10(k_ref * 0.3), np.log10(k_ref * 3), 20)
        ax.loglog(k53, p_ref * (k53 / k_ref) ** (-8 / 3), 'k:', lw=1.5, alpha=0.5,
                  label=r'$k^{-8/3}$ (P(k) of $E \propto k^{-5/3}$)')
        ax.set_xlabel('Wavenumber k (rad/m)', fontsize=13)
        ax.set_ylabel('P(k), mean power per mode', fontsize=13)
        ax.set_title(rf'Power Spectrum at $\tau$={tau:.3f} (t={t:.1f} s){_sfx(meta)}', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        save(fig, 'spectra', f'spectrum_t{t:.0f}.png')
