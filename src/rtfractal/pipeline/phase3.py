"""Phase 3: power spectra and velocity statistics at selected times.

Ported from FractalParameterAI ``analyze_rt.py`` (run_phase3 /
_run_phase3_openfoam, which are identical apart from the reader) and
``analyze_rt_3d.py`` (run_phase3).  Data outputs only; the plots are drawn
from them by rtfractal.plots.phase3.  Outputs (same names and columns as the
legacy analyzers):

2D  analysis/spectra/dalziel_pk_t<ms>.csv        k_over_k0, power
    analysis/spectra/spectrum_summary.csv        one row per time
    analysis/velocity/velocity_statistics.csv    one row per time (if U, V)
3D  analysis/spectra/spectrum_3d_t<t:.2f>.csv     k_over_k0, P
    analysis/spectra/tke_spectrum_3d_t<t:.2f>.csv k_over_k0, E_u, E_v, E_w, E_tke
    analysis/spectra/spectrum_summary_3d.csv, tke_spectrum_summary_3d.csv
    analysis/velocity/velocity_profile_t<t:.2f>.csv, velocity_summary_3d.csv
Additive (for the plots, see rtfractal.plotdata):
    analysis/rtfractal_run.json                    phase-3 times
2D  analysis/plotdata/radial_spectra_t<ms>.csv    k, S_F[, S_u, S_v, S_tke]  (S = E or P, the convention)
    analysis/plotdata/velocity_t<idx>.npz         |V| and vorticity at the picture times

Radial-spectrum convention (spectrum='E' default, 'P' legacy; see
spectra.py).  With the default, names say E(k) explicitly:
2D  spectrum_summary.csv: <field>_slope_E, <field>_R2_E,
    <field>_dom_wavelength_E, <field>_total_energy (exact variance; TKE for
    tke); the dalziel_* columns are unchanged (1-D spectrum).
3D  spectra/spectrum_E_3d_t<t>.csv (k_over_k0, E),
    spectra/tke_spectrum_E_3d_t<t>.csv (k_over_k0, E_u, E_v, E_w, E_tke),
    spectrum_summary_3d.csv vof_slope_E_<band>, vof_r2_E_<band>,
    tke_spectrum_summary_3d.csv <comp>_slope_E_<band>, <comp>_r2_E_<band>.
With spectrum='P' the legacy names above are written (P(k) values, including
the legacy 'E_*' columns of tke_spectrum_3d_t<t>.csv).

Time selection follows the legacy code:
- target times above (last snapshot time + 0.5 s) are dropped;
- 2D uses the nearest snapshot to each target (so two targets can pick the
  same snapshot), and reports the snapshot's time;
- 3D reads the time directory named exactly like the target
  (str(int(t)) for whole numbers, else f"{t}") and reports the target time.
  A target with no such directory is skipped.
"""
import os
import time as _time

import numpy as np
import pandas as pd

from .. import plotdata, spectra, velocity
from ..io.case import Snapshot

DEFAULT_TIMES_2D = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0)          # legacy --mf-times
DEFAULT_TIMES_3D = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)    # legacy --spec-times


def _dirs(analysis_dir):
    sd, vd = os.path.join(analysis_dir, 'spectra'), os.path.join(analysis_dir, 'velocity')
    os.makedirs(sd, exist_ok=True)
    os.makedirs(vd, exist_ok=True)
    return sd, vd


def clamp_times(case, times):
    """Legacy clamp: keep targets <= last snapshot time + 0.5 s."""
    max_time = max(s.t for s in case.snapshots)
    return [t for t in times if t <= max_time + 0.5]


# ─── 2D ──────────────────────────────────────────────────────────────────────

def spectrum_summary_row(sr, spectrum='E'):
    """One spectrum_summary.csv row (legacy column order) from a per-time
    result.  spectrum='E' suffixes the radial-spectrum slope, R^2 and
    dominant-wavelength columns with _E; 'P' gives the legacy names."""
    sfx = '' if spectrum == 'P' else '_E'
    row = {'time': sr['time'], 'tau': sr['tau']}
    if 'dalziel_spectrum' in sr:
        ds = sr['dalziel_spectrum']
        row['dalziel_n_slab_rows'] = ds['n_slab_rows']
        row['dalziel_k_6dx'] = ds.get('k_6dx', np.nan)
        for bk in ds.get('band_keys', []):
            f = ds[bk]
            tag = bk.replace('fit_', '')
            row[f'dalziel_beta_{tag}'] = f['power_law_slope']
            row[f'dalziel_R2_{tag}'] = f['power_law_r_squared']
            row[f'dalziel_resolved_{tag}'] = f.get('fully_resolved', True)
    for key in ['vof_spectrum', 'u_velocity_spectrum', 'v_velocity_spectrum', 'tke_spectrum']:
        if key in sr:
            prefix = key.replace('_spectrum', '')
            row[f'{prefix}_slope{sfx}'] = sr[key]['power_law_slope']
            row[f'{prefix}_R2{sfx}'] = sr[key]['power_law_r_squared']
            row[f'{prefix}_dom_wavelength{sfx}'] = sr[key]['dominant_wavelength']
            row[f'{prefix}_total_energy'] = sr[key]['total_energy']
    return row


def run_2d(case, analysis_dir, times=DEFAULT_TIMES_2D, log=print, tke='uv', spectrum='E'):
    """Dalziel slab spectrum, 2D radial spectra of F/u/v, the TKE spectrum
    and whole-field velocity statistics at the snapshots nearest `times`.
    tke='uv' (default): spectrum of 1/2(u'^2 + v'^2); tke='legacy': the
    legacy u-only spectrum (see spectra.tke_spectrum_2d).  spectrum='E'
    (default) or 'P' selects the radial-spectrum convention (module notes).

    Returns dict(summary=DataFrame, velocity=DataFrame or None,
    results=[per-time dict with 'time', 'tau', 'dalziel_spectrum',
    'vof_spectrum'[, 'u_velocity_spectrum', 'v_velocity_spectrum',
    'tke_spectrum', 'velocity']]); the per-time spectra carry their k/P
    arrays for plotting.  None if no target time is within the data.
    """
    spectra_dir, velocity_dir = _dirs(analysis_dir)
    phys = case.physics
    times = clamp_times(case, times)
    if not times:
        log("WARNING: No spectrum times within data range. Skipping Phase 3.")
        return None

    # Slab anchored at the initial interface, taken as the mid-height of the
    # first snapshot's cell centres; "H" is the cell-centre extent (H - dy).
    s0 = min(case.snapshots, key=lambda s: s.t)
    y0v = case.read_2d(s0, fields=('F',))['y'][0, :]
    H_domain = float(y0v.max() - y0v.min())
    y0_init = 0.5 * (float(y0v.min()) + float(y0v.max()))
    log(f"  Dalziel slab anchored at initial interface y0={y0_init:.4f} m, "
        f"H={H_domain:.4f} m, slab=[{y0_init - 0.1 * H_domain:.4f}, {y0_init:.4f}] m")

    # velocity_field_snapshots.png pictures (legacy times, from the snapshots actually used)
    used = [min(case.snapshots, key=lambda s: abs(s.t - tt)) for tt in times]
    pictures = {id(plotdata.nearest(case.snapshots, tt)): None
                for tt in plotdata.velocity_snapshot_targets([s.t for s in used])}

    results, vel_rows = [], []
    for target_t in times:
        snap = min(case.snapshots, key=lambda s: abs(s.t - target_t))
        data = case.read_2d(snap, fields=('F', 'U', 'V'))
        t = snap.t
        tau = float(phys.nondim_time(t))
        if id(snap) in pictures and pictures[id(snap)] is None:
            plotdata.save_velocity_snapshot(analysis_dir, data, t, tau, snap.time_idx)
            pictures[id(snap)] = True
        log(f"\n  t={t:.3f} s (tau={tau:.4f})")
        sr = {'time': t, 'tau': tau}

        ds = spectra.dalziel_spectrum_2d(data['f'], data['x'], data['y'], y0_init, H_domain, field_name='F')
        sr['dalziel_spectrum'] = ds
        bands = []
        for bk in ds.get('band_keys', []):
            f = ds[bk]
            flag = '' if f.get('fully_resolved', True) else '*'
            bands.append(f"[{f['fit_k_k0_min']:.0f}-{f['fit_k_k0_max']:.0f}{flag}] "
                         f"beta={f['power_law_slope']:.3f} (R2={f['power_law_r_squared']:.2f})")
        log(f"    Dalziel spectrum ({ds['n_slab_rows']} slab rows): " + "  ".join(bands)
            + "   (* = past 6dx limit)")
        if len(ds.get('k_over_k0', [])) > 0:
            pd.DataFrame({'k_over_k0': ds['k_over_k0'], 'power': ds['power']}).to_csv(
                os.path.join(spectra_dir, f'dalziel_pk_t{int(round(t * 1000))}.csv'), index=False)

        vof = spectra.field_spectrum_2d(data['f'], data['x'], data['y'], 'F', True, return_spectrum=True,
                                        spectrum=spectrum)
        sr['vof_spectrum'] = vof
        log(f"    [2D-radial] slope={vof['power_law_slope']:.3f}, R2={vof['power_law_r_squared']:.3f}, "
            f"lambda_dom={vof['dominant_wavelength']:.4f}")

        if 'U' in data and 'V' in data:
            u_spec = spectra.field_spectrum_2d(data['U'], data['x'], data['y'], 'u', True, return_spectrum=True,
                                               spectrum=spectrum)
            v_spec = spectra.field_spectrum_2d(data['V'], data['x'], data['y'], 'v', True, return_spectrum=True,
                                               spectrum=spectrum)
            sr['u_velocity_spectrum'], sr['v_velocity_spectrum'] = u_spec, v_spec
            sr['tke_spectrum'] = spectra.tke_spectrum_2d(data['U'], data['V'], data['x'], data['y'],
                                                         u_spec=u_spec, v_spec=v_spec, mode=tke,
                                                         return_spectrum=True, spectrum=spectrum)
            log(f"    TKE spectrum: slope={sr['tke_spectrum']['power_law_slope']:.3f}, "
                f"R2={sr['tke_spectrum']['power_law_r_squared']:.3f}")
            vs = velocity.velocity_stats_2d(data['U'], data['V'])
            sr['velocity'] = vs
            vel_rows.append({'time': t, 'tau': tau, **{k: vs[k] for k in (
                'u_mean', 'v_mean', 'u_rms', 'v_rms', 'TKE', 'reynolds_stress', 'turbulence_intensity',
                'velocity_mag_max', 'velocity_mag_mean')}})
            log(f"    Velocity: u_rms={vs['u_rms']:.4e}, v_rms={vs['v_rms']:.4e}, "
                f"TKE={vs['TKE']:.4e}, |V|_max={vs['velocity_mag_max']:.4e}")
        plotdata.save_radial_spectra(analysis_dir, sr, spectrum)
        results.append(sr)

    for snap in case.snapshots:                 # picture times not among the phase-3 times
        if id(snap) in pictures and pictures[id(snap)] is None:
            plotdata.save_velocity_snapshot(analysis_dir, case.read_2d(snap, fields=('F', 'U', 'V')), snap.t,
                                            float(phys.nondim_time(snap.t)), snap.time_idx)

    vel_df = None
    if vel_rows:
        vel_df = pd.DataFrame(vel_rows)
        vel_df.to_csv(os.path.join(velocity_dir, 'velocity_statistics.csv'), index=False)
    summary = pd.DataFrame([spectrum_summary_row(sr, spectrum) for sr in results])
    out = os.path.join(spectra_dir, 'spectrum_summary.csv')
    summary.to_csv(out, index=False)
    log(f"Spectrum summary saved to {out}")
    plotdata.record_run(analysis_dir, case, 3, times=list(times), tke=tke, spectrum=spectrum)
    return {'summary': summary, 'velocity': vel_df, 'results': results}


# ─── 3D ──────────────────────────────────────────────────────────────────────

def legacy_time_dir(case_path, t):
    """Time directory the legacy 3D analyzer reads for target time t."""
    return os.path.join(case_path, str(int(t)) if t == int(t) else f"{t}")


def run_3d(case, analysis_dir, times=DEFAULT_TIMES_3D, log=print, spectrum='E'):
    """Slab concentration spectrum, slab velocity/TKE spectra and horizontal
    velocity statistics (profiles + values at H0) at each target time.

    Returns dict(spectrum=, tke=, velocity= DataFrames or None, results=[per-time
    dict with 'time', 'tau' and 'spectrum', 'velocity', 'tke' when computed]).
    None if no target time is within the data.
    """
    spec_dir, vel_dir = _dirs(analysis_dir)
    p, g3, phys = case.params, case.grid, case.physics
    H, H0 = p['H'], case.H0
    y = g3.y_centers
    decomposed = os.path.isdir(os.path.join(case.path, 'processor0'))
    times = clamp_times(case, times)
    if not times:
        log("WARNING: No Phase 3 times within data range. Skipping.")
        return None

    spectra._check_convention(spectrum)
    sfx, fsfx = ('', '') if spectrum == 'P' else ('_E', '_E')
    spec_rows, tke_rows, vel_rows, results = [], [], [], []
    for target_t in times:
        t0 = _time.time()
        tau = float(phys.nondim_time(target_t))
        log(f"\n  Phase 3: t={target_t:.2f} s (tau={tau:.4f})")
        res = {'time': target_t, 'tau': tau}
        try:
            snap = Snapshot(target_t, int(round(target_t * 1000)), legacy_time_dir(case.path, target_t))
            alpha = None
            if os.path.exists(os.path.join(snap.path, 'alpha.water')):
                try:
                    alpha = case.read_alpha_3d(snap)
                except Exception as e:
                    log(f"  WARNING: could not read alpha.water: {e}")
            else:
                log(f"  WARNING: no {snap.path}/alpha.water; skipping t={target_t}"
                    + (" (decomposed case: run reconstructPar)" if decomposed else ""))
            if alpha is not None:
                spec = spectra.horizontal_spectrum_3d(alpha, y, H0, H, g3.dx, g3.dz, spectrum=spectrum)
                if spec is not None:
                    res['spectrum'] = spec
                    row = {'time': target_t, 'tau': tau, 'n_slab_rows': spec['n_slab_rows']}
                    for band in ['10_25', '10_50']:
                        row[f'vof_slope{sfx}_{band}'] = spec.get(f'slope_{band}', np.nan)
                        row[f'vof_r2{sfx}_{band}'] = spec.get(f'r2_{band}', np.nan)
                    spec_rows.append(row)
                    pd.DataFrame({'k_over_k0': spec['k_over_k0'], spectrum: spec[spectrum]}).to_csv(
                        os.path.join(spec_dir, f'spectrum{fsfx}_3d_t{target_t:.2f}.csv'), index=False)
                    log(f"    spectrum {spectrum}(k): slope_10_25={row[f'vof_slope{sfx}_10_25']:.3f}, "
                        f"slope_10_50={row[f'vof_slope{sfx}_10_50']:.3f}")
                del alpha

            U = None
            if os.path.exists(os.path.join(snap.path, 'U')):
                try:
                    U = case.read_U_3d(snap)
                except Exception as e:
                    log(f"  WARNING: could not read U: {e}")
            if U is not None:
                vel = velocity.velocity_stats_3d(U, y, H0)
                res['velocity'] = vel
                vel_rows.append({'time': target_t, 'tau': tau, **{k: vel[k] for k in (
                    'u_rms_centreline', 'v_rms_centreline', 'w_rms_centreline', 'anisotropy_v_over_u',
                    'u_skew_centreline', 'v_skew_centreline', 'w_skew_centreline',
                    'u_kurt_centreline', 'v_kurt_centreline', 'w_kurt_centreline')}})
                pd.DataFrame({'y_m': vel['y_centers'], **{k: vel[k] for k in (
                    'u_rms', 'v_rms', 'w_rms', 'u_skew', 'v_skew', 'w_skew', 'u_kurt', 'v_kurt', 'w_kurt')}}).to_csv(
                    os.path.join(vel_dir, f'velocity_profile_t{target_t:.2f}.csv'), index=False)
                log(f"    velocity: v'/u'={vel['anisotropy_v_over_u']:.2f}, "
                    f"v_rms={vel['v_rms_centreline']:.4f} m/s")

                tke = spectra.tke_spectrum_3d(U, y, H0, H, g3.dx, g3.dz, spectrum=spectrum)
                if tke is not None:
                    res['tke'] = tke
                    trow = {'time': target_t, 'tau': tau, 'n_slab_rows': tke['n_slab_rows']}
                    for comp in ('u', 'v', 'w', 'tke'):
                        for band in ('10_25', '10_50', '5_50'):
                            trow[f'{comp}_slope{sfx}_{band}'] = tke.get(f'{comp}_slope_{band}', np.nan)
                            trow[f'{comp}_r2{sfx}_{band}'] = tke.get(f'{comp}_r2_{band}', np.nan)
                    tke_rows.append(trow)
                    # Legacy wrote P(k) under 'E_*' names; keep that only for spectrum='P'.
                    pd.DataFrame({'k_over_k0': tke['k_over_k0'],
                                  **{f'E_{c}': tke[f'{spectrum}_{c}'] for c in ('u', 'v', 'w', 'tke')}}).to_csv(
                        os.path.join(spec_dir, f'tke_spectrum{fsfx}_3d_t{target_t:.2f}.csv'), index=False)
                    ref = '[KO -1.67, BO -2.20]' if spectrum == 'E' else '[P(k): KO -2.67, BO -3.20]'
                    log(f"    TKE spectrum {spectrum}(k): slope_10_50={trow[f'tke_slope{sfx}_10_50']:.3f} "
                        f"(R2={trow[f'tke_r2{sfx}_10_50']:.2f})  {ref}")
                del U
        except Exception as e:
            log(f"  WARNING: Phase 3 failed at t={target_t:.2f}: {e}")
        results.append(res)
        log(f"    ({_time.time() - t0:.1f}s)")

    out = {'spectrum': None, 'tke': None, 'velocity': None, 'results': results}
    for key, rows, d, name in (('spectrum', spec_rows, spec_dir, 'spectrum_summary_3d.csv'),
                               ('velocity', vel_rows, vel_dir, 'velocity_summary_3d.csv'),
                               ('tke', tke_rows, spec_dir, 'tke_spectrum_summary_3d.csv')):
        if rows:
            out[key] = pd.DataFrame(rows)
            out[key].to_csv(os.path.join(d, name), index=False)
            log(f"  Wrote {name} ({len(rows)} timesteps)")
    plotdata.record_run(analysis_dir, case, 3, times=list(times), spectrum=spectrum)
    return out
