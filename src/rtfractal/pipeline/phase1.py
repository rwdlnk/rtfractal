"""Phase 1: mixing widths and fractal dimension at every snapshot.

Outputs (same layout and columns as the legacy analyzers):
    analysis/summary/temporal_results.csv
    analysis/profiles/alpha_profile_t<ms>.csv        (+ theta_profile_* in 3D)
    analysis/interfaces/interface_t<ms>.dat          (2D)
    analysis/fractal/boxcount_t<ms>.csv | boxcount_3d_t<t>.csv
"""
import os
import time

import numpy as np
import pandas as pd

from .. import fractal, interface, mixing

T_SKIP = 0.01          # snapshots before this are the flat initial interface
_NONDIM = ['ht_geo', 'hb_geo', 'h_total_geo', 'ht_stat', 'hb_stat', 'h_total_stat',
           'ht_int', 'hb_int', 'h_total_int', 'h_10', 'h_11', 'h_00', 'h_01']


def _level_col(lv):
    return f'F{int(round(lv * 10)):02d}'


def _dirs(analysis_dir, names):
    for n in names:
        os.makedirs(os.path.join(analysis_dir, n), exist_ok=True)


def run_2d(case, analysis_dir, contour_levels=(0.5,), backend=None, log=print, offsets='covering'):
    contour_levels = sorted(set(contour_levels))
    _dirs(analysis_dir, ['interfaces', 'fractal', 'summary', 'profiles'])
    phys, H, H0 = case.physics, case.physics.H, case.H0

    # Fixed, grid-tied ladder: L/2 down to 2 dx by 1.5, with L and dx measured
    # between cell centres (legacy convention: L here is L_domain - dx).
    first = case.read_2d(case.snapshots[0])
    xc = first['x'][:, 0]
    max_box, min_box, factor = float(xc[-1] - xc[0]) / 2, float(xc[1] - xc[0]) * 2, 1.5
    log(f"  box ladder: max={max_box:.6f}, min={min_box:.6f}, factor={factor}")

    rows = []
    for i, snap in enumerate(case.snapshots):
        tau = float(phys.nondim_time(snap.t))
        log(f"[{i + 1}/{len(case.snapshots)}] t={snap.t:.3f} s (tau={tau:.4f})")
        data = first if i == 0 else case.read_2d(snap)
        f_avg, yv = np.mean(data['f'], axis=0), data['y'][0, :]
        pd.DataFrame({'y_m': yv, 'alpha_mean': f_avg}).to_csv(
            os.path.join(analysis_dir, 'profiles', f'alpha_profile_t{snap.time_idx:05d}.csv'), index=False)

        if snap.t < T_SKIP:
            row = dict(time=snap.t, tau=tau, ht_geo=0.0, hb_geo=0.0, h_total_geo=0.0,
                       ht_stat=0.0, hb_stat=0.0, h_total_stat=0.0, ht_int=0.0, hb_int=0.0, h_total_int=0.0,
                       h_10=0.0, h_11=H / 2.0, h_00=H / 2.0, h_01=0.0,
                       fractal_dim=np.nan, fd_error=np.nan, fd_r_squared=np.nan, n_segments=0)
            for lv in contour_levels:
                if lv != 0.5:
                    row[f'fractal_dim_{_level_col(lv)}'] = row[f'fd_r_squared_{_level_col(lv)}'] = np.nan
            rows.append(row)
            continue

        cs05 = interface.contours(data['f'], data['x'], data['y'], 0.5)
        geo = mixing.geometric_width(cs05, H0)
        stat = mixing.statistical_width(f_avg, yv, H0)
        intg = mixing.integral_widths(f_avg, yv, H0)

        fd = dict(fractal.NAN_RESULT)
        n_segments, extra = 0, {}
        for lv in contour_levels:
            col = _level_col(lv)
            tag = '' if lv == 0.5 else f'{col}_'
            cs = cs05 if lv == 0.5 else interface.contours(data['f'], data['x'], data['y'], lv)
            segs = interface.segments(cs)
            if lv == 0.5:
                n_segments = len(segs)
            with open(os.path.join(analysis_dir, 'interfaces', f'interface_{tag}t{snap.time_idx:05d}.dat'), 'w') as fh:
                fh.write(f"# Interface (F={lv}) at t={snap.t:.6f} s (tau={tau:.6f})\n")
                fh.write(f"# {len(segs)} segments from {len(cs)} contour(s)\n# x1,y1 x2,y2\n")
                for s in segs:
                    fh.write(f"{s[0]:.7f},{s[1]:.7f} {s[2]:.7f},{s[3]:.7f}\n")
            try:
                r = fractal.dimension_of_segments(segs, min_box, max_box, factor, backend, offsets)
            except Exception as e:
                log(f"  WARNING: fractal dimension (F={lv}) failed: {e}")
                r = dict(fractal.NAN_RESULT)
            if lv == 0.5:
                fd = r
            else:
                extra[lv] = (r['dimension'], r['r_squared'])
            if not np.isnan(r['dimension']) and 'box_sizes' in r:
                pd.DataFrame({'box_size': r['box_sizes'], 'box_count': r['box_counts'],
                              'box_size_nondim': phys.nondim_box_size(r['box_sizes'])}).to_csv(
                    os.path.join(analysis_dir, 'fractal', f'boxcount_{tag}t{snap.time_idx:05d}.csv'), index=False)

        log(f"  h_geo={geo['h_total']:.5f}  h_stat={stat['h_total']:.5f}  h_int={intg['h_total']:.5f}  "
            f"D={fd['dimension']:.4f}+/-{fd['error']:.4f}  segs={n_segments}")
        row = dict(time=snap.t, tau=tau, ht_geo=geo['ht'], hb_geo=geo['hb'], h_total_geo=geo['h_total'],
                   ht_stat=stat['ht'], hb_stat=stat['hb'], h_total_stat=stat['h_total'],
                   ht_int=intg['ht'], hb_int=intg['hb'], h_total_int=intg['h_total'],
                   h_10=intg['h_10'], h_11=intg['h_11'], h_00=intg['h_00'], h_01=intg['h_01'],
                   fractal_dim=fd['dimension'], fd_error=fd['error'], fd_r_squared=fd['r_squared'],
                   n_segments=n_segments)
        for lv, (d, r2) in extra.items():
            row[f'fractal_dim_{_level_col(lv)}'], row[f'fd_r_squared_{_level_col(lv)}'] = d, r2
        rows.append(row)

    df = pd.DataFrame(rows)
    for c in _NONDIM:
        df[c + '_nondim'] = df[c] / H
    out = os.path.join(analysis_dir, 'summary', 'temporal_results.csv')
    df.to_csv(out, index=False)
    log(f"Temporal results saved to {out}")
    return df


def run_3d(case, analysis_dir, energy=True, backend=None, log=print, offsets='covering'):
    _dirs(analysis_dir, ['fractal', 'summary', 'profiles'])
    p, g3, phys = case.params, case.grid, case.physics
    H, H0 = p['H'], case.H0
    y = g3.y_centers
    rows = []
    for i, snap in enumerate(case.snapshots):
        t0 = time.time()
        tau = float(phys.nondim_time(snap.t))
        log(f"[{i + 1}/{len(case.snapshots)}] t = {snap.t:.2f} s  (tau = {tau:.4f})")
        row = {'time': snap.t, 'tau': tau}
        if snap.t < T_SKIP:
            row.update(ht_stat=0.0, hb_stat=0.0, h_total_stat=0.0, ht_int=0.0, hb_int=0.0, h_total_int=0.0,
                       h_10=0.0, h_11=H / 2.0, h_00=H / 2.0, h_01=0.0, youngs_W=0.0, mixing_efficiency=0.0,
                       fractal_dim=np.nan, fd_error=np.nan, fd_r_squared=np.nan, n_triangles=0,
                       surface_area=0.0, Theta=0.0, Theta_hat=0.0)
            if energy:
                row.update(mixing.reference_energy_3d(y, g3, H0, p['rho_heavy'], p['rho_light'], p['g']))
            pd.DataFrame({'y_m': y, 'alpha_mean': np.where(y >= H0, 1.0, 0.0)}).to_csv(
                os.path.join(analysis_dir, 'profiles', 'alpha_profile_t00000.csv'), index=False)
            rows.append(row)
            continue

        mix = None
        try:
            alpha = case.read_alpha_3d(snap)
            f_avg = np.mean(alpha, axis=(0, 2))
            st, it = mixing.statistical_width(f_avg, y, H0), mixing.integral_widths(f_avg, y, H0)
            mix = dict(ht_stat=st['ht'], hb_stat=st['hb'], h_total_stat=st['h_total'],
                       ht_int=it['ht'], hb_int=it['hb'], h_total_int=it['h_total'],
                       h_10=it['h_10'], h_11=it['h_11'], h_00=it['h_00'], h_01=it['h_01'])
            mix.update(mixing.youngs(f_avg, y))
            mm = mixing.molecular_mixing_3d(alpha, y)
            row['Theta'], row['Theta_hat'] = mm['Theta'], mm['Theta_hat']
            pd.DataFrame({'y_m': y, 'theta': mm['theta'], 'theta_hat': mm['theta_hat']}).to_csv(
                os.path.join(analysis_dir, 'profiles', f'theta_profile_t{snap.time_idx:05d}.csv'), index=False)
            if energy:
                try:
                    row.update(mixing.energy_budget_3d(alpha, case.read_U_3d(snap), y, g3,
                                                       p['rho_heavy'], p['rho_light'], p['g']))
                except Exception as e:
                    log(f"  WARNING: energy budget failed: {e}")
            pd.DataFrame({'y_m': y, 'alpha_mean': f_avg}).to_csv(
                os.path.join(analysis_dir, 'profiles', f'alpha_profile_t{snap.time_idx:05d}.csv'), index=False)
            del alpha
        except Exception as e:
            log(f"  WARNING: mixing computation failed: {e}")
        if mix is None:
            mix = {k: np.nan for k in ['ht_stat', 'hb_stat', 'h_total_stat', 'ht_int', 'hb_int', 'h_total_int',
                                       'h_10', 'h_11', 'h_00', 'h_01', 'youngs_W', 'mixing_efficiency']}
        row.update(mix)

        fd, n_tri, area, dproj, r2proj = dict(fractal.NAN_RESULT), 0, 0.0, np.nan, np.nan
        mesh = None
        try:
            mesh = case.interface_mesh(snap)
            n_tri, area = mesh.n_triangles, mesh.surface_area
            if n_tri >= 10:
                fd = fractal.dimension_3d(mesh, initial=min(p['L'], p['W']) / 2, factor=1.5, num_steps=20,
                                          min_delta=max(g3.dx, g3.dy, g3.dz) * 2, backend=backend, log=log,
                                          offsets=offsets)
                pd.DataFrame({'cube_size': fd['box_sizes'], 'cube_count': fd['box_counts'],
                              'cube_size_nondim': np.asarray(fd['box_sizes']) / H}).to_csv(
                    os.path.join(analysis_dir, 'fractal', f'boxcount_3d_t{snap.t:.2f}.csv'), index=False)
            else:
                log(f"  WARNING: only {n_tri} triangles, skipping fractal")
        except Exception as e:
            log(f"  WARNING: fractal dimension failed: {e}")
        try:
            if mesh is not None and n_tri >= 10:
                dproj, r2proj = fractal.projected_dimension(mesh, g3.dx, g3.dy)
        except Exception as e:
            log(f"  WARNING: projected D failed: {e}")
        row.update(fractal_dim=fd['dimension'], fd_error=fd['error'], fd_r_squared=fd['r_squared'],
                   n_triangles=n_tri, surface_area=area, fractal_dim_proj=dproj, fd_r_squared_proj=r2proj)
        log(f"  h_int={row.get('h_total_int', np.nan):.5f}  D={fd['dimension']:.4f}+/-{fd['error']:.4f}  "
            f"Dproj={dproj:.4f}  tri={n_tri}  ({time.time() - t0:.1f}s)")
        rows.append(row)

    df = pd.DataFrame(rows)
    for c in _NONDIM[3:]:
        if c in df.columns:
            df[c + '_nondim'] = df[c] / H
    if {'PE', 'KE'}.issubset(df.columns) and df['PE'].notna().any():
        _energy_columns(df, p)
    out = os.path.join(analysis_dir, 'summary', 'temporal_results.csv')
    df.to_csv(out, index=False)
    log(f"Temporal results saved to {out}")
    return df


def _energy_columns(df, p):
    """Cumulative dissipation PE(0) - PE - KE, its rate, and mixing efficiencies."""
    PE0, BPE0, APE0 = df['PE'].iloc[0], df['BPE'].iloc[0], df['APE'].iloc[0]
    df['PE_released'] = PE0 - df['PE']
    df['E_diss'] = df['PE_released'] - df['KE']
    with np.errstate(divide='ignore', invalid='ignore'):
        df['diss_fraction'] = df['E_diss'] / df['PE_released']
        df['ke_fraction'] = df['KE'] / df['PE_released']
        df['mixing_eff_bpe'] = (df['BPE'] - BPE0) / df['PE_released']        # share of released energy
        df['eta_dalziel'] = (df['BPE'] - BPE0) / (APE0 - df['APE']).abs()    # Dalziel et al. 2008 Eq. 23
    t, e = df['time'].values.astype(float), df['E_diss'].values.astype(float)
    df['diss_rate'] = np.gradient(e, t) if len(t) > 2 and np.isfinite(e).sum() > 2 else np.nan
    scale = 0.5 * (p['rho_heavy'] + p['rho_light']) * (p['A'] * p['g']) ** 1.5 * p['H'] ** 0.5 * p['L'] * p['H'] * p['W']
    df['diss_rate_star'] = df['diss_rate'] / scale if scale > 0 else np.nan
