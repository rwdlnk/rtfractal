"""Small additive data files that let `rtfractal plot` redraw every figure
from an analysis folder without recomputing (and, where possible, without
the case).  None of these files exists in the legacy layout, and none of
them changes a legacy file.

    analysis/rtfractal_run.json
        case metadata (format, dim, grid, physics, H0) and, per phase, the
        settings it ran with (box counting: offsets / ladder / trim; times),
        so plots can say which box-counting settings produced D, D_q, f(alpha).
    analysis/plotdata/field_t<idx>.npz          2D, phase 1
        F at the interface_snapshots.png times (x, y, f, t, tau, H0, time_idx).
    analysis/plotdata/velocity_t<idx>.npz       2D, phase 3
        |V| and vorticity at the velocity_field_snapshots.png times.
    analysis/plotdata/radial_spectra_t<ms>.csv  2D, phase 3
        the binned radial spectra the slopes are fitted to: k [rad/m], E_F,
        E_u, E_v, E_tke (velocity columns only if the case has velocity;
        E_tke is 1/2(P_u+P_v) with --tke uv, the u spectrum with --tke legacy).

Fields larger than MAX_PLOT_CELLS along an axis are stored with a stride
(every s-th cell), float32: they are for pictures only.  Derived fields
(|V|, vorticity) are computed at full resolution first.

Nothing here imports matplotlib.
"""
import datetime
import json
import math
import os

import numpy as np
import pandas as pd

from . import __version__

RUN_JSON = 'rtfractal_run.json'
PLOTDATA = 'plotdata'
MAX_PLOT_CELLS = 1000

# Legacy picture times (analyze_rt.py _plot_phase1 / _plot_phase3).
INTERFACE_SNAPSHOT_TIMES = (2.0, 5.0, 8.0, 10.0, 14.0, 18.0)
VELOCITY_SNAPSHOT_TIMES = (2.0, 8.0, 14.0, 18.0)


# ─── run metadata and settings ───────────────────────────────────────────────

def case_meta(case):
    """JSON-able description of a case (what the plots need from it)."""
    p = case.physics
    if case.dim == 3:
        g = case.grid
        grid = [g.nx, g.ny, g.nz]
    else:
        grid = [int(case.nx), int(case.ny)]
    return {'path': case.path, 'format': case.format, 'dim': case.dim, 'grid': grid, 'H0': float(case.H0),
            'physics': {'A': float(p.A), 'g': float(p.g), 'H': float(p.H), 'L': float(p.L),
                        'W': None if p.W is None else float(p.W)}}


def read_run(analysis_dir):
    """Contents of rtfractal_run.json, or None."""
    path = os.path.join(analysis_dir, RUN_JSON)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def record_run(analysis_dir, case, phase, **settings):
    """Merge this phase's settings into rtfractal_run.json (other phases kept)."""
    os.makedirs(analysis_dir, exist_ok=True)
    run = read_run(analysis_dir) or {}
    run['case'] = case_meta(case)
    settings = {k: (list(v) if isinstance(v, (tuple, np.ndarray)) else v) for k, v in settings.items()}
    settings['rtfractal_version'] = __version__
    settings['written'] = datetime.datetime.now().isoformat(timespec='seconds')
    run.setdefault('phases', {})[str(phase)] = settings
    path = os.path.join(analysis_dir, RUN_JSON)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(run, fh, indent=1, default=float)
    os.replace(tmp, path)


# ─── picture-time selection (legacy rules) ───────────────────────────────────

def nearest(snapshots, t):
    return min(snapshots, key=lambda s: abs(s.t - t))


def interface_snapshot_targets(all_times):
    """Legacy: 2, 5, 8, 10, 14, 18 s up to the last time, else 6 evenly spaced."""
    t_max = max(all_times)
    sel = [t for t in INTERFACE_SNAPSHOT_TIMES if t <= t_max]
    return sel if sel else np.linspace(min(all_times), t_max, 6).tolist()


def velocity_snapshot_targets(phase3_times):
    """Legacy: 2, 8, 14, 18 s up to the last phase-3 time + 0.5 s, else first and last."""
    t_max = max(phase3_times)
    sel = [t for t in VELOCITY_SNAPSHOT_TIMES if t <= t_max + 0.5]
    return sel if sel else [phase3_times[0], phase3_times[-1]]


# ─── 2D field snapshots ──────────────────────────────────────────────────────

def stride_for(shape):
    return max(1, math.ceil(max(shape) / MAX_PLOT_CELLS))


def field_snapshot(data, t, tau, H0, time_idx):
    """Dict for interface_snapshots.png from a read_2d() result."""
    s = stride_for(data['f'].shape)
    return {'x': data['x'][::s, 0].astype(np.float64), 'y': data['y'][0, ::s].astype(np.float64),
            'f': data['f'][::s, ::s].astype(np.float32), 't': float(t), 'tau': float(tau), 'H0': float(H0),
            'time_idx': int(time_idx), 'stride': s}


def velocity_snapshot(data, t, tau, time_idx):
    """Dict for velocity_field_snapshots.png: |V| and the legacy vorticity
    dV/dx - dU/dy (np.gradient, uniform spacing) at full resolution, then strided."""
    U, V = data['U'], data['V']
    vel_mag = np.sqrt(U ** 2 + V ** 2)
    dx = data['x'][1, 0] - data['x'][0, 0]
    dy = data['y'][0, 1] - data['y'][0, 0]
    vort = np.gradient(V, dx, axis=0) - np.gradient(U, dy, axis=1)
    s = stride_for(U.shape)
    return {'x': data['x'][::s, 0].astype(np.float64), 'y': data['y'][0, ::s].astype(np.float64),
            'vel_mag': vel_mag[::s, ::s].astype(np.float32), 'vorticity': vort[::s, ::s].astype(np.float32),
            't': float(t), 'tau': float(tau), 'time_idx': int(time_idx), 'stride': s,
            # percentile taken at full resolution, as legacy
            'vort_p98': float(np.percentile(np.abs(vort), 98))}


def _save_npz(analysis_dir, name, d):
    os.makedirs(os.path.join(analysis_dir, PLOTDATA), exist_ok=True)
    np.savez_compressed(os.path.join(analysis_dir, PLOTDATA, name), **d)


def save_field_snapshot(analysis_dir, data, t, tau, H0, time_idx):
    _save_npz(analysis_dir, f'field_t{time_idx:05d}.npz', field_snapshot(data, t, tau, H0, time_idx))


def save_velocity_snapshot(analysis_dir, data, t, tau, time_idx):
    if 'U' in data and 'V' in data:
        _save_npz(analysis_dir, f'velocity_t{time_idx:05d}.npz', velocity_snapshot(data, t, tau, time_idx))


def load_npz(path):
    with np.load(path) as z:
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def list_npz(analysis_dir, kind):
    """{time_idx: path} of plotdata/<kind>_t*.npz."""
    d = os.path.join(analysis_dir, PLOTDATA)
    out = {}
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.startswith(kind + '_t') and f.endswith('.npz'):
                try:
                    out[int(f[len(kind) + 2:-4])] = os.path.join(d, f)
                except ValueError:
                    pass
    return out


# ─── 2D radial spectra ───────────────────────────────────────────────────────

def save_radial_spectra(analysis_dir, sr):
    """k and E(k) of the F (and u, v) radial spectra of one phase-3 result."""
    vof = sr.get('vof_spectrum', {})
    if 'k' not in vof or len(vof['k']) == 0:
        return
    cols = {'k': vof['k'], 'E_F': vof['power']}
    for key, name in (('u_velocity_spectrum', 'E_u'), ('v_velocity_spectrum', 'E_v'), ('tke_spectrum', 'E_tke')):
        s = sr.get(key, {})
        if 'k' in s and len(s['k']) == len(vof['k']) and np.array_equal(s['k'], vof['k']):
            cols[name] = s['power']
    os.makedirs(os.path.join(analysis_dir, PLOTDATA), exist_ok=True)
    pd.DataFrame(cols).to_csv(os.path.join(analysis_dir, PLOTDATA,
                                           f'radial_spectra_t{int(round(sr["time"] * 1000))}.csv'), index=False)
