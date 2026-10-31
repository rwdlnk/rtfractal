"""Phase 2: multifractal spectrum at selected times (2D contours, 3D surfaces).

Outputs (same names and columns as the legacy analyzers; no plots):
  2D  analysis/multifractal/multifractal_summary.csv
          time, tau, h_H, D0, D1, D2, alpha_width, degree_multifractality
      analysis/multifractal/<parameter_|time_><t>/multifractal_results.csv
          q, tau, Dq, alpha, f_alpha, r_squared, tau_raw, Dq_raw
      analysis/multifractal/<parameter_|time_><t>/multifractal_parameters.csv
          Parameter, Value  (Time, D0, D1, D2, alpha_width, degree_multifractality,
                             D0_raw, D2_raw, convexity_adjustment, convexity_collapsed)
      (directory prefix parameter_ for SOLA-VOF, time_ for OpenFOAM/rtlbm, as legacy)
      + multifractal_scales.csv (box_size, n_occupied) in the same directory (new)
  3D  analysis/multifractal/multifractal_summary_3d.csv
          time, tau, D0, D1, D2, alpha_width, degree_multifractality
      analysis/multifractal/t<t:.2f>__multifractal_spectrum_3d.csv
          q, tau_q, D_q, alpha, f_alpha, R_squared, tau_q_raw, D_q_raw   (%.6f text)
      analysis/multifractal/t<t:.2f>__multifractal_params_3d.csv     (parameter, value)
      analysis/multifractal/t<t:.2f>__multifractal_scales_3d.csv     (delta, n_occupied)

Times: each target time (default 2, 4, ..., 18 s) maps to the nearest
snapshot; targets beyond the last snapshot + 0.5 s are dropped.  2D keeps
one row per target (as legacy); 3D merges targets that map to the same
snapshot.

Box sizes.  Legacy phase 2 reused the box sizes of phase 1's FITTED window
(read back from analysis/fractal/boxcount_*.csv), so that D_0 equals the
phase-1 D.  Here that window is recomputed in-process with the same offsets,
ladder and trim, which gives the same sizes without depending on files left
on disk:
  ladder='L8' (default): phase-1 ladder L/8 -> 2 dx (2D), L/8 -> 2 max(dx,dy,dz) (3D);
      if phase 1 has no fitted window the whole phase-1 ladder is used (2D)
      or the time is skipped (3D, no non-zero counts).
  ladder='legacy': phase-1 ladder L/2 (2D) or min(L, W)/2 (3D).  Where legacy
      phase 1 wrote no box-count file (2D: D is NaN; 3D: counting failed) the
      legacy fallbacks are reproduced: 2D extent/2 -> 0.001 m by 1.5; 3D the
      fractal3d surface-feature heuristic (suggest_parameters).
offsets='covering' (default) | 'legacy' applies to phase 2's counting as well.
"""
import csv
import os
import time

import numpy as np
import pandas as pd

from .. import fractal, interface, multifractal
from .phase1 import _ladder_div

MF_TIMES = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0)
LEGACY_2D_MIN_BOX = 0.001       # RTAnalyzer.compute_multifractal_spectrum default (absolute, metres)


def _clamp_times(times, t_max):
    return [t for t in times if t <= t_max + 0.5]


def _load_phase1_df(analysis_dir, df):
    if df is not None:
        return df
    p = os.path.join(analysis_dir, 'summary', 'temporal_results.csv')
    return pd.read_csv(p) if os.path.isfile(p) else None


# ----------------------------------------------------------------------- 2D
def box_sizes_2d(segs, x_centres, ladder='L8', offsets='covering', trim='r2', backend=None):
    """Phase-2 box sizes for one contour (see module docstring) -> (sizes, source)."""
    div = _ladder_div(ladder)
    max_box, min_box = float(x_centres[-1] - x_centres[0]) / div, float(x_centres[1] - x_centres[0]) * 2
    try:
        r = fractal.dimension_of_segments(segs, min_box, max_box, 1.5, backend, offsets, trim)
    except Exception:
        r = None
    if ladder == 'legacy':
        if r is not None and not np.isnan(r['dimension']) and len(r['box_sizes']) >= 3:
            return np.asarray(r['box_sizes']), 'phase1'
        extent, _ = multifractal.segments_bbox(segs)
        sizes, s = [], extent / 2
        while s >= LEGACY_2D_MIN_BOX:
            sizes.append(s)
            s /= 1.5
        return np.array(sizes), 'legacy-fallback'
    if r is None:
        raise ValueError('phase-1 box counting failed: no box sizes')
    return np.asarray(r['box_sizes']), 'phase1' if not np.isnan(r['dimension']) else 'phase1-ladder'


def run_2d(case, analysis_dir, times=MF_TIMES, q_values=multifractal.Q_DEFAULT, df=None, backend=None, log=print,
           offsets='covering', ladder='L8', trim='r2'):
    """Multifractal spectrum of the F = 0.5 contour at the target times.

    df: phase-1 temporal results (for h_H); default analysis/summary/temporal_results.csv,
    h_H = NaN if absent.  -> summary DataFrame."""
    mf_dir = os.path.join(analysis_dir, 'multifractal')
    os.makedirs(mf_dir, exist_ok=True)
    df = _load_phase1_df(analysis_dir, df)
    phys = case.physics
    times = _clamp_times(times, max(s.t for s in case.snapshots))
    if not times:
        log("WARNING: no multifractal times within the data range; skipping phase 2")
        return None
    xc = case.read_2d(case.snapshots[0], fields=('F',))['x'][:, 0]
    prefix = 'parameter_' if case.format == 'solavof' else 'time_'
    rows = []
    for target in times:
        snap = min(case.snapshots, key=lambda s: abs(s.t - target))
        t0 = time.time()
        log(f"  MF target t={target:.1f} -> actual t={snap.t:.3f}")
        try:
            data = case.read_2d(snap, fields=('F',))
            segs = interface.segments(interface.contours(data['f'], data['x'], data['y'], 0.5))
            if len(segs) == 0:
                log("    no interface segments; skipped")
                continue
            sizes, src = box_sizes_2d(segs, xc, ladder, offsets, trim, backend)
            log(f"    {len(sizes)} box sizes ({src}): {sizes.max():.6f} .. {sizes.min():.6f}")
            res = multifractal.multifractal_2d(segs, sizes, q_values, backend, offsets)
        except Exception as e:
            log(f"    ERROR: {e}")
            continue
        point_dir = os.path.join(mf_dir, f"{prefix}{snap.t}")
        os.makedirs(point_dir, exist_ok=True)
        pd.DataFrame({'q': res['q'], 'tau': res['tau'], 'Dq': res['Dq'], 'alpha': res['alpha'],
                      'f_alpha': res['f_alpha'], 'r_squared': res['r_squared'], 'tau_raw': res['tau_raw'],
                      'Dq_raw': res['Dq_raw']}).to_csv(os.path.join(point_dir, 'multifractal_results.csv'),
                                                       index=False)
        pd.DataFrame({'Parameter': ['Time', 'D0', 'D1', 'D2', 'alpha_width', 'degree_multifractality', 'D0_raw',
                                    'D2_raw', 'convexity_adjustment', 'convexity_collapsed'],
                      'Value': [snap.t, res['D0'], res['D1'], res['D2'], res['alpha_width'],
                                res['degree_multifractality'], res['D0_raw'], res['D2_raw'],
                                res['convexity_adjustment'], int(res['convexity_collapsed'])]}).to_csv(
            os.path.join(point_dir, 'multifractal_parameters.csv'), index=False)
        pd.DataFrame({'box_size': res['box_sizes'], 'n_occupied': res['n_occupied']}).to_csv(
            os.path.join(point_dir, 'multifractal_scales.csv'), index=False)
        h_H = np.nan
        if df is not None and 'h_total_geo_nondim' in df.columns:
            row = df.iloc[(df['time'] - snap.t).abs().argsort()[:1]]
            h_H = float(row['h_total_geo_nondim'].values[0])
        rows.append(dict(time=snap.t, tau=float(phys.nondim_time(snap.t)), h_H=h_H, D0=res['D0'], D1=res['D1'],
                         D2=res['D2'], alpha_width=res['alpha_width'],
                         degree_multifractality=res['degree_multifractality']))
        log(f"    D0={res['D0']:.4f}, D1={res['D1']:.4f}, D2={res['D2']:.4f}, "
            f"alpha_width={res['alpha_width']:.4f}  ({time.time() - t0:.1f}s)")
    if not rows:
        return None
    out = pd.DataFrame(rows, columns=['time', 'tau', 'h_H', 'D0', 'D1', 'D2', 'alpha_width',
                                      'degree_multifractality'])
    out.to_csv(os.path.join(mf_dir, 'multifractal_summary.csv'), index=False)
    log(f"Multifractal summary saved to {mf_dir}")
    return out


# ----------------------------------------------------------------------- 3D
def _legacy_suggested_cube_sizes(mesh):
    """Legacy 3D fallback scales (analyze_rt_3d.run_phase2 when phase 1 wrote
    no box-count file): fractal3d.surface_features.suggest_parameters on the
    surface features (connected components, convex hull, bounding box), then
    max_delta = initial_delta, min_delta = char_length/500.

    Component areas are summed per component in index order (legacy: BFS
    order); this only feeds threshold tests of the surface classification."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import ConvexHull

    bb, tri = mesh.bbox, mesh.triangles
    areas = mesh.triangle_areas()
    sa = np.sum(areas)
    w, h, d = bb.max_x - bb.min_x, bb.max_y - bb.min_y, bb.max_z - bb.min_z
    vol = w * h * d
    bbox_volume = vol if vol > 0 else 1e-10
    char_len = bb.characteristic_length

    # connected components over shared edges
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    tid = np.tile(np.arange(len(tri)), 3)
    _, inv = np.unique(e, axis=0, return_inverse=True)
    order = np.lexsort((tid, inv.ravel()))
    ei, ti = inv.ravel()[order], tid[order]
    same = ei[1:] == ei[:-1]
    g = coo_matrix((np.ones(same.sum()), (ti[:-1][same], ti[1:][same])), shape=(len(tri), len(tri)))
    n_comp, labels = connected_components(g, directed=False)
    comp_area = np.sort(np.bincount(labels, weights=areas, minlength=n_comp))[::-1]
    largest = comp_area[0] / sa if n_comp > 0 and sa > 0 else 1.0
    fragmentation = 1.0 - largest

    dims = sorted([w, h, d], reverse=True)
    max_dim, mid_dim, min_dim = (max(v, 1e-10) for v in dims)
    aspect_ratio = max_dim / min_dim
    sphericity = min((36 * np.pi * bbox_volume ** 2) ** (1 / 3) / max(sa, 1e-10), 1.0)
    try:
        if len(mesh.vertices) < 4:
            raise ValueError
        hull = ConvexHull(mesh.vertices)
        hull_area = hull.area
    except Exception:
        hull_area = sa
    compactness = sa / max(hull_area, 1e-10)
    roughness = sa / max(max(w * h, h * d, d * w), 1e-10)
    score = max(0.0, 0.3 * (compactness - 1.0) + 0.2 * fragmentation * 10 + 0.2 * (roughness - 2.0) / 10
                + 0.15 * np.log1p(n_comp) + 0.15 * (1 - sphericity))

    if n_comp > 10 or fragmentation > 0.5:
        kind = 'turbulent_mixing' if (roughness > 5.0 or compactness > 2.0) else 'highly_fragmented'
    elif score > 1.5 or compactness > 3.0 or roughness > 10.0:
        kind = 'complex'
    elif score > 0.5 or compactness > 1.5 or roughness > 4.0:
        kind = 'moderate'
    else:
        kind = 'smooth'
    div, factor, steps = {'smooth': (8, 1.6, 12), 'moderate': (10, 1.5, 15), 'complex': (15, 1.4, 18),
                          'highly_fragmented': (15, 1.4, 18), 'turbulent_mixing': (20, 1.3, 22)}[kind]
    initial = char_len / div
    if n_comp > 5:
        initial *= 0.8
        steps += 2
    if aspect_ratio > 5:
        steps += 2
    min_delta, sizes, dl = char_len / 500, [], initial
    for _ in range(steps):
        if dl < min_delta:
            break
        sizes.append(dl)
        dl /= factor
    return np.array(sizes), kind


def cube_sizes_3d(case, mesh, ladder='L8', offsets='covering', trim='r2', backend=None):
    """Phase-2 cube sizes for one surface (see module docstring) -> (sizes, source)."""
    p, g3 = case.params, case.grid
    initial = p['L'] / 8 if ladder == 'L8' else min(p['L'], p['W']) / _ladder_div(ladder)
    try:
        r = fractal.dimension_3d(mesh, initial=initial, factor=1.5, num_steps=20,
                                 min_delta=max(g3.dx, g3.dy, g3.dz) * 2, backend=backend, offsets=offsets, trim=trim)
        return np.asarray(r['box_sizes']), 'phase1'
    except ValueError:
        if ladder != 'legacy':
            raise
    sizes, kind = _legacy_suggested_cube_sizes(mesh)
    return sizes, f'legacy-fallback ({kind})'


def _f6(v):
    return 'NaN' if np.isnan(v) else f"{v:.6f}"


def _save_3d(res, mf_dir, label, measure):
    prefix = f"{label}_" if label else ""
    q = res['q']
    with open(os.path.join(mf_dir, f"{prefix}multifractal_spectrum_3d.csv"), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['q', 'tau_q', 'D_q', 'alpha', 'f_alpha', 'R_squared', 'tau_q_raw', 'D_q_raw'])
        for i in range(len(q)):
            w.writerow([f"{q[i]:.2f}"] + [_f6(res[k][i]) for k in ('tau', 'Dq', 'alpha', 'f_alpha', 'r_squared',
                                                                   'tau_raw', 'Dq_raw')])
    with open(os.path.join(mf_dir, f"{prefix}multifractal_params_3d.csv"), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['parameter', 'value'])
        w.writerow(['measure', measure])
        for k in ('D0', 'D1', 'D2', 'alpha_width', 'degree_multifractality', 'convexity_adjustment'):
            w.writerow([k, f"{res[k]:.6f}"])
        w.writerow(['convexity_collapsed', int(res['convexity_collapsed'])])
        w.writerow(['D0_raw', f"{res['D0_raw']:.6f}"])
        w.writerow(['D2_raw', f"{res['D2_raw']:.6f}"])
        w.writerow(['n_scales', len(res['box_sizes'])])
        w.writerow(['delta_min', f"{res['box_sizes'][-1]:.6f}"])
        w.writerow(['delta_max', f"{res['box_sizes'][0]:.6f}"])
    with open(os.path.join(mf_dir, f"{prefix}multifractal_scales_3d.csv"), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['delta', 'n_occupied'])
        for d, n in zip(res['box_sizes'], res['n_occupied']):
            w.writerow([f"{d:.6f}", int(n)])


def run_3d(case, analysis_dir, times=MF_TIMES, q_values=multifractal.Q_DEFAULT, measure='area', backend=None,
           log=print, offsets='covering', ladder='L8', trim='r2'):
    """Multifractal spectrum of the interface surface at the target times -> summary DataFrame."""
    mf_dir = os.path.join(analysis_dir, 'multifractal')
    os.makedirs(mf_dir, exist_ok=True)
    phys = case.physics
    times = _clamp_times(times, max(s.t for s in case.snapshots))
    if not times:
        log("WARNING: no multifractal times within the data range; skipping phase 2")
        return None
    picked = {}
    for target in times:
        snap = min(case.snapshots, key=lambda s: abs(s.t - target))
        picked[snap.t] = snap
        log(f"  MF target t={target:.1f} -> actual t={snap.t:.2f}")
    rows = []
    for t in sorted(picked):
        snap, t0 = picked[t], time.time()
        tau = float(phys.nondim_time(t))
        log(f"  Multifractal analysis at t={t:.2f} (tau={tau:.4f})")
        try:
            mesh = case.interface_mesh(snap)
            if mesh.n_triangles < 50:
                log(f"    only {mesh.n_triangles} triangles; skipped")
                continue
            sizes, src = cube_sizes_3d(case, mesh, ladder, offsets, trim, backend)
            log(f"    {len(sizes)} cube sizes ({src}): {sizes.max():.6f} .. {sizes.min():.6f}")
            res = multifractal.multifractal_3d(mesh, sizes, q_values, measure, backend, offsets)
        except Exception as e:
            log(f"    WARNING: multifractal failed at t={t:.2f}: {e}")
            continue
        _save_3d(res, mf_dir, f't{t:.2f}_', measure)
        rows.append(dict(time=t, tau=tau, D0=res['D0'], D1=res['D1'], D2=res['D2'], alpha_width=res['alpha_width'],
                         degree_multifractality=res['degree_multifractality']))
        log(f"    D0={res['D0']:.4f}  D1={res['D1']:.4f}  D2={res['D2']:.4f}  "
            f"dalpha={res['alpha_width']:.4f}  ({time.time() - t0:.1f}s)")
    if not rows:
        return None
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(mf_dir, 'multifractal_summary_3d.csv'), index=False)
    log(f"Multifractal summary saved to {mf_dir}")
    return out
