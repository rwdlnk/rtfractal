"""Multifractal spectra (D_q, tau(q), f(alpha)) of 2D contours and 3D surfaces.

Method (both dimensions): at every box size the grid-optimised measure is
taken (the origin offset with the fewest occupied boxes, as for D in
phase 1), normalised to probabilities p_i, and
    Z_q(eps) = sum p_i^q,   tau(q) = slope of ln Z_q vs ln eps (all sizes, no trim)
    D_q = tau(q)/(q - 1),   D_1 = central difference of tau at q = 1.
tau(q) over q >= 0 is then projected onto the nearest convex function with
tau(0) and tau(1) pinned (SLSQP), and f(alpha) comes from the Legendre
transform of a cubic interpolating spline of tau(q).

The 2D and 3D legacy implementations differ in small ways; both are kept
exactly (spectrum_2d = RTAnalyzer.compute_multifractal_spectrum in
FractalParameterAI/rt_analyzer/rt_analyzer.py, spectrum_3d =
fractal3d.multifractal_3d.MultifractalAnalyzer3D.compute_multifractal_spectrum):

                        2D                              3D
  q == 1 tolerance      1e-6                            1e-8
  Z_q terms             all p_i                         p_i > 1e-15, and Z_q > 0
  tau/D_q/R^2 start     0.0 (a q with < 3 sizes -> NaN)  NaN
  < 4 valid tau(q)      D_q left at 0.0, no f(alpha)    D_q = tau/(q-1), no f(alpha)
  degree of MF          D(-5) - D(5)                    max D_q - min D_q
  D0, D1, D2 lookup     searchsorted (exact q)          nearest q within 0.01

CAVEAT (documented in the legacy code): the projection pins tau(0) and
tau(1), so a raw tau(q) that is concave over q >= 0 is replaced by the
straight line through the pins and D_q = D_0 for every q >= 0
('convexity_collapsed').  Compare Dq with Dq_raw before calling an
interface monofractal.
"""
import numpy as np
from scipy import stats

Q_DEFAULT = np.arange(-5, 5.1, 0.5)


def _convex_projection(q_values, taus, valid_tau, tol):
    """Project tau(q >= 0) onto the nearest convex function, tau(0) and tau(1)
    pinned (legacy, both dimensions).  Modifies taus in place."""
    from scipy.optimize import minimize as sp_minimize

    mask_pos = (q_values >= -1e-10) & valid_tau
    q_pos = q_values[mask_pos]
    tau_pos_raw = taus[mask_pos].copy()
    n_pos = len(q_pos)
    if n_pos < 3:
        return
    q0_pos_idx = np.argmin(np.abs(q_pos - 0.0))
    has_q0 = abs(q_pos[q0_pos_idx]) < tol
    q1_pos_idx = np.argmin(np.abs(q_pos - 1.0))
    has_q1 = abs(q_pos[q1_pos_idx] - 1.0) < tol

    def _objective(tau):
        return np.sum((tau - tau_pos_raw) ** 2)

    constraints = []
    for k in range(n_pos - 2):
        dq0 = q_pos[k + 1] - q_pos[k]
        dq1 = q_pos[k + 2] - q_pos[k + 1]

        def _conv(tau, k=k, dq0=dq0, dq1=dq1):
            return (tau[k + 1] - tau[k]) / dq0 - (tau[k + 2] - tau[k + 1]) / dq1
        constraints.append({'type': 'ineq', 'fun': _conv})
    if has_q0:
        raw_tau0 = float(tau_pos_raw[q0_pos_idx])
        constraints.append({'type': 'eq', 'fun': lambda tau, idx=q0_pos_idx, val=raw_tau0: tau[idx] - val})
    if has_q1:
        constraints.append({'type': 'eq', 'fun': lambda tau, idx=q1_pos_idx: tau[idx]})
    try:
        result = sp_minimize(_objective, tau_pos_raw, constraints=constraints,
                             method='SLSQP', options={'maxiter': 500, 'ftol': 1e-14})
        if result.success:
            taus[mask_pos] = result.x
    except Exception:
        pass


def _dq_from_tau(q_values, taus, Dqs, tol):
    """D_q = tau/(q-1); D_1 by central difference (legacy).  NaN tau -> NaN D_q."""
    for i, q_val in enumerate(q_values):
        if np.isnan(taus[i]):
            Dqs[i] = np.nan
            continue
        if abs(q_val - 1.0) < tol:
            if 0 < i < len(q_values) - 1 and not np.isnan(taus[i - 1]) and not np.isnan(taus[i + 1]):
                Dqs[i] = (taus[i + 1] - taus[i - 1]) / (q_values[i + 1] - q_values[i - 1])
        else:
            Dqs[i] = taus[i] / (q_val - 1)


def _legendre(q_values, taus, valid_tau, alpha, f_alpha):
    """alpha = d tau/dq from an interpolating cubic spline (finite differences
    if the spline fails), f = q alpha - tau (legacy)."""
    from scipy.interpolate import UnivariateSpline
    q_valid = q_values[valid_tau]
    tau_valid = taus[valid_tau]
    try:
        spline = UnivariateSpline(q_valid, tau_valid, k=3, s=0)
        dtau_dq = spline.derivative()(q_valid)
        j = 0
        for i in range(len(q_values)):
            if valid_tau[i]:
                alpha[i] = dtau_dq[j]
                f_alpha[i] = q_values[i] * alpha[i] - taus[i]
                j += 1
    except Exception:
        for i in range(len(q_values)):
            if not valid_tau[i]:
                continue
            if 0 < i < len(q_values) - 1 and valid_tau[i - 1] and valid_tau[i + 1]:
                alpha[i] = (taus[i + 1] - taus[i - 1]) / (q_values[i + 1] - q_values[i - 1])
            elif i == 0 and valid_tau[i + 1]:
                alpha[i] = (taus[i + 1] - taus[i]) / (q_values[i + 1] - q_values[i])
            elif i == len(q_values) - 1 and valid_tau[i - 1]:
                alpha[i] = (taus[i] - taus[i - 1]) / (q_values[i] - q_values[i - 1])
            if not np.isnan(alpha[i]):
                f_alpha[i] = q_values[i] * alpha[i] - taus[i]


def probabilities(measures):
    """Normalise one scale's measures (empty -> empty)."""
    measures = np.asarray(measures)
    total = measures.sum() if len(measures) > 0 else 0
    return measures / total if total > 0 else np.array([])


def spectrum_2d(box_sizes, probs, q_values=Q_DEFAULT):
    """Legacy 2D spectrum from per-scale probabilities (box_sizes large -> small).
    -> dict(q, tau, Dq, alpha, f_alpha, r_squared, tau_raw, Dq_raw, D0, D1, D2,
            alpha_width, degree_multifractality, D0_raw, D2_raw,
            convexity_adjustment, convexity_collapsed)"""
    q_values = np.asarray(q_values)
    box_sizes = np.asarray(box_sizes, dtype=np.float64)
    nq = len(q_values)
    taus, Dqs, r_squared = np.zeros(nq), np.zeros(nq), np.zeros(nq)
    for q_idx, q in enumerate(q_values):
        if abs(q - 1.0) < 1e-6:
            taus[q_idx], r_squared[q_idx] = 0.0, 1.0
            continue
        Z_q = np.zeros(len(box_sizes))
        for i, p in enumerate(probs):
            Z_q[i] = np.sum(p ** q) if len(p) > 0 else np.nan
        valid = ~np.isnan(Z_q)
        if np.sum(valid) < 3:
            taus[q_idx] = Dqs[q_idx] = r_squared[q_idx] = np.nan
            continue
        slope, _, r_value, _, _ = stats.linregress(np.log(box_sizes[valid]), np.log(Z_q[valid]))
        taus[q_idx], r_squared[q_idx] = slope, r_value ** 2

    taus_raw = taus.copy()
    Dqs_raw = np.full(nq, np.nan)
    convexity_adjustment, convexity_collapsed, D0_raw, D2_raw = 0.0, False, np.nan, np.nan
    valid_tau = ~np.isnan(taus)
    alpha, f_alpha = np.full(nq, np.nan), np.full(nq, np.nan)
    if np.sum(valid_tau) >= 4:
        _convex_projection(q_values, taus, valid_tau, 1e-6)
        _dq_from_tau(q_values, taus, Dqs, 1e-6)
        Dqs_raw = np.full(nq, np.nan)
        for i, q_val in enumerate(q_values):
            if np.isnan(taus_raw[i]):
                continue
            if abs(q_val - 1.0) < 1e-6:
                if 0 < i < nq - 1 and not np.isnan(taus_raw[i - 1]) and not np.isnan(taus_raw[i + 1]):
                    Dqs_raw[i] = (taus_raw[i + 1] - taus_raw[i - 1]) / (q_values[i + 1] - q_values[i - 1])
            else:
                Dqs_raw[i] = taus_raw[i] / (q_val - 1)
        _pos = q_values >= -1e-10
        with np.errstate(invalid='ignore'):
            _d = np.abs(taus[_pos] - taus_raw[_pos])
        convexity_adjustment = float(np.nanmax(_d)) if _d.size else 0.0
        _hi = Dqs[q_values >= 0.5]
        _hi = _hi[~np.isnan(_hi)]
        convexity_collapsed = bool(_hi.size >= 3 and (_hi.max() - _hi.min()) < 1e-6)
        D0_raw = float(Dqs_raw[int(np.argmin(np.abs(q_values)))])
        D2_raw = float(Dqs_raw[int(np.argmin(np.abs(q_values - 2.0)))])
        _legendre(q_values, taus, valid_tau, alpha, f_alpha)

    if np.sum(~np.isnan(Dqs)) >= 3:
        D0 = Dqs[np.searchsorted(q_values, 0)] if 0 in q_values else np.nan
        D1 = Dqs[np.searchsorted(q_values, 1)] if 1 in q_values else np.nan
        D2 = Dqs[np.searchsorted(q_values, 2)] if 2 in q_values else np.nan
        va = ~np.isnan(alpha)
        alpha_width = np.max(alpha[va]) - np.min(alpha[va]) if np.sum(va) >= 2 else np.nan
        if -5 in q_values and 5 in q_values:
            degree = Dqs[np.searchsorted(q_values, -5)] - Dqs[np.searchsorted(q_values, 5)]
        else:
            degree = np.nan
    else:
        D0 = D1 = D2 = alpha_width = degree = np.nan
    return dict(q=q_values, tau=taus, Dq=Dqs, alpha=alpha, f_alpha=f_alpha, r_squared=r_squared,
                tau_raw=taus_raw, Dq_raw=Dqs_raw, D0=D0, D1=D1, D2=D2, alpha_width=alpha_width,
                degree_multifractality=degree, D0_raw=D0_raw, D2_raw=D2_raw,
                convexity_adjustment=convexity_adjustment, convexity_collapsed=convexity_collapsed)


def spectrum_3d(cube_sizes, probs, q_values=Q_DEFAULT):
    """Legacy 3D spectrum from per-scale probabilities (cube_sizes large -> small).
    Same return keys as spectrum_2d."""
    q_values = np.asarray(q_values, dtype=np.float64)
    cube_sizes = np.asarray(cube_sizes, dtype=np.float64)
    nq, n_scales = len(q_values), len(cube_sizes)
    taus, r_squared = np.full(nq, np.nan), np.full(nq, np.nan)
    for q_idx, q in enumerate(q_values):
        if abs(q - 1.0) < 1e-8:
            taus[q_idx], r_squared[q_idx] = 0.0, 1.0
            continue
        Z_q = np.full(n_scales, np.nan)
        for s, p in enumerate(probs):
            if p.size > 0:
                mask = p > 1e-15
                if np.any(mask):
                    Z_q[s] = np.sum(p[mask] ** q)
        valid = ~np.isnan(Z_q) & (Z_q > 0)
        if np.sum(valid) >= 3:
            slope, _, r_val, _, _ = stats.linregress(np.log(cube_sizes[valid]), np.log(Z_q[valid]))
            taus[q_idx], r_squared[q_idx] = slope, r_val ** 2

    taus_raw = taus.copy()
    Dq, alpha, f_alpha = np.full(nq, np.nan), np.full(nq, np.nan), np.full(nq, np.nan)
    valid_tau = ~np.isnan(taus)
    if np.sum(valid_tau) >= 4:
        _convex_projection(q_values, taus, valid_tau, 1e-8)
        _dq_from_tau(q_values, taus, Dq, 1e-8)
        _legendre(q_values, taus, valid_tau, alpha, f_alpha)
    else:
        for i, q in enumerate(q_values):
            if np.isnan(taus[i]) or abs(q - 1.0) < 1e-8:
                continue
            Dq[i] = taus[i] / (q - 1)

    def _get(q_target):
        idx = np.argmin(np.abs(q_values - q_target))
        return Dq[idx] if abs(q_values[idx] - q_target) < 0.01 else np.nan

    va = alpha[~np.isnan(alpha)]
    alpha_width = (va.max() - va.min()) if len(va) >= 2 else np.nan
    vd = Dq[~np.isnan(Dq)]
    degree = (vd.max() - vd.min()) if len(vd) >= 3 else np.nan

    Dq_raw = np.full(nq, np.nan)
    for i, q in enumerate(q_values):
        if np.isnan(taus_raw[i]):
            continue
        if abs(q - 1.0) < 1e-8:
            if 0 < i < nq - 1 and not np.isnan(taus_raw[i - 1]) and not np.isnan(taus_raw[i + 1]):
                Dq_raw[i] = (taus_raw[i + 1] - taus_raw[i - 1]) / (q_values[i + 1] - q_values[i - 1])
        else:
            Dq_raw[i] = taus_raw[i] / (q - 1.0)
    pos = q_values >= -1e-10
    with np.errstate(invalid='ignore'):
        diff = np.abs(taus[pos] - taus_raw[pos])
    cvx_adjust = float(np.nanmax(diff)) if diff.size else 0.0
    hi = Dq[q_values >= 0.5]
    hi = hi[~np.isnan(hi)]
    return dict(q=q_values, tau=taus, Dq=Dq, alpha=alpha, f_alpha=f_alpha, r_squared=r_squared,
                tau_raw=taus_raw, Dq_raw=Dq_raw, D0=_get(0.0), D1=_get(1.0), D2=_get(2.0),
                alpha_width=alpha_width, degree_multifractality=degree,
                D0_raw=Dq_raw[int(np.argmin(np.abs(q_values)))], D2_raw=Dq_raw[int(np.argmin(np.abs(q_values - 2.0)))],
                convexity_adjustment=cvx_adjust, convexity_collapsed=bool(hi.size >= 3 and (hi.max() - hi.min()) < 1e-6))


# ------------------------------------------------------------------ drivers
def segments_bbox(segs, margin_factor=0.01):
    """Legacy 2D covering box: segment extent plus a 1 % margin.
    -> (extent, (min_x, min_y, max_x, max_y))."""
    xs = np.concatenate([segs[:, 0], segs[:, 2]])
    ys = np.concatenate([segs[:, 1], segs[:, 3]])
    min_x, max_x, min_y, max_y = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    extent = max(max_x - min_x, max_y - min_y)
    m = extent * margin_factor
    return extent, (min_x - m, min_y - m, max_x + m, max_y + m)


def multifractal_2d(segs, box_sizes, q_values=Q_DEFAULT, backend=None, offsets='covering', log=None):
    """Spectrum of a 2D contour: segment count per box, grid-optimised.

    box_sizes are sorted large -> small first (legacy).  The returned dict
    also holds box_sizes and n_occupied (per size)."""
    from .boxcount.measures2d import SegmentMeasure, grid_optimized_measure
    box_sizes = np.sort(np.asarray(box_sizes, dtype=np.float64))[::-1]
    _, bbox = segments_bbox(segs)
    counter = SegmentMeasure(segs, backend)
    probs, n_occ = [], []
    for box in box_sizes:
        occ, n, tests = grid_optimized_measure(counter, box, bbox, offsets)
        probs.append(probabilities(occ))
        n_occ.append(len(occ))
        if log:
            log(f"    box {box:.6f}: {len(occ)} occupied [{tests} offsets]")
    res = spectrum_2d(box_sizes, probs, q_values)
    res.update(box_sizes=box_sizes, n_occupied=np.array(n_occ))
    return res


def multifractal_3d(mesh, cube_sizes, q_values=Q_DEFAULT, measure='area', backend=None, offsets='covering', log=None):
    """Spectrum of a 3D surface: triangle area (or count) per cube, grid-optimised.

    cube_sizes are sorted large -> small first (legacy); at least 3 needed."""
    from .boxcount.measures3d import TriangleMeasure, grid_optimized_measure
    cube_sizes = np.sort(np.asarray(cube_sizes, dtype=np.float64))[::-1]
    if len(cube_sizes) < 3:
        raise ValueError(f'insufficient scale range: {len(cube_sizes)} scales (need >= 3)')
    counter = TriangleMeasure(mesh, measure, backend)
    probs, n_occ = [], []
    for d in cube_sizes:
        m, n, tests = grid_optimized_measure(counter, d, mesh.bbox, offsets)
        probs.append(m / m.sum() if m.size > 0 else np.array([]))
        n_occ.append(n)
        if log:
            log(f"    delta {d:.6f}: {n} occupied [{tests} offsets]")
    res = spectrum_3d(cube_sizes, probs, q_values)
    res.update(box_sizes=cube_sizes, n_occupied=np.array(n_occ))
    return res
