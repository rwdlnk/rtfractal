"""Mixing-layer measures.

2D (fields (nx, ny)) and 3D (alpha (nz, ny, nx)) share the horizontally
averaged profile <F>(y) and its statistical (1 % threshold) and integral
(Dalziel et al. 1999 Eq. 7) widths.  F = 1 is the heavy fluid; orientation is
detected from the profile.
"""
import numpy as np

_trapezoid = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
EPS = 0.01


def statistical_width(f_avg, y, h0):
    if f_avg[-1] > f_avg[0]:          # heavy on top
        up = np.where(f_avg < 1 - EPS)[0]
        lo = np.where(f_avg > EPS)[0]
    else:
        up = np.where(f_avg > EPS)[0]
        lo = np.where(f_avg < 1 - EPS)[0]
    y_up = y[up[-1]] if len(up) else y[-1]
    y_lo = y[lo[0]] if len(lo) else y[0]
    ht, hb = max(0, y_up - h0), max(0, h0 - y_lo)
    return {'ht': ht, 'hb': hb, 'h_total': ht + hb}


def integral_widths(f_avg, y, h0):
    """h_{m,n}: fluid m (1 heavy, 0 light) found in region n (0 lower, 1 upper);
    no conservation imposed.  ht = h_01, hb = h_10."""
    lo, up = y <= h0, y >= h0
    f_lo, f_up, y_lo, y_up = f_avg[lo], f_avg[up], y[lo], y[up]
    if f_avg[-1] > f_avg[0]:
        h10, h11 = float(_trapezoid(f_lo, y_lo)), float(_trapezoid(f_up, y_up))
        h00, h01 = float(_trapezoid(1.0 - f_lo, y_lo)), float(_trapezoid(1.0 - f_up, y_up))
    else:
        h10, h11 = float(_trapezoid(1.0 - f_lo, y_lo)), float(_trapezoid(1.0 - f_up, y_up))
        h00, h01 = float(_trapezoid(f_lo, y_lo)), float(_trapezoid(f_up, y_up))
    return {'ht': h01, 'hb': h10, 'h_total': h01 + h10, 'h_10': h10, 'h_11': h11, 'h_00': h00, 'h_01': h01}


def geometric_width(cs, h0):
    """Largest excursion of the interface contours above/below h0."""
    ht = hb = 0.0
    for c in cs:
        ht = max(ht, np.max(c[:, 1] - h0))
        hb = max(hb, np.max(h0 - c[:, 1]))
    return {'ht': ht, 'hb': hb, 'h_total': ht + hb}


# --------------------------------------------------------------------------- 3D
def youngs(f_avg, y):
    w = float(_trapezoid(f_avg * (1.0 - f_avg), y))
    m = (f_avg > 0.05) & (f_avg < 0.95)
    if np.any(m):
        idx = np.where(m)[0]
        width = y[idx[-1]] - y[idx[0]]
        perfect = 0.25 * width if width > 0 else 0
        eff = min(1.0, w / perfect) if perfect > 0 else 0
    else:
        eff = 0.0
    return {'youngs_W': w, 'mixing_efficiency': eff}


def molecular_mixing_3d(alpha, y):
    """Dalziel, Linden & Youngs (1999) Sect. 7.2, Eqs. (10)-(13); alpha (nz, ny, nx)."""
    C = alpha
    P = C * (1.0 - C)
    C_pl, P_pl = C.mean(axis=(0, 2)), P.mean(axis=(0, 2))
    den_pl = C_pl * (1.0 - C_pl)
    theta = np.divide(P_pl, den_pl, out=np.zeros_like(P_pl), where=den_pl > 0)
    C_x, P_x = C.mean(axis=2), P.mean(axis=2)
    den_x = C_x * (1.0 - C_x)
    theta_hat = np.divide(P_x, den_x, out=np.zeros_like(P_x), where=den_x > 0).mean(axis=0)
    num, den = float(_trapezoid(P_pl, y)), float(_trapezoid(den_pl, y))
    Theta = num / den if den > 0 else 0.0
    num_s = np.array([float(_trapezoid(P_x[k], y)) for k in range(P_x.shape[0])])
    den_s = np.array([float(_trapezoid(den_x[k], y)) for k in range(den_x.shape[0])])
    Theta_hat = float(np.divide(num_s, den_s, out=np.zeros_like(num_s), where=den_s > 0).mean())
    return {'theta': theta, 'theta_hat': theta_hat, 'Theta': Theta, 'Theta_hat': Theta_hat}


def energy_budget_3d(alpha, U, y, grid, rho_h, rho_l, g):
    """PE, background PE (adiabatic re-sorting), APE = PE - BPE, KE."""
    dV = grid.dx * grid.dy * grid.dz
    rho = rho_l + (rho_h - rho_l) * alpha
    PE = float(np.sum(rho * y[None, :, None]) * g * dV)
    BPE = float(np.sum(np.sort(rho, axis=None)[::-1] * np.repeat(y, grid.nx * grid.nz)) * g * dV)
    KE = float(0.5 * np.sum(rho * np.sum(U * U, axis=-1)) * dV) if U is not None else np.nan
    return {'PE': PE, 'BPE': BPE, 'APE': PE - BPE, 'KE': KE}


def reference_energy_3d(y, grid, H0, rho_h, rho_l, g):
    """Energy of the unperturbed initial state (heavy above H0, at rest)."""
    dV = grid.dx * grid.dy * grid.dz
    ncol = grid.nx * grid.nz
    rho_col = np.where(y >= H0, rho_h, rho_l)
    PE0 = float(np.sum(rho_col * y) * g * dV * ncol)
    BPE0 = float(np.sum(np.sort(rho_col)[::-1] * y) * g * dV * ncol)
    return {'PE': PE0, 'BPE': BPE0, 'APE': PE0 - BPE0, 'KE': 0.0}
