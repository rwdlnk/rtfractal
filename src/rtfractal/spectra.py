"""Power spectra of RT fields (phase 3).

Ported from FractalParameterAI: the 2D functions from
``integration/rt_power_spectrum.py`` (PowerSpectrumAnalyzer), the 3D ones from
``rt_analyzer/analyze_rt_3d.py``.  The arithmetic is kept operation for
operation, so results are bit-identical to the legacy code.

2D (fields indexed [i_x, j_y], legacy layout):
    dalziel_spectrum_2d   Dalziel et al. (1999) §6.1 horizontal-slab spectrum:
                          1-D FFT along x for each row in y0 - 0.1H <= y <= y0,
                          end-ramp padding to a power of two, mean power, and
                          weighted power-law fits over fixed k/k0 bands.
    field_spectrum_2d     full-domain 2-D FFT, log-binned radial average,
                          power-law fit over the middle half of the bins.
    tke_spectrum_2d       legacy "TKE" spectrum (see its docstring: it is the
                          u spectrum with the total energy averaged with v).

3D (fields indexed [k_z, j_y, i_x]):
    horizontal_spectrum_3d  slab-averaged F(x, z), end-ramp padded 2-D FFT,
                            linear radial bins, unweighted band fits.
    tke_spectrum_3d         the same for u, v, w and E_tke = (E_u+E_v+E_w)/2.

Wavenumbers: 2D Dalziel and 3D spectra return k/k0 with k0 = 2 pi / L
(equivalently 1/L in cycles), L the unpadded domain length, so k/k0 is the
mode number.  The 2D radial spectra return k in rad/m.
"""
import numpy as np

_trapezoid = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

DALZIEL_BANDS = ((10.0, 25.0), (10.0, 50.0))
BANDS_3D = (('10_25', 10, 25), ('10_50', 10, 50))
BANDS_3D_TKE = (('10_25', 10, 25), ('10_50', 10, 50), ('5_50', 5, 50))


# ─── shared helpers ──────────────────────────────────────────────────────────

def next_pow2(n):
    """Smallest power of two >= n."""
    p = 1 << (int(n - 1).bit_length())
    return p if p >= n else p << 1


def endramp_pad_1d(line, n_pad):
    """Extend `line` to length n_pad by a linear ramp from its last value back
    to its first (Dalziel's non-periodic continuation)."""
    m = n_pad - len(line)
    if m <= 0:
        return line
    ramp = np.linspace(line[-1], line[0], m + 1)[1:]
    return np.concatenate([line, ramp])


def pad_endramp_2d(f):
    """Extend a (nz, nx) field to the next power of two in both directions by
    end ramps, x first, then z (legacy ``_pad_endramp_2d``)."""
    nz, nx = f.shape
    nx_pad = next_pow2(nx)
    if nx_pad > nx:
        m = nx_pad - nx
        ramp = np.linspace(f[:, -1], f[:, 0], m + 1, axis=-1)[:, 1:]
        f = np.concatenate([f, ramp], axis=1)
    nz_pad = next_pow2(nz)
    if nz_pad > nz:
        m = nz_pad - nz
        ramp = np.linspace(f[-1, :], f[0, :], m + 1, axis=0)[1:, :]
        f = np.concatenate([f, ramp], axis=0)
    return f


# ─── 2D: Dalziel horizontal-slab spectrum ────────────────────────────────────

def fit_powerlaw_weighted(k_over_k0, power, k_lo, k_hi):
    """Least-squares P ~ (k/k0)^beta over k_lo <= k/k0 <= k_hi, weighted
    uniformly per logarithmic interval (residual weight 1/sqrt(k/k0))."""
    band = (k_over_k0 >= k_lo) & (k_over_k0 <= k_hi) & (power > 0)
    if np.sum(band) < 3:
        return {'power_law_slope': np.nan, 'beta_abs': np.nan, 'power_law_r_squared': np.nan,
                'fit_k_k0_min': k_lo, 'fit_k_k0_max': k_hi, 'n_fit_points': int(np.sum(band))}
    kf, pf = k_over_k0[band], power[band]
    log_k, log_p = np.log10(kf), np.log10(pf)
    w = 1.0 / np.sqrt(kf)
    slope, intercept = np.polyfit(log_k, log_p, 1, w=w)
    predicted = slope * log_k + intercept
    ss_res = np.sum(w**2 * (log_p - predicted) ** 2)
    ss_tot = np.sum(w**2 * (log_p - np.average(log_p, weights=w**2)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {'power_law_slope': float(slope), 'beta_abs': float(abs(slope)),
            'power_law_intercept': float(intercept), 'power_law_r_squared': float(r_squared),
            'fit_k_k0_min': float(k_lo), 'fit_k_k0_max': float(k_hi), 'n_fit_points': int(np.sum(band))}


def _empty_dalziel():
    return {'power_law_slope': np.nan, 'beta_abs': np.nan, 'power_law_r_squared': np.nan,
            'k_over_k0': np.array([]), 'power': np.array([]), 'k0': np.nan, 'L': np.nan,
            'n_slab_rows': 0, 'slab_y_range': (np.nan, np.nan), 'n_pad': 0, 'n_orig': 0,
            'k_6dx': np.nan, 'band_keys': [], 'field_name': 'unknown'}


def dalziel_spectrum_2d(field, x, y, y0, H, slab_frac=0.1, bands=DALZIEL_BANDS, field_name='F'):
    """Dalziel, Linden & Youngs (1999) horizontal concentration spectrum.

    field, x, y: (nx, ny) arrays, axis 0 along the tank.  Rows with
    y0 - slab_frac*H <= y <= y0 (or the single row nearest y0 if none) are
    end-ramp padded to the next power of two, rFFT'd, and |F|^2/N_pad is
    averaged over rows.  Returns k_over_k0 (DC dropped), power, k0, L,
    n_slab_rows, n_pad, n_orig, k_6dx = N/6, band_keys and one fit dict per
    band ('fit_10_25', ...; see fit_powerlaw_weighted, plus fully_resolved =
    hi <= k_6dx).  The first band is mirrored at top level.
    """
    n_orig, ny = field.shape
    if n_orig < 8 or ny < 2:
        return _empty_dalziel()
    dx = np.abs(x[1, 0] - x[0, 0])
    L = n_orig * dx
    k0 = 2.0 * np.pi / L
    k_6dx = n_orig / 6.0

    yv = y[0, :]
    slab_idx = np.where((yv >= y0 - slab_frac * H) & (yv <= y0))[0]
    if len(slab_idx) == 0:
        slab_idx = np.array([int(np.argmin(np.abs(yv - y0)))])

    n_pad = next_pow2(n_orig)
    n_modes = n_pad // 2 + 1
    power_sum = np.zeros(n_modes)
    for j in slab_idx:
        padded = endramp_pad_1d(field[:, j].astype(np.float64), n_pad)
        power_sum += (np.abs(np.fft.rfft(padded)) ** 2) / n_pad
    power = power_sum / len(slab_idx)

    m = np.arange(n_modes)
    k_over_k0 = m * (n_orig / n_pad)
    nz = m > 0
    k_over_k0, power = k_over_k0[nz], power[nz]

    out = {'field_name': field_name, 'k_over_k0': k_over_k0, 'power': power, 'k0': k0, 'L': L,
           'n_slab_rows': int(len(slab_idx)), 'slab_y_range': (float(y0 - slab_frac * H), float(y0)),
           'n_pad': int(n_pad), 'n_orig': int(n_orig), 'k_6dx': float(k_6dx), 'band_keys': []}
    for lo, hi in bands:
        fit = fit_powerlaw_weighted(k_over_k0, power, lo, hi)
        fit['fully_resolved'] = bool(hi <= k_6dx)
        key = f'fit_{int(round(lo))}_{int(round(hi))}'
        out[key] = fit
        out['band_keys'].append(key)
    if out['band_keys']:
        first = out[out['band_keys'][0]]
        out['power_law_slope'] = first['power_law_slope']
        out['beta_abs'] = first['beta_abs']
        out['power_law_r_squared'] = first['power_law_r_squared']
    return out


# ─── 2D: full-domain radial spectra ──────────────────────────────────────────

def detrend_plane(field):
    """Subtract the least-squares plane a*i + b*j + c (index coordinates)."""
    nx, ny = field.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    A = np.column_stack([X.ravel(), Y.ravel(), np.ones(field.size)])
    coeffs = np.linalg.lstsq(A, field.ravel(), rcond=None)[0]
    return field - (coeffs[0] * X + coeffs[1] * Y + coeffs[2])


def radial_average_log(K, power_2d, n_bins=50):
    """Mean power and mean |k| in n_bins-1 log-spaced bins between the
    smallest and largest non-zero |k|; empty bins are dropped.  Bins are
    half-open [lo, hi), so the single largest |k| is never counted (legacy)."""
    k_flat, p_flat = K.ravel(), power_2d.ravel()
    nonzero = k_flat > 0
    k_flat, p_flat = k_flat[nonzero], p_flat[nonzero]
    bins = np.logspace(np.log10(k_flat.min()), np.log10(k_flat.max()), n_bins)
    k_binned, p_binned = [], []
    for i in range(len(bins) - 1):
        mask = (k_flat >= bins[i]) & (k_flat < bins[i + 1])
        if np.sum(mask) > 0:
            k_binned.append(np.mean(k_flat[mask]))
            p_binned.append(np.mean(p_flat[mask]))
    return np.array(k_binned), np.array(p_binned)


def radial_spectrum_2d(field, dx, dy, detrend=True, n_bins=50):
    """(k [rad/m], E(k)) of a (nx, ny) field: mean removed, optional plane
    detrend, |FFT|^2 / N, log-binned radial average."""
    fc = field - np.mean(field)
    if detrend:
        fc = detrend_plane(fc)
    power_2d = np.abs(np.fft.fft2(fc)) ** 2 / field.size
    nx, ny = field.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    return radial_average_log(np.sqrt(KX**2 + KY**2), power_2d, n_bins)


def _empty_features():
    return {'power_law_slope': np.nan, 'power_law_r_squared': np.nan, 'dominant_wavelength': np.nan,
            'cutoff_wavelength': np.nan, 'total_energy': np.nan, 'peak_power': np.nan, 'n_modes': 0,
            'domain_length': 0.0, 'wavenumber_range': (np.nan, np.nan), 'field_name': 'unknown'}


def spectrum_features(k, power, x_domain):
    """Legacy ``_analyze_1d_spectrum``: trapezoid 'total energy' over the
    binned spectrum, dominant wavelength 2 pi / k_peak, unweighted log-log
    slope over bins [n/4, 3n/4), and the 10%-of-peak cutoff wavelength.
    Slope and R^2 are 0.0 (not NaN) when too few bins are available."""
    if len(k) < 3:
        return _empty_features()
    total_energy = _trapezoid(power, k)
    peak = np.argmax(power)
    dominant_k = k[peak]
    dominant_wavelength = 2 * np.pi / dominant_k if dominant_k > 0 else np.inf
    n_k = len(k)
    fit_start, fit_end = max(1, n_k // 4), min(n_k - 1, 3 * n_k // 4)
    slope, r_squared = 0.0, 0.0
    if fit_end > fit_start + 3:
        k_fit, p_fit = k[fit_start:fit_end], power[fit_start:fit_end]
        valid = p_fit > 0
        if np.sum(valid) > 2:
            log_k, log_p = np.log10(k_fit[valid]), np.log10(p_fit[valid])
            coeffs = np.polyfit(log_k, log_p, 1)
            slope = coeffs[0]
            predicted = slope * log_k + coeffs[1]
            ss_res = np.sum((log_p - predicted) ** 2)
            ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    cut = np.where(power[peak:] < 0.1 * power[peak])[0]
    if len(cut) > 0:
        ck = k[peak + cut[0]]
        cutoff_wavelength = 2 * np.pi / ck if ck > 0 else np.inf
    else:
        cutoff_wavelength = np.inf
    return {'power_law_slope': slope, 'power_law_r_squared': r_squared,
            'dominant_wavelength': dominant_wavelength, 'cutoff_wavelength': cutoff_wavelength,
            'total_energy': total_energy, 'peak_power': power[peak], 'n_modes': len(k),
            'domain_length': x_domain[-1] - x_domain[0] if len(x_domain) > 1 else 0.0,
            'wavenumber_range': (k[0], k[-1])}


def field_spectrum_2d(field, x, y, field_name='field', detrend=True, return_spectrum=False):
    """Radially averaged spectrum of a 2D field plus its features (legacy
    ``analyze_2d_field_spectrum``).  With return_spectrum=True the binned
    'k' and 'power' arrays are added to the result."""
    if field.size < 100:
        return _empty_features()
    nx, ny = field.shape
    dx = np.abs(x[1, 0] - x[0, 0]) if nx > 1 else 1.0
    dy = np.abs(y[0, 1] - y[0, 0]) if ny > 1 else 1.0
    k, power = radial_spectrum_2d(field, dx, dy, detrend)
    out = spectrum_features(k, power, np.array([x.min(), x.max()]))
    out['field_name'] = field_name
    out['grid_shape'] = field.shape
    if return_spectrum:
        out['k'], out['power'] = k, power
    return out


def tke_spectrum_2d(u, v, x, y, detrend=True, u_spec=None, v_spec=None):
    """Legacy 2D 'TKE spectrum'.

    LEGACY QUIRK, reproduced: the result is the u spectrum's features (slope,
    R^2, dominant wavelength are those of u alone); only total_energy is
    (E_u + E_v)/2.  Pass precomputed u_spec/v_spec to avoid recomputing them
    (identical result: the computation is deterministic)."""
    u_spec = field_spectrum_2d(u, x, y, 'u', detrend) if u_spec is None else u_spec
    v_spec = field_spectrum_2d(v, x, y, 'v', detrend) if v_spec is None else v_spec
    out = dict(u_spec)
    out['field_name'] = 'turbulent_kinetic_energy'
    if not np.isnan(u_spec['total_energy']) and not np.isnan(v_spec['total_energy']):
        out['total_energy'] = (u_spec['total_energy'] + v_spec['total_energy']) / 2
    return out


# ─── 3D: horizontal spectra over the Dalziel slab ────────────────────────────

def radial_power_spectrum_xz(f_slab, dx, dz, n_bins=None):
    """Radially averaged horizontal spectrum of a (nz, nx) field.

    Mean removed, end-ramp padded, |FFT2|^2 / N_pad, binned in n_bins
    (default min(nx, nz)//2) linear bins of k_h on [0, 0.5/min(dx, dz)].
    Returns (k/k0, P, k0) with k0 = 1/Lx (cycles per metre, unpadded Lx).
    Empty bins give P = 0."""
    nz, nx = f_slab.shape
    k0 = 1.0 / (nx * dx)
    f = pad_endramp_2d(f_slab - np.mean(f_slab))
    nz_p, nx_p = f.shape
    power_2d = np.abs(np.fft.fft2(f)) ** 2 / (nx_p * nz_p)
    KX, KZ = np.meshgrid(np.fft.fftfreq(nx_p, d=dx), np.fft.fftfreq(nz_p, d=dz))
    K_h = np.sqrt(KX ** 2 + KZ ** 2)
    k_max = 0.5 / min(dx, dz)
    n_bins = min(nx, nz) // 2 if n_bins is None else n_bins
    k_edges = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])
    P = np.zeros(n_bins)
    for ib in range(n_bins):
        m = (K_h >= k_edges[ib]) & (K_h < k_edges[ib + 1])
        if m.any():
            P[ib] = np.mean(power_2d[m])
    return k_centers / k0, P, k0


def fit_bands(k_norm, P, bands, out=None, prefix=''):
    """Unweighted log-log slope and R^2 of P(k/k0) over each (name, lo, hi)
    band, stored as out[prefix + 'slope_' + name], out[prefix + 'r2_' + name]."""
    out = {} if out is None else out
    for name, k_lo, k_hi in bands:
        m = (k_norm >= k_lo) & (k_norm <= k_hi) & (P > 0)
        if m.sum() >= 3:
            lk, lp = np.log10(k_norm[m]), np.log10(P[m])
            c = np.polyfit(lk, lp, 1)
            res = lp - np.polyval(c, lk)
            ss_tot = np.sum((lp - np.mean(lp)) ** 2)
            out[f'{prefix}slope_{name}'] = c[0]
            out[f'{prefix}r2_{name}'] = 1 - np.sum(res ** 2) / ss_tot if ss_tot > 0 else 0.0
        else:
            out[f'{prefix}slope_{name}'] = np.nan
            out[f'{prefix}r2_{name}'] = np.nan
    return out


def _slab(y_centers, y0, H):
    return (y_centers >= y0 - 0.1 * H) & (y_centers <= y0)


def horizontal_spectrum_3d(alpha, y_centers, y0, H, dx, dz, bands=BANDS_3D):
    """Concentration spectrum of alpha (nz, ny, nx) averaged over the slab
    y0 - 0.1H <= y <= y0.  None if the slab has fewer than 2 rows.
    Returns k_over_k0, P, n_slab_rows, k0, slope_<band>, r2_<band>."""
    slab = _slab(y_centers, y0, H)
    n_slab = int(slab.sum())
    if n_slab < 2:
        return None
    k_norm, P, k0 = radial_power_spectrum_xz(np.mean(alpha[:, slab, :], axis=1), dx, dz)
    out = {'k_over_k0': k_norm, 'P': P, 'n_slab_rows': n_slab, 'k0': k0}
    return fit_bands(k_norm, P, bands, out)


def tke_spectrum_3d(U, y_centers, y0, H, dx, dz, bands=BANDS_3D_TKE):
    """Slab-averaged horizontal spectra E_u, E_v, E_w of U (nz, ny, nx, 3) and
    E_tke = (E_u + E_v + E_w)/2, each with band fits (prefixes u_, v_, w_,
    tke_).  None if U is None or the slab has fewer than 2 rows."""
    slab = _slab(y_centers, y0, H)
    if int(slab.sum()) < 2 or U is None:
        return None
    out = {'n_slab_rows': int(slab.sum())}
    E = {}
    for i, comp in enumerate(('u', 'v', 'w')):
        f = np.mean(U[..., i][:, slab, :], axis=1)        # component first: (nz, nx)
        k_norm, P, k0 = radial_power_spectrum_xz(f, dx, dz)
        E[comp] = P
        out['k_over_k0'], out['k0'], out[f'E_{comp}'] = k_norm, k0, P
        fit_bands(k_norm, P, bands, out, prefix=f'{comp}_')
    out['E_tke'] = 0.5 * (E['u'] + E['v'] + E['w'])
    fit_bands(out['k_over_k0'], out['E_tke'], bands, out, prefix='tke_')
    return out
