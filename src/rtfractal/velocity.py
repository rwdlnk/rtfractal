"""Velocity statistics (phase 3).

Ported from FractalParameterAI: ``integration/rt_velocity_statistics.py``
(VelocityAnalyzer.analyze_velocity_field, 2D) and
``rt_analyzer/analyze_rt_3d.py`` (compute_velocity_statistics_3d).  The
arithmetic is kept operation for operation (bit-identical results).
"""
import numpy as np


def velocity_stats_2d(u, v):
    """Whole-field statistics of a 2D velocity field (u, v same shape).

    Fluctuations are taken about the spatial mean of the whole domain (not a
    horizontal mean per height).  Returns u_mean, v_mean, u_rms, v_rms,
    velocity_mag_mean/_rms/_max, TKE = (u_rms^2 + v_rms^2)/2,
    reynolds_stress = <u'v'>, turbulence_intensity = sqrt(TKE)/|<V>| or, when
    |<V>| <= 1e-12, sqrt(TKE)/|V|_rms, plus n_points and grid_shape.
    """
    if u.shape != v.shape:
        raise ValueError(f"u and v shapes must match: {u.shape} vs {v.shape}")
    u_mean, v_mean = np.mean(u), np.mean(v)
    u_prime, v_prime = u - u_mean, v - v_mean
    u_rms = np.sqrt(np.mean(u_prime**2))
    v_rms = np.sqrt(np.mean(v_prime**2))
    vmag = np.sqrt(u**2 + v**2)
    vmag_mean = np.mean(vmag)
    vmag_rms = np.sqrt(np.mean(vmag**2))
    vmag_max = np.max(vmag)
    tke = 0.5 * (u_rms**2 + v_rms**2)
    reynolds_stress = np.mean(u_prime * v_prime)
    mean_mag = np.sqrt(u_mean**2 + v_mean**2)
    if mean_mag > 1e-12:
        ti = np.sqrt(tke) / mean_mag
    elif vmag_rms > 1e-12:
        ti = np.sqrt(tke) / vmag_rms
    else:
        ti = 0.0
    return {'u_mean': u_mean, 'v_mean': v_mean, 'u_rms': u_rms, 'v_rms': v_rms,
            'velocity_mag_mean': vmag_mean, 'velocity_mag_rms': vmag_rms, 'velocity_mag_max': vmag_max,
            'TKE': tke, 'reynolds_stress': reynolds_stress, 'turbulence_intensity': ti,
            'n_points': u.size, 'grid_shape': u.shape}


def velocity_stats_3d(U, y_centers, H0):
    """Horizontal-plane statistics of U (nz, ny, nx, 3) at every height.

    Fluctuations are about the (x, z) mean at each y.  Profiles: u_rms,
    v_rms, w_rms and the raw normalised moments S = <u'^3>/<u'^2>^1.5,
    K = <u'^4>/<u'^2>^2 (Gaussian: S = 0, K = 3; NaN where <u'^2> <= 1e-24).
    '*_centreline' scalars are taken at the row nearest the FIXED initial
    interface H0 (not the mixing-layer centre), and anisotropy_v_over_u is
    v_rms/u_rms there (NaN if u_rms <= 1e-12).
    """
    u_mean = np.mean(U, axis=(0, 2))                               # (ny, 3)
    u_prime = U - u_mean[np.newaxis, :, np.newaxis, :]
    rms = {c: np.sqrt(np.mean(u_prime[:, :, :, i]**2, axis=(0, 2))) for i, c in enumerate('uvw')}
    j0 = np.argmin(np.abs(y_centers - H0))
    u_c, v_c, w_c = float(rms['u'][j0]), float(rms['v'][j0]), float(rms['w'][j0])
    out = {'y_centers': y_centers, 'u_rms': rms['u'], 'v_rms': rms['v'], 'w_rms': rms['w'],
           'u_rms_centreline': u_c, 'v_rms_centreline': v_c, 'w_rms_centreline': w_c,
           'anisotropy_v_over_u': v_c / u_c if u_c > 1e-12 else np.nan}
    skew, kurt = {}, {}
    for i, c in enumerate('uvw'):
        f = u_prime[:, :, :, i]
        m2 = np.mean(f**2, axis=(0, 2))
        m3 = np.mean(f**3, axis=(0, 2))
        m4 = np.mean(f**4, axis=(0, 2))
        with np.errstate(invalid='ignore', divide='ignore'):
            skew[c] = np.where(m2 > 1e-24, m3 / m2**1.5, np.nan)
            kurt[c] = np.where(m2 > 1e-24, m4 / m2**2, np.nan)
    for c in 'uvw':
        out[f'{c}_skew'] = skew[c]
    for c in 'uvw':
        out[f'{c}_kurt'] = kurt[c]
    for c in 'uvw':
        out[f'{c}_skew_centreline'] = float(skew[c][j0])
    for c in 'uvw':
        out[f'{c}_kurt_centreline'] = float(kurt[c][j0])
    return out
