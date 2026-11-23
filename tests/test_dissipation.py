"""Self-contained tests for the dissipation kernel (synthetic, known answers).

The bit-identity check against the ported scripts needs the paper-1 case
tree, so it is not here; it was run by hand over OpenFOAM 2D, OpenFOAM 3D and
SOLA-VOF, and all nine reported quantities matched exactly in each.
"""
import numpy as np
import pytest

from rtfractal import dissipation as D


def uniform_alpha(shape, value=0.5):
    return np.full(shape, float(value))


# ------------------------------------------------------- band bookkeeping
def test_mixing_rows_clamps_one_cell_inside():
    abar = np.zeros(20)
    abar[5:15] = 0.5
    assert D.mixing_rows(abar, 20) == (5, 15)
    abar[:] = 0.5                       # band touches both walls
    assert D.mixing_rows(abar, 20) == (1, 19)


def test_mixing_rows_none_when_interface_is_sharp():
    abar = np.r_[np.zeros(10), np.ones(10)]     # no cell strictly between
    assert D.mixing_rows(abar, 20) is None


def test_integral_width_is_calibrated_on_a_linear_ramp():
    # a mean profile ramping 0 -> 1 over width W must return W
    ny, H, W = 2000, 1.0, 0.25
    dy = H / ny
    y = (np.arange(ny) + 0.5) * dy
    abar = np.clip((y - (0.5 - W / 2)) / W, 0.0, 1.0)
    assert D.integral_width(abar, dy) == pytest.approx(W, rel=2e-3)


# ------------------------------------------------------------ 2D kernel
def test_2d_pure_shear_has_known_eps_and_enstrophy():
    ny, nx = 60, 40
    dx = dy = 0.01
    G, nu = 3.0, 1.0e-6
    y = (np.arange(ny) + 0.5) * dy
    u = np.repeat(G * y[:, None], nx, axis=1)   # u = G y, v = 0
    v = np.zeros_like(u)
    r = D.dissipation_stats_2d(u, v, uniform_alpha(u.shape), dx, dy, nu, nu)
    # S_xy = G/2 is the only non-zero strain: eps = 2 nu (2 S_xy^2) = nu G^2
    assert r['eps_mix'] == pytest.approx(nu * G ** 2, rel=1e-12)
    assert r['eps_dom'] == pytest.approx(nu * G ** 2, rel=1e-12)
    assert r['ens_mix'] == pytest.approx(G ** 2, rel=1e-12)      # omega_z = -G
    assert r['nu_mix'] == pytest.approx(nu, rel=1e-12)
    assert r['eta'] == pytest.approx((nu ** 3 / (nu * G ** 2)) ** 0.25, rel=1e-12)


def test_2d_uniform_flow_has_no_dissipation():
    u = np.full((30, 20), 0.7)
    r = D.dissipation_stats_2d(u, np.full_like(u, -0.2), uniform_alpha(u.shape),
                               0.01, 0.01, 1e-6, 1e-6)
    assert r['eps_mix'] == 0.0
    assert np.isnan(r['eta'])           # eta is undefined, not infinite
    assert r['urms'] == pytest.approx(np.hypot(0.7, 0.2), rel=1e-12)


def test_2d_viscosity_follows_the_heavy_phase():
    ny, nx = 40, 20
    nu_h, nu_l = 2.0e-6, 1.0e-6
    u = np.repeat(((np.arange(ny) + 0.5) * 0.01)[:, None], nx, axis=1)
    v = np.zeros_like(u)
    heavy = D.dissipation_stats_2d(u, v, uniform_alpha(u.shape, 1.0), 0.01, 0.01, nu_h, nu_l)
    light = D.dissipation_stats_2d(u, v, uniform_alpha(u.shape, 0.0), 0.01, 0.01, nu_h, nu_l)
    # a uniform alpha of 1 or 0 leaves no mixing zone, so both must decline
    assert heavy is None and light is None
    half = D.dissipation_stats_2d(u, v, uniform_alpha(u.shape, 0.5), 0.01, 0.01, nu_h, nu_l)
    assert half['nu_mix'] == pytest.approx(0.5 * (nu_h + nu_l), rel=1e-12)


# ------------------------------------------------------------ 3D kernel
def test_3d_pure_shear_has_known_eps_and_enstrophy():
    nz, ny, nx = 12, 60, 40
    dx = dy = dz = 0.01
    G, nu = 3.0, 1.0e-6
    y = (np.arange(ny) + 0.5) * dy
    u = np.broadcast_to((G * y)[None, :, None], (nz, ny, nx)).copy()
    v = np.zeros_like(u)
    w = np.zeros_like(u)
    r = D.dissipation_stats_3d(u, v, w, uniform_alpha(u.shape), dx, dy, dz, nu, nu)
    assert r['eps_mix'] == pytest.approx(nu * G ** 2, rel=1e-12)
    assert r['ens_mix'] == pytest.approx(G ** 2, rel=1e-12)


def test_3d_matches_2d_for_a_z_invariant_field():
    nz, ny, nx = 8, 50, 30
    dx = dy = dz = 0.02
    nu_h, nu_l = 1.044e-6, 1.005e-6
    rng = np.random.default_rng(11)
    u2 = rng.standard_normal((ny, nx)) * 0.01
    v2 = rng.standard_normal((ny, nx)) * 0.01
    a2 = np.clip(np.linspace(0, 1, ny)[:, None] + 0 * u2, 0, 1)
    u3 = np.broadcast_to(u2, (nz, ny, nx)).copy()
    v3 = np.broadcast_to(v2, (nz, ny, nx)).copy()
    w3 = np.zeros_like(u3)
    a3 = np.broadcast_to(a2, (nz, ny, nx)).copy()
    r2 = D.dissipation_stats_2d(u2, v2, a2, dx, dy, nu_h, nu_l)
    r3 = D.dissipation_stats_3d(u3, v3, w3, a3, dx, dy, dz, nu_h, nu_l)
    assert (r3['j0'], r3['j1']) == (r2['j0'], r2['j1'])
    for k in ('h99', 'hint', 'urms', 'eps_mix', 'ens_mix', 'nu_mix', 'eta_over_dx'):
        assert r3[k] == pytest.approx(r2[k], rel=1e-12), k


def test_shape_mismatch_is_an_error_not_a_broadcast():
    u = np.zeros((10, 8))
    with pytest.raises(ValueError, match='shape'):
        D.dissipation_stats_2d(u, np.zeros((10, 9)), np.zeros((10, 8)),
                               0.01, 0.01, 1e-6, 1e-6)
    U = np.zeros((4, 10, 8))
    with pytest.raises(ValueError, match='shape'):
        D.dissipation_stats_3d(U, U, U, np.zeros((4, 10, 9)),
                               0.01, 0.01, 0.01, 1e-6, 1e-6)
