"""Phase 3 (spectra, velocity statistics): unit tests with known answers, and
live bit-identity against the legacy analyzers on the reference cases.

The regression tests run the legacy phase-3 drivers in-process
(analyze_rt.run_phase3 / _run_phase3_openfoam, analyze_rt_3d.run_phase3) and
rtfractal.pipeline.phase3 into separate temporary directories, then require
every CSV to be byte-identical (pandas writes floats with repr, so equal text
means equal bits).  The legacy 2D drivers write their CSVs from inside the
plotting functions; savefig is stubbed out to save time.
"""
import contextlib
import filecmp
import importlib.util
import io
import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest

from rtfractal import spectra, velocity
from rtfractal.io.case import open_case
from rtfractal.pipeline import phase3

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPOS = os.path.dirname(HERE)
FPAI = f'{REPOS}/FractalParameterAI'
LBM2D = '/media/rod/RT-Simulations-2/LatticeBoltzmann/cases/2D/M0.002/160x200'
SVOF = '/media/rod/RT-Simulations/Dalziel_1999/sola-vof/NDIS-6/160x200'
LBM3D = '/media/rod/RT-Simulations-2/LatticeBoltzmann/cases/3D/80x100x40-M0.002'


def _need(*paths):
    for p in paths:
        if not os.path.exists(p):
            pytest.skip(f'missing {p}')


# ─── unit tests ──────────────────────────────────────────────────────────────

def _grid(nx, ny, L=1.0, H=1.0):
    xc = (np.arange(nx) + 0.5) * L / nx
    yc = (np.arange(ny) + 0.5) * H / ny
    x, y = np.meshgrid(xc, yc, indexing='ij')
    return x, y


def test_next_pow2():
    assert [spectra.next_pow2(n) for n in (1, 2, 3, 64, 65, 160, 1920)] == [1, 2, 4, 64, 128, 256, 2048]


def test_endramp_padding():
    line = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    p = spectra.endramp_pad_1d(line, 8)
    assert np.allclose(p, [1, 3, 2, 5, 4, 3, 2, 1])
    f = np.arange(15.0).reshape(3, 5)                        # (nz, nx)
    q = spectra.pad_endramp_2d(f)
    assert q.shape == (4, 8)
    assert np.array_equal(q[:3, :5], f)
    assert np.allclose(np.diff(q[0, 4:]), (f[0, 0] - f[0, 4]) / 3) and q[0, -1] == f[0, 0]  # ramp ends ON the first value
    assert np.array_equal(q[3, :], q[0, :])                  # one-point z ramp is the first row itself


def test_dalziel_single_mode_peak():
    nx, ny, m = 128, 40, 12
    x, y = _grid(nx, ny)
    f = 0.5 + 0.3 * np.sin(2 * np.pi * m * x)
    d = spectra.dalziel_spectrum_2d(f, x, y, y0=0.5, H=1.0)
    assert d['n_pad'] == 128 and d['n_orig'] == 128
    assert d['n_slab_rows'] == 4                             # rows with 0.4 <= y <= 0.5
    assert d['k_over_k0'][np.argmax(d['power'])] == m
    assert np.isclose(d['power'][m - 1], (0.3 * nx / 2) ** 2 / nx)
    assert d['power'][np.arange(len(d['power'])) != m - 1].max() < 1e-20


def test_dalziel_padded_mode_number():
    """Non-power-of-two rows are padded; k/k0 stays the physical mode number."""
    nx, m = 160, 20
    x, y = _grid(nx, 20)
    d = spectra.dalziel_spectrum_2d(np.sin(2 * np.pi * m * x), x, y, 0.5, 1.0)
    assert d['n_pad'] == 256 and d['k_6dx'] == nx / 6
    assert d['fit_10_25']['fully_resolved'] and not d['fit_10_50']['fully_resolved']
    assert abs(d['k_over_k0'][np.argmax(d['power'])] - m) <= nx / 256


@pytest.mark.parametrize('beta', [5 / 3, 2.2, 3.0])
def test_dalziel_recovers_power_law(beta):
    nx, ny = 1024, 10
    rng = np.random.default_rng(1)
    x, y = _grid(nx, ny)
    m = np.arange(1, nx // 2)
    f = np.zeros((nx, ny))
    for j in range(ny):
        ph = rng.uniform(0, 2 * np.pi, m.size)
        f[:, j] = (m[None, :] ** (-beta / 2) * np.cos(2 * np.pi * m[None, :] * x[:, :1] + ph)).sum(1)
    d = spectra.dalziel_spectrum_2d(f, x, y, 1.0, 1.0)
    for bk in d['band_keys']:
        assert np.isclose(d[bk]['power_law_slope'], -beta, atol=1e-9)
        assert np.isclose(d[bk]['power_law_r_squared'], 1.0, atol=1e-12)
    assert d['power_law_slope'] == d['fit_10_25']['power_law_slope']


def test_field_spectrum_dominant_wavelength():
    nx, ny, m = 128, 128, 8
    x, y = _grid(nx, ny, L=0.4, H=0.4)
    r = spectra.field_spectrum_2d(np.sin(2 * np.pi * m * x / 0.4), x, y, return_spectrum=True)
    assert np.isclose(r['dominant_wavelength'], 0.4 / m, rtol=0.05)
    assert r['n_modes'] == len(r['k'])


def test_field_spectrum_recovers_power_law():
    """Isotropic k^-beta field: slope of the radial spectrum ~ -beta."""
    n, beta = 256, 3.0
    rng = np.random.default_rng(2)
    k = np.fft.fftfreq(n) * n
    K = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    amp = np.where(K > 0, K, 1.0) ** (-beta / 2) * (K > 0)
    f = np.real(np.fft.ifft2(amp * np.exp(2j * np.pi * rng.uniform(size=(n, n)))))
    x, y = _grid(n, n)
    r = spectra.field_spectrum_2d(f, x, y, detrend=False)
    assert abs(r['power_law_slope'] + beta) < 0.15 and r['power_law_r_squared'] > 0.95


def test_tke_spectrum_2d_legacy_quirk():
    x, y = _grid(64, 64)
    u = np.sin(2 * np.pi * 3 * x) * np.cos(2 * np.pi * 2 * y)
    v = np.sin(2 * np.pi * 7 * y)
    us, vs = spectra.field_spectrum_2d(u, x, y, 'u'), spectra.field_spectrum_2d(v, x, y, 'v')
    t = spectra.tke_spectrum_2d(u, v, x, y)
    assert t['power_law_slope'] == us['power_law_slope'] and t['dominant_wavelength'] == us['dominant_wavelength']
    assert t['total_energy'] == (us['total_energy'] + vs['total_energy']) / 2


def test_fit_bands_exact():
    k = np.arange(1.0, 60.0)
    out = spectra.fit_bands(k, 7 * k ** -2.5, spectra.BANDS_3D_TKE, prefix='x_')
    for name, _, _ in spectra.BANDS_3D_TKE:
        assert np.isclose(out[f'x_slope_{name}'], -2.5) and np.isclose(out[f'x_r2_{name}'], 1.0)
    assert np.isnan(spectra.fit_bands(k, k * 0, spectra.BANDS_3D)['slope_10_25'])


def test_horizontal_spectrum_3d_power_law():
    nx = nz = 128
    ny, beta, L = 20, 3.0, 0.4
    rng = np.random.default_rng(3)
    k = np.fft.fftfreq(nx) * nx
    K = np.sqrt(k[None, :] ** 2 + k[:, None] ** 2)
    amp = np.where(K > 0, K, 1.0) ** (-beta / 2) * (K > 0)
    f2 = np.real(np.fft.ifft2(amp * np.exp(2j * np.pi * rng.uniform(size=(nz, nx)))))   # periodic: no padding
    alpha = np.repeat(f2[:, None, :], ny, axis=1)
    yc = (np.arange(ny) + 0.5) / ny
    s = spectra.horizontal_spectrum_3d(alpha, yc, 0.5, 1.0, L / nx, L / nz)
    assert s['n_slab_rows'] == 2 and len(s['P']) == 64
    assert np.isclose(s['k_over_k0'][0], 0.5)                # bin centres in mode numbers
    assert abs(s['slope_10_50'] + beta) < 0.15 and s['r2_10_50'] > 0.98
    U = np.stack([alpha, 2 * alpha, 0 * alpha], axis=-1)
    t = spectra.tke_spectrum_3d(U, yc, 0.5, 1.0, L / nx, L / nz)
    assert np.allclose(t['E_v'], 4 * t['E_u']) and np.allclose(t['E_tke'], 2.5 * t['E_u'])
    assert np.isclose(t['tke_slope_10_50'], t['u_slope_10_50'])
    assert spectra.horizontal_spectrum_3d(alpha, yc, 0.5, 0.1, L / nx, L / nz) is None   # slab < 2 rows


def test_velocity_stats_2d_known_field():
    x, y = _grid(100, 50)
    U0, A = 0.3, 0.2
    u = U0 + A * np.sin(2 * np.pi * x)
    v = np.zeros_like(u)
    s = velocity.velocity_stats_2d(u, v)
    assert np.isclose(s['u_mean'], U0) and s['v_mean'] == 0
    assert np.isclose(s['u_rms'], A / np.sqrt(2)) and s['v_rms'] == 0
    assert np.isclose(s['TKE'], A ** 2 / 4) and np.isclose(s['reynolds_stress'], 0, atol=1e-15)
    assert np.isclose(s['turbulence_intensity'], (A / 2) / U0)
    assert np.isclose(s['velocity_mag_max'], U0 + A, rtol=1e-3)
    s0 = velocity.velocity_stats_2d(u - np.mean(u), v)       # zero mean flow: TI = sqrt(TKE)/|V|_rms
    assert np.isclose(s0['turbulence_intensity'], np.sqrt(s0['TKE']) / s0['velocity_mag_rms'])


def test_velocity_stats_3d_known_moments():
    nz, ny, nx, p = 8, 5, 8, 0.25
    U = np.zeros((nz, ny, nx, 3))
    U[..., 0] = np.where((np.arange(nx) % 2) == 0, 1.0, -1.0)[None, None, :] + 0.7     # +-1 about a mean
    U[..., 1] = np.where((np.arange(nx) % 4) == 0, 1.0, 0.0)[None, None, :] * 2.0      # Bernoulli(p) x 2
    y = (np.arange(ny) + 0.5) / ny
    s = velocity.velocity_stats_3d(U, y, H0=0.5)
    assert np.allclose(s['u_rms'], 1) and np.allclose(s['u_skew'], 0) and np.allclose(s['u_kurt'], 1)
    assert np.allclose(s['v_rms'], 2 * np.sqrt(p * (1 - p)))
    assert np.isclose(s['v_skew_centreline'], (1 - 2 * p) / np.sqrt(p * (1 - p)))
    assert np.isclose(s['v_kurt_centreline'], (1 - 3 * p + 3 * p * p) / (p * (1 - p)))
    assert np.isnan(s['w_skew_centreline']) and s['w_rms_centreline'] == 0
    assert np.isclose(s['anisotropy_v_over_u'], 2 * np.sqrt(p * (1 - p)))


def test_legacy_time_dir_names():
    assert phase3.legacy_time_dir('/c', 2.0) == '/c/2'
    assert phase3.legacy_time_dir('/c', 10.4) == '/c/10.4'


# ─── live bit-identity against the legacy analyzers ──────────────────────────

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def legacy_rt():
    _need(FPAI, f'{REPOS}/FractalAnalyzer')
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()):
        mod = _load('_legacy_analyze_rt', f'{FPAI}/rt_analyzer/analyze_rt.py')
    mod.plt.savefig = lambda *a, **k: None                   # data CSVs are written inside the plot functions
    return mod


def _same_csvs(legacy_dir, new_dir):
    for sub in ('spectra', 'velocity'):
        a, b = os.path.join(legacy_dir, sub), os.path.join(new_dir, sub)
        la = sorted(f for f in os.listdir(a) if f.endswith('.csv'))
        lb = sorted(f for f in os.listdir(b) if f.endswith('.csv'))
        assert la == lb, (sub, la, lb)
        for f in la:
            assert filecmp.cmp(os.path.join(a, f), os.path.join(b, f), shallow=False), f'{sub}/{f} differs'
    return la


TIMES_2D = (2.0, 10.0, 16.0, 40.0)        # 40 s is beyond the data: dropped by the clamp


@pytest.mark.regression
@pytest.mark.parametrize('path', [LBM2D, SVOF])
def test_phase3_2d_bit_identical_to_legacy(legacy_rt, path, tmp_path):
    _need(path)
    m = legacy_rt
    case = open_case(path)
    with contextlib.redirect_stdout(io.StringIO()):
        if case.format == 'solavof':
            phys, _ = m.get_physics_vtk(path)
            physics = m.RTPhysics(A=phys['A'], g=phys['g'], H=phys['H'], L=phys['L'])
            analyzer = m.RTAnalyzer(output_dir=str(tmp_path / 'legacy'), rt_physics=physics)
            files = m.find_vtk_files(path)[0]
            df = pd.DataFrame({'time': [m.vtk_time(f) for f in files]})
            m.run_phase3(analyzer, physics, files, str(tmp_path / 'legacy'), list(TIMES_2D), df)
        else:
            phys = m.get_physics_openfoam(path)
            physics = m.RTPhysics(A=phys['A'], g=phys['g'], H=phys['H'], L=phys['L'])
            analyzer = m.RTAnalyzer(output_dir=str(tmp_path / 'legacy'), rt_physics=physics)
            of_times = [t for t, _ in m.find_openfoam_timesteps(path)]
            df = pd.DataFrame({'time': of_times})
            m._run_phase3_openfoam(analyzer, physics, path, of_times, str(tmp_path / 'legacy'), list(TIMES_2D),
                                   df, phys_nx=phys['NX'], phys_ny=phys['NY'])
    res = phase3.run_2d(case, str(tmp_path / 'new'), times=TIMES_2D, log=lambda *a: None)
    files = _same_csvs(str(tmp_path / 'legacy'), str(tmp_path / 'new'))
    assert len(res['summary']) == 3 and len(res['velocity']) == 3
    assert 'velocity_statistics.csv' in files

    # Reader equivalence for the fields phase 3 uses.
    snap = min(case.snapshots, key=lambda s: abs(s.t - 10.0))
    d = case.read_2d(snap)
    with contextlib.redirect_stdout(io.StringIO()):
        dl = analyzer.read_vtk_file(snap.path) if case.format == 'solavof' else analyzer.read_openfoam_field(path, snap.t)
    for k in ('x', 'y', 'f', 'U', 'V'):
        assert np.array_equal(d[k], dl[k]), k
    assert dl['time'] == snap.t


@pytest.mark.regression
def test_phase3_2d_kernels_match_legacy_classes(legacy_rt):
    """Function-level check against PowerSpectrumAnalyzer / VelocityAnalyzer,
    including every returned key (not just the CSV columns)."""
    _need(LBM2D)
    from rt_power_spectrum import PowerSpectrumAnalyzer
    from rt_velocity_statistics import VelocityAnalyzer
    case = open_case(LBM2D)
    d = case.read_2d(min(case.snapshots, key=lambda s: abs(s.t - 12.0)))
    psa = PowerSpectrumAnalyzer()

    def same(a, b):
        assert set(a) <= set(b)
        for k in a:
            va, vb = a[k], b[k]
            if isinstance(va, dict):
                same(va, vb)
            else:
                assert np.array_equal(np.asarray(va), np.asarray(vb), equal_nan=np.asarray(va).dtype.kind == 'f'), k

    yv = d['y'][0, :]
    H, y0 = float(yv.max() - yv.min()), 0.5 * (float(yv.min()) + float(yv.max()))
    same(psa.analyze_dalziel_horizontal_spectrum(d['f'], d['x'], d['y'], y0, H),
         spectra.dalziel_spectrum_2d(d['f'], d['x'], d['y'], y0, H))
    for fld in ('f', 'U', 'V'):
        same(psa.analyze_2d_field_spectrum(d[fld], d['x'], d['y'], fld, True),
             spectra.field_spectrum_2d(d[fld], d['x'], d['y'], fld, True))
    same(psa.analyze_tke_spectrum(d['U'], d['V'], d['x'], d['y'], True),
         spectra.tke_spectrum_2d(d['U'], d['V'], d['x'], d['y']))
    ref = VelocityAnalyzer().analyze_velocity_field(d['U'], d['V'])
    new = velocity.velocity_stats_2d(d['U'], d['V'])
    for a, b in [('u_mean', 'u_mean'), ('v_mean', 'v_mean'), ('u_rms', 'u_rms'), ('v_rms', 'v_rms'),
                 ('velocity_magnitude_mean', 'velocity_mag_mean'), ('velocity_magnitude_rms', 'velocity_mag_rms'),
                 ('velocity_magnitude_max', 'velocity_mag_max'), ('turbulent_kinetic_energy', 'TKE'),
                 ('reynolds_stress', 'reynolds_stress'), ('turbulence_intensity', 'turbulence_intensity')]:
        assert getattr(ref, a) == new[b], a


TIMES_3D = (6.0, 10.1, 16.0, 40.0)        # 10.1: no such time directory (skipped by both); 40: clamped


@pytest.mark.regression
def test_phase3_3d_bit_identical_to_legacy(tmp_path):
    _need(FPAI, f'{REPOS}/3D-FractalParameterAI', LBM3D)
    with contextlib.redirect_stdout(io.StringIO()):
        m = _load('_legacy_analyze_rt_3d', f'{FPAI}/rt_analyzer/analyze_rt_3d.py')
        mesh = m.read_mesh_info(LBM3D)
        phys = m.get_physics_3d(LBM3D, mesh)
        rtp = m.RTPhysics3D(A=phys['A'], g=phys['g'], H=phys['H'], L=phys['L'], W=phys['W'])
        df = pd.DataFrame({'time': [t for t, _ in m.find_interface_timesteps(LBM3D)]})
        m.run_phase3(LBM3D, mesh, phys, rtp, str(tmp_path / 'legacy'), list(TIMES_3D), df)
    case = open_case(LBM3D)
    res = phase3.run_3d(case, str(tmp_path / 'new'), times=TIMES_3D, log=lambda *a: None)
    _same_csvs(str(tmp_path / 'legacy'), str(tmp_path / 'new'))
    assert list(res['spectrum']['time']) == [6.0, 16.0]
    assert os.path.isfile(tmp_path / 'new' / 'spectra' / 'tke_spectrum_3d_t16.00.csv')


@pytest.mark.regression
def test_phase3_3d_archived_spectrum():
    """Archived legacy output of the reference case, to 1e-12 (library builds differ)."""
    arch = os.path.join(LBM3D, 'analysis', 'spectra', 'spectrum_summary_3d.csv')
    _need(arch)
    case = open_case(LBM3D)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        res = phase3.run_3d(case, td, times=(4.0, 12.0), log=lambda *a: None)
    ref = pd.read_csv(arch).set_index('time').loc[[4.0, 12.0]].reset_index()
    got = res['spectrum']
    assert list(got.columns) == list(ref.columns)
    assert np.allclose(got.values.astype(float), ref.values.astype(float), rtol=1e-12, atol=0)
