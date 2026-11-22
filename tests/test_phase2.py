"""Phase 2 (multifractal): known answers, CPU = CUDA, and bit-identity with
the legacy analyzers run in-process on the reference cases.

Legacy references:
  2D  RTAnalyzer.compute_multifractal_spectrum (FractalParameterAI/rt_analyzer),
      GPU path, which is what analyze_rt.py runs on a CUDA machine (integer
      segment counts, so it is deterministic).
  3D  fractal3d MultifractalAnalyzer3D.  Its GPU path sums cube areas with
      float atomics, so it is not reproducible bit for bit (run to run it
      varies by ~1e-15), and its SAT test is FMA-contracted, which assigns
      triangles lying on a cube face differently from the CPU path (tau(q)
      differs by up to ~1e-4).  rtfractal reproduces the legacy CPU path
      (use_gpu='never') bit for bit on both backends; against the legacy GPU
      path the occupied counts, the box sizes and D0 are identical and the
      rest agrees to 1e-3.
"""
import contextlib
import functools
import io
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from rtfractal import backend, interface, multifractal
from rtfractal.shapes import koch
from rtfractal.boxcount.measures2d import SegmentMeasure
from rtfractal.boxcount.measures3d import TriangleMeasure, grid_optimized_measure
from rtfractal.io.case import open_case
from rtfractal.io.mesh import TriangleMesh
from rtfractal.pipeline import phase2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPOS = os.path.dirname(HERE)
LBM2D = '/media/rod/RT-Simulations-2/LatticeBoltzmann/cases/2D/M0.002/160x200'
SVOF = '/media/rod/RT-Simulations/Dalziel_1999/sola-vof/NDIS-6/160x200'
LBM3D = '/media/rod/RT-Simulations-2/LatticeBoltzmann/cases/3D/80x100x40-M0.002'
LEGACY = dict(offsets='legacy', ladder='legacy', trim='legacy')
Q = np.arange(-5, 5.1, 0.5)
ARRAYS = ('tau', 'Dq', 'alpha', 'f_alpha', 'r_squared', 'tau_raw', 'Dq_raw')
SCALARS = ('D0', 'D1', 'D2', 'alpha_width', 'degree_multifractality', 'convexity_adjustment', 'convexity_collapsed')
BACKENDS = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not backend.cuda_available(), reason='no CUDA'))]
needs_cuda = pytest.mark.skipif(not backend.cuda_available(), reason='no CUDA device')

warnings.filterwarnings('ignore', message='.*Grid size.*')


def _need(*paths):
    for p in paths:
        if not os.path.exists(p):
            pytest.skip(f'missing {p}')


def _same(a, b):
    return a == b or (np.isnan(a) and np.isnan(b))


def _tmp(name):
    d = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'rtfractal_phase2_tests', name)
    os.makedirs(d, exist_ok=True)
    return d


# ===================================================================== unit
def plane_mesh(n=60, L=0.4, y=0.25, tilt=0.0):
    xs = np.linspace(0, L, n)
    X, Z = np.meshgrid(xs, xs / 2)
    V = np.c_[X.ravel(), y + tilt * X.ravel(), Z.ravel()]
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    return TriangleMesh.from_arrays(V, np.r_[np.c_[a, b, c], np.c_[b, d, c]])


def binomial(p, k):
    m = np.array([1.0])
    for _ in range(k):
        m = np.concatenate([m * p, m * (1 - p)])
    return m


@pytest.mark.parametrize('spectrum', [multifractal.spectrum_2d, multifractal.spectrum_3d])
def test_binomial_cascade_spectrum(spectrum):
    """Exact binomial measure at dyadic sizes: tau(q) = -log2(p^q + (1-p)^q)."""
    p, ks = 0.3, range(3, 13)
    res = spectrum(2.0 ** -np.array(ks, dtype=float), [binomial(p, k) for k in ks], Q)
    tau_exact = -np.log2(p ** Q + (1 - p) ** Q)
    assert np.allclose(res['tau_raw'], tau_exact, atol=1e-10)
    assert res['convexity_adjustment'] < 1e-8               # tau(q) already concave: projection is a no-op
    assert res['D0'] == pytest.approx(1.0, abs=1e-10)
    assert res['D2'] == pytest.approx(-np.log2(p ** 2 + (1 - p) ** 2), abs=1e-8)
    assert res['D1'] == pytest.approx(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)), abs=0.02)   # central difference
    a_lo, a_hi = [-(p ** q * np.log2(p) + (1 - p) ** q * np.log2(1 - p)) / (p ** q + (1 - p) ** q) for q in (5, -5)]
    assert res['alpha_width'] == pytest.approx(a_hi - a_lo, abs=0.02)
    assert not res['convexity_collapsed']


def test_koch_curve_is_monofractal_with_covering_offsets():
    segs = koch(7)
    sizes = 0.4 / 8 / 1.5 ** np.arange(12)
    res = multifractal.multifractal_2d(segs, sizes, Q, 'cpu', 'covering')
    D = np.log(4) / np.log(3)
    assert res['D0'] == pytest.approx(D, abs=0.04)
    pos = Q >= 0
    assert np.nanmax(np.abs(res['Dq_raw'][pos] - res['D0'])) < 0.06     # D_q ~ constant before projection too


def test_legacy_offsets_bias_d0_high():
    segs = koch(7)
    sizes = 0.4 / 8 / 1.5 ** np.arange(12)
    leg = multifractal.multifractal_2d(segs, sizes, Q, 'cpu', 'legacy')['D0']
    cov = multifractal.multifractal_2d(segs, sizes, Q, 'cpu', 'covering')['D0']
    assert leg > cov + 0.05


def test_plane_is_monofractal_d2_with_covering_offsets():
    m = plane_mesh(n=201, tilt=0.05)                 # triangles (2 mm) well below the smallest cube (6.6 mm)
    res = multifractal.multifractal_3d(m, 0.05 / 1.5 ** np.arange(6), Q, 'area', 'cpu', 'covering')
    assert res['D0'] == pytest.approx(2.0, abs=0.06)
    assert np.nanmax(np.abs(res['Dq_raw'][Q >= 0] - 2.0)) < 0.06


def test_area_measure_conserves_area_and_matches_legacy_dict_order():
    """Measures sum to the surface area; order and values follow the legacy CPU
    loop (dict filled triangle by triangle)."""
    m = plane_mesh(n=12, tilt=0.37)
    tm = TriangleMeasure(m, 'area', 'cpu')
    bb, d = m.bbox, 0.03
    dmin = np.array([bb.min_x - 0.2 * d, bb.min_y - 0.2 * d, bb.min_z - 0.2 * d])
    dmax = np.array([bb.max_x, bb.max_y, bb.max_z])
    got = tm.measures(d, dmin, dmax)
    assert got.sum() == pytest.approx(m.surface_area, rel=1e-12)
    nx, ny, nz = (max(1, int(np.ceil((dmax[i] - dmin[i]) / d))) for i in range(3))
    h, off = tm.hits(d, dmin, nx, ny, nz)
    ref, areas = {}, m.triangle_areas()
    for t in range(m.n_triangles):
        cells = h[off[t]:off[t + 1]]
        for c in cells:
            ref[c] = ref.get(c, 0.0) + areas[t] / len(cells)
    assert np.array_equal(got, np.array(list(ref.values())))


def test_flat_surface_lost_by_legacy_offsets_has_zero_scales():
    """A flat surface below every shifted legacy origin: empty measures at those
    sizes, which the spectrum drops (legacy behaviour)."""
    m = plane_mesh(n=20)
    occ, n, _ = grid_optimized_measure(TriangleMeasure(m, 'area', 'cpu'), 0.05, m.bbox, 'legacy')
    assert n == 0 and occ.size == 0


@needs_cuda
def test_segment_measure_cpu_equals_cuda():
    segs = koch(6)
    cpu, gpu = SegmentMeasure(segs, 'cpu'), SegmentMeasure(segs, 'cuda')
    for d in (0.05, 0.01, 0.003, 0.0011):
        for o in (0.0, 0.37):
            args = (d, -0.001 + o * d, 0.24 + o * d, 0.401, 0.37)
            a, b = cpu.box_counts(*args)[0], gpu.box_counts(*args)[0]
            assert np.array_equal(a, b)


@needs_cuda
def test_triangle_measure_cpu_equals_cuda_on_grid_aligned_faces():
    """Triangles lying exactly on cube faces (an FMA-contracted CUDA SAT test
    disagrees with the CPU there); the measures must be bit-identical."""
    m = plane_mesh(n=41, L=0.4, y=0.25, tilt=0.0)
    m2 = plane_mesh(n=33, tilt=0.3)
    for mesh in (m, m2):
        cpu, gpu = TriangleMeasure(mesh, 'area', 'cpu'), TriangleMeasure(mesh, 'area', 'cuda')
        bb = mesh.bbox
        for d in (0.05, 0.02, 0.01):
            for f in (0.0, 0.25, 0.5):
                dmin = np.array([bb.min_x - f * d, bb.min_y - f * d, bb.min_z - f * d])
                dmax = np.array([bb.max_x, bb.max_y, bb.max_z])
                assert np.array_equal(cpu.measures(d, dmin, dmax), gpu.measures(d, dmin, dmax))


def test_phase2_run_2d_writes_legacy_layout(tmp_path):
    """Synthetic OpenFOAM-layout case: file names and columns."""
    from test_unit import _write_foam_case
    _write_foam_case(str(tmp_path / 'case'), nx=40, ny=50)
    import shutil
    c0 = str(tmp_path / 'case')
    # a wavy interface so the contour is not a straight line
    xc = (np.arange(40) + 0.5) * 0.01
    yc = (np.arange(50) + 0.5) * 0.01
    f = (yc[:, None] > 0.25 + 0.03 * np.sin(2 * np.pi * xc[None, :] / 0.1)).astype(float)
    for t in ('2', '4'):
        os.makedirs(f'{c0}/{t}', exist_ok=True)
        open(f'{c0}/{t}/alpha.water', 'w').write(
            f'internalField   nonuniform List<scalar>\n{f.size}\n(\n' + '\n'.join(f'{v:g}' for v in f.ravel()) + '\n)\n;\n')
    shutil.rmtree(f'{c0}/1')
    out = tmp_path / 'out'
    df = phase2.run_2d(open_case(c0), str(out), times=(2, 4), backend='cpu', log=lambda s: None)
    assert list(df.columns) == ['time', 'tau', 'h_H', 'D0', 'D1', 'D2', 'alpha_width', 'degree_multifractality']
    assert len(df) == 2 and df['h_H'].isna().all()
    r = pd.read_csv(out / 'multifractal' / 'time_2.0' / 'multifractal_results.csv')
    assert list(r.columns) == ['q', 'tau', 'Dq', 'alpha', 'f_alpha', 'r_squared', 'tau_raw', 'Dq_raw']
    p = pd.read_csv(out / 'multifractal' / 'time_4.0' / 'multifractal_parameters.csv')
    assert list(p['Parameter'])[:4] == ['Time', 'D0', 'D1', 'D2']


# =============================================================== regression
@pytest.fixture(scope='module')
def legacy2d(legacy_rtanalyzer):
    """The analyzer pinned in conftest.py, not whatever sits in the sibling tree."""
    with contextlib.redirect_stdout(io.StringIO()):
        return legacy_rtanalyzer(output_dir=_tmp('legacy_rt_out'))


def _legacy_phase1_boxes(A, dl, xc):
    """What legacy phase 1 writes to fractal/boxcount_t*.csv (None if D is NaN)."""
    mx, mn = float(xc[-1] - xc[0]) / 2, float(xc[1] - xc[0]) * 2
    ref = A.compute_fractal_dimension(dl, min_box_size=mn, max_box_size=mx, box_size_factor=1.5)
    return None if np.isnan(ref['dimension']) else ref['box_sizes']


def _check_2d(legacy2d, path, t, be, phase1_boxes=True):
    c = open_case(path)
    snap = min(c.snapshots, key=lambda s: abs(s.t - t))
    d = c.read_2d(snap)
    xc = d['x'][:, 0]
    with contextlib.redirect_stdout(io.StringIO()):
        dl = legacy2d.read_vtk_file(snap.path) if c.format == 'solavof' else legacy2d.read_openfoam_field(path, snap.t)
        boxes = _legacy_phase1_boxes(legacy2d, dl, xc) if phase1_boxes else None
        ref = legacy2d.compute_multifractal_spectrum(dl, q_values=Q, box_sizes=boxes)
    assert dl['time'] == snap.t
    segs = interface.segments(interface.contours(d['f'], d['x'], d['y'], 0.5))
    if phase1_boxes:
        sizes, src = phase2.box_sizes_2d(segs, xc, **LEGACY)
        assert src == ('phase1' if boxes is not None else 'legacy-fallback')
        if boxes is not None:
            assert np.array_equal(sizes, boxes)
    else:
        extent, _ = multifractal.segments_bbox(segs)
        sizes = [extent / 2]
        while sizes[-1] / 1.5 >= 0.001:
            sizes.append(sizes[-1] / 1.5)
    new = multifractal.multifractal_2d(segs, sizes, Q, be, 'legacy')
    for k in ARRAYS:
        assert np.array_equal(ref[k], new[k], equal_nan=True), k
    for k in SCALARS:
        assert _same(ref[k], new[k]), k


@pytest.mark.regression
@pytest.mark.parametrize('path', [LBM2D, SVOF])
@pytest.mark.parametrize('t', [4.0, 10.0, 16.0])
def test_2d_spectrum_bit_identical_to_legacy(legacy2d, path, t):
    _need(path)
    _check_2d(legacy2d, path, t, None)


@pytest.mark.regression
@pytest.mark.parametrize('be', BACKENDS)
def test_2d_spectrum_bit_identical_each_backend(legacy2d, be):
    _need(SVOF)
    _check_2d(legacy2d, SVOF, 8.0, be)


@pytest.mark.regression
def test_2d_legacy_fallback_ladder_bit_identical(legacy2d):
    """No phase-1 box-count file: legacy uses extent/2 -> 0.001 m."""
    _need(LBM2D)
    _check_2d(legacy2d, LBM2D, 10.0, None, phase1_boxes=False)


def _files_equal(a, b):
    with open(a, 'rb') as fa, open(b, 'rb') as fb:
        return fa.read() == fb.read()


@pytest.mark.regression
@pytest.mark.parametrize('path', [LBM2D, SVOF])
def test_2d_pipeline_files_identical_to_legacy_driver(legacy2d, path):
    """analyze_rt.run_phase2 / _run_phase2_openfoam, fed the phase-1 box-count
    files legacy phase 1 would write, vs phase2.run_2d with legacy options:
    every CSV byte for byte."""
    _need(path)
    import matplotlib
    matplotlib.use('Agg')
    import analyze_rt as AR
    times = [4.0, 16.0]
    c = open_case(path)
    leg_dir, new_dir = _tmp(f'leg2d_{c.format}'), _tmp(f'new2d_{c.format}')
    os.makedirs(os.path.join(leg_dir, 'fractal'), exist_ok=True)
    xc = c.read_2d(c.snapshots[0], fields=('F',))['x'][:, 0]
    df = pd.DataFrame({'time': [s.t for s in c.snapshots],
                       'h_total_geo_nondim': np.linspace(0, 1, len(c.snapshots))})
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        phys = AR.RTPhysics(A=c.physics.A, g=c.physics.g, H=c.physics.H, L=c.physics.L)
        an = AR.RTAnalyzer(output_dir=leg_dir, rt_physics=phys)
        for t in times:
            snap = min(c.snapshots, key=lambda s: abs(s.t - t))
            dl = an.read_vtk_file(snap.path) if c.format == 'solavof' else an.read_openfoam_field(path, snap.t)
            b = _legacy_phase1_boxes(an, dl, xc)
            if b is not None:
                pd.DataFrame({'box_size': b}).to_csv(
                    os.path.join(leg_dir, 'fractal', f'boxcount_t{snap.time_idx:05d}.csv'), index=False)
        if c.format == 'solavof':
            vtk_files, _, _ = AR.find_vtk_files(path)
            AR.run_phase2(an, phys, vtk_files, leg_dir, times, df)
        else:
            AR._run_phase2_openfoam(an, phys, path, [t for t, _ in AR.find_openfoam_timesteps(path)], leg_dir, times, df)
    phase2.run_2d(c, new_dir, times=times, df=df, log=lambda s: None, **LEGACY)
    lm, nm = os.path.join(leg_dir, 'multifractal'), os.path.join(new_dir, 'multifractal')
    assert _files_equal(os.path.join(lm, 'multifractal_summary.csv'), os.path.join(nm, 'multifractal_summary.csv'))
    subdirs = sorted(e for e in os.listdir(lm) if os.path.isdir(os.path.join(lm, e)))
    assert len(subdirs) == len(times)
    for sd in subdirs:
        for f in ('multifractal_results.csv', 'multifractal_parameters.csv'):
            assert _files_equal(os.path.join(lm, sd, f), os.path.join(nm, sd, f)), (sd, f)


@pytest.fixture(scope='module')
def legacy3d(legacy_rt_analyzer_dir):
    """fractal3d, with the pinned rt_analyzer beside it for its rt_physics."""
    _need(f'{REPOS}/3D-FractalParameterAI', LBM3D)
    sys.path.insert(0, f'{REPOS}/3D-FractalParameterAI')
    sys.path.insert(0, legacy_rt_analyzer_dir)
    import fractal3d
    return fractal3d


def _legacy3d_spectrum(fractal3d, mesh, sizes, use_gpu):
    from fractal3d.multifractal_3d import MultifractalAnalyzer3D
    return MultifractalAnalyzer3D(measure='area', verbose=False, use_gpu=use_gpu).compute_multifractal_spectrum(
        mesh, q_values=Q, cube_sizes=sizes)


@pytest.mark.regression
@pytest.mark.parametrize('t', [4.0, 10.0])
def test_3d_spectrum_bit_identical_to_legacy_cpu_path(legacy3d, t):
    c = open_case(LBM3D)
    snap = min(c.snapshots, key=lambda s: abs(s.t - t))
    lm, nm = legacy3d.load_mesh(snap.iface), c.interface_mesh(snap)
    sizes, src = phase2.cube_sizes_3d(c, nm, **LEGACY)
    assert src == 'phase1'
    ref = _legacy3d_spectrum(legacy3d, lm, sizes, 'never')
    for be in ['cpu'] + (['cuda'] if backend.cuda_available() else []):
        new = multifractal.multifractal_3d(nm, sizes, Q, 'area', be, 'legacy')
        assert list(ref.n_occupied) == list(new['n_occupied'])
        for k in ARRAYS:
            assert np.array_equal(getattr(ref, k), new[k], equal_nan=True), (be, k)
        for k in SCALARS:
            assert _same(getattr(ref, k), new[k]), (be, k)


@needs_cuda
@pytest.mark.regression
@pytest.mark.parametrize('t', [2.0, 16.0])
def test_3d_spectrum_matches_legacy_gpu_path(legacy3d, t):
    """The production (GPU) legacy path, including t = 2 where legacy phase 1
    fails with legacy offsets and the surface-feature fallback ladder is used."""
    from fractal3d.surface_features import extract_surface_features, suggest_parameters
    c = open_case(LBM3D)
    snap = min(c.snapshots, key=lambda s: abs(s.t - t))
    lm, nm = legacy3d.load_mesh(snap.iface), c.interface_mesh(snap)
    sizes, src = phase2.cube_sizes_3d(c, nm, **LEGACY)
    from fractal3d.multifractal_3d import MultifractalAnalyzer3D
    A = MultifractalAnalyzer3D(measure='area', verbose=False, use_gpu='always')
    if src.startswith('legacy-fallback'):
        prm = suggest_parameters(extract_surface_features(lm, compute_curvature=False))
        ref = A.compute_multifractal_spectrum(lm, q_values=Q, max_delta=prm['initial_delta'],
                                              delta_factor=prm['delta_factor'], num_scales=prm['num_steps'],
                                              min_delta=lm.bbox.characteristic_length / 500)
    else:
        ref = A.compute_multifractal_spectrum(lm, q_values=Q, cube_sizes=sizes)
    assert np.array_equal(ref.cube_sizes, sizes)
    new = multifractal.multifractal_3d(nm, sizes, Q, 'area', 'cuda', 'legacy')
    assert list(ref.n_occupied) == list(new['n_occupied'])
    assert ref.D0 == new['D0']
    for k in ('tau', 'Dq', 'alpha'):
        assert np.allclose(getattr(ref, k), new[k], atol=1e-3, equal_nan=True), k


@pytest.mark.regression
def test_3d_pipeline_files_identical_to_legacy_driver(legacy3d):
    """analyze_rt_3d.run_phase2 (forced onto its deterministic CPU path), fed the
    box-count files legacy phase 1 writes, vs phase2.run_3d with legacy options."""
    import matplotlib
    matplotlib.use('Agg')
    import analyze_rt_3d as AR3
    from fractal3d.multifractal_3d import MultifractalAnalyzer3D
    times = [4.0, 10.0]
    c = open_case(LBM3D)
    p, g = c.params, c.grid
    leg_dir, new_dir = _tmp('leg3d'), _tmp('new3d')
    os.makedirs(os.path.join(leg_dir, 'fractal'), exist_ok=True)
    for t in times:
        snap = min(c.snapshots, key=lambda s: abs(s.t - t))
        with contextlib.redirect_stdout(io.StringIO()):
            r = legacy3d.compute_fractal_dimension_3d(legacy3d.load_mesh(snap.iface), initial_delta=min(p['L'], p['W']) / 2,
                                                      delta_factor=1.5, num_steps=20,
                                                      min_delta=max(g.dx, g.dy, g.dz) * 2)
        pd.DataFrame({'cube_size': r.deltas}).to_csv(os.path.join(leg_dir, 'fractal', f'boxcount_3d_t{snap.t:.2f}.csv'),
                                                     index=False)
    df = pd.DataFrame({'time': [s.t for s in c.snapshots], 'h_total_int_nondim': np.linspace(0, 1, len(c.snapshots))})
    phys = {'H': p['H']}
    rtp = legacy3d.RTPhysics3D(A=p['A'], g=p['g'], H=p['H'], L=p['L'], W=p['W'])
    orig = AR3.MultifractalAnalyzer3D
    AR3.MultifractalAnalyzer3D = functools.partial(MultifractalAnalyzer3D, use_gpu='never')
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            AR3.run_phase2(LBM3D, phys, rtp, AR3.find_interface_timesteps(LBM3D), leg_dir, times, df)
    finally:
        AR3.MultifractalAnalyzer3D = orig
    phase2.run_3d(c, new_dir, times=times, log=lambda s: None, **LEGACY)
    lm, nm = os.path.join(leg_dir, 'multifractal'), os.path.join(new_dir, 'multifractal')
    names = sorted(f for f in os.listdir(lm) if f.endswith('.csv'))
    assert len(names) == 1 + 3 * len(times)
    for f in names:
        assert _files_equal(os.path.join(lm, f), os.path.join(nm, f)), f


def test_select_times_warns_on_dropped_and_far_targets(capsys):
    from types import SimpleNamespace
    from rtfractal.pipeline.phase2 import _select_times
    snaps = [SimpleNamespace(t=0.0452 * i) for i in range(40)]     # last t = 1.763
    assert _select_times([2, 4, 6], snaps, print) == [2]
    err = capsys.readouterr().err
    assert '2 of 3 target times dropped' in err and '2->1.763' in err
    assert _select_times([0.4071, 1.593], snaps, print) == [0.4071, 1.593]
    assert capsys.readouterr().err == ''
