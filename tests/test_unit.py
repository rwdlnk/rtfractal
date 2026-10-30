"""Self-contained tests (synthetic data, known answers)."""
import os

import numpy as np
import pytest

from rtfractal import backend
from rtfractal.boxcount import scaling
from rtfractal.boxcount.segments2d import SegmentCounter, box_counting_2d
from rtfractal.boxcount.triangles3d import TriangleCounter, box_counting_3d
from rtfractal.io import openfoam, solavof
from rtfractal.io.case import open_case
from rtfractal.io.mesh import TriangleMesh, load_mesh

needs_cuda = pytest.mark.skipif(not backend.cuda_available(), reason='no CUDA device')


def koch(level, L=0.4, y0=0.25):
    pts = np.array([[0.0, y0], [L, y0]])
    rot = np.array([[0.5, -np.sqrt(3) / 2], [np.sqrt(3) / 2, 0.5]])
    for _ in range(level):
        new = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            d = (b - a) / 3
            new += [a + d, a + d + rot @ d, a + 2 * d, b]
        pts = np.array(new)
    return np.hstack([pts[:-1], pts[1:]])


def plane_mesh(n=40, L=0.4, y=0.25, tilt=0.0):
    xs = np.linspace(0, L, n)
    X, Z = np.meshgrid(xs, xs / 2)
    V = np.c_[X.ravel(), y + tilt * X.ravel(), Z.ravel()]
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    return TriangleMesh.from_arrays(V, np.r_[np.c_[a, b, c], np.c_[b, d, c]])


# ----------------------------------------------------------------- scaling
def test_scaling_recovers_exact_power_law():
    sizes = 0.2 / 1.5 ** np.arange(14)
    res = scaling.dimension_3d(sizes, 7.0 * sizes ** -2.4)
    assert res['dimension'] == pytest.approx(2.4, abs=1e-12)
    res2 = scaling.dimension_2d(sizes, 3.0 * sizes ** -1.3)
    assert res2['dimension'] == pytest.approx(1.3, abs=1e-12)


def test_2d_trim_quirk_always_trims_both_ends():
    """Documented legacy quirk: in 2D r < 0 so the r < 0.95 test always trims."""
    x = np.log(0.2 / 1.5 ** np.arange(12))      # ln(size)
    assert scaling._trim_ends(x, -1.5 * x) == (1, 11)    # 2D form: r = -1, always trimmed
    assert scaling._trim_ends(-x, -1.5 * x) == (0, 12)   # 3D form (1/size): r = +1, perfect line kept


# ------------------------------------------------------------- 2D counting
def test_covering_offsets_recover_koch_dimension():
    segs = koch(7)
    s, n, _ = box_counting_2d(segs, 2 * 0.4 / 640, 0.2, 1.5, 'cpu', offsets='covering')
    assert scaling.dimension_2d(s, n)['dimension'] == pytest.approx(np.log(4) / np.log(3), abs=0.05)


def test_legacy_offsets_bias_high():
    segs = koch(7)
    leg = scaling.dimension_2d(*box_counting_2d(segs, 2 * 0.4 / 640, 0.2, 1.5, 'cpu')[:2])['dimension']
    cov = scaling.dimension_2d(*box_counting_2d(segs, 2 * 0.4 / 640, 0.2, 1.5, 'cpu', offsets='covering')[:2])['dimension']
    assert leg > cov + 0.1


def test_straight_segment_counts():
    seg = np.array([[0.0, 0.0, 1.0, 0.0]])
    c = SegmentCounter(seg, 'cpu')
    assert c.count(0.1, -0.05, -0.05, 1.05, 0.05) == 11


@needs_cuda
def test_segment_counts_cpu_equals_cuda():
    segs = koch(6)
    cpu, gpu = SegmentCounter(segs, 'cpu'), SegmentCounter(segs, 'cuda')
    for d in (0.05, 0.01, 0.003, 0.0011):
        for o in (0.0, 0.37):
            args = (d, -0.001 + o * d, 0.24 + o * d, 0.401, 0.37)
            assert cpu.count(*args) == gpu.count(*args)


# ------------------------------------------------------------- 3D counting
def test_flat_plane_dimension_2_with_covering_offsets():
    m = plane_mesh(tilt=0.05)
    d, n = box_counting_3d(m, initial=0.1, num_steps=8, min_delta=0.004, backend='cpu', offsets='covering')
    assert scaling.dimension_3d(d, n)['dimension'] == pytest.approx(2.0, abs=0.08)


def test_legacy_offsets_lose_flat_surface():
    """A perfectly flat surface lies below every shifted legacy origin."""
    m = plane_mesh()
    with pytest.raises(ValueError):
        box_counting_3d(m, initial=0.1, num_steps=8, min_delta=0.004, backend='cpu')


@needs_cuda
def test_triangle_counts_cpu_equals_cuda():
    m = plane_mesh(tilt=0.3)
    tv = m.triangle_vertices()
    cpu, gpu = TriangleCounter(tv, 'cpu'), TriangleCounter(tv, 'cuda')
    for d in (0.05, 0.01, 0.004):
        lo = np.array([-0.001, 0.2, -0.001])
        hi = np.array([0.401, 0.4, 0.201])
        assert cpu.count(d, lo, hi) == gpu.count(d, lo, hi)


# --------------------------------------------------------------------- I/O
def _write_foam_case(root, nx=8, ny=10, nz=1):
    os.makedirs(f'{root}/constant/polyMesh')
    os.makedirs(f'{root}/system')
    xn, yn = np.linspace(0, 0.4, nx + 1), np.linspace(0, 0.5, ny + 1)
    pts = [f'({v} 0 0)' for v in xn] + [f'(0 {v} 0)' for v in yn[1:]]
    open(f'{root}/constant/polyMesh/points', 'w').write(f'{len(pts)}\n(\n' + '\n'.join(pts) + '\n)\n')
    open(f'{root}/system/blockMeshDict', 'w').write(
        'vertices\n(\n (0 0 0)\n (0.4 0 0)\n (0.4 0.5 0)\n (0 0.5 0)\n'
        f' (0 0 0.2)\n (0.4 0 0.2)\n (0.4 0.5 0.2)\n (0 0.5 0.2)\n);\nblocks\n(\n hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)\n);\n')
    for name, rho in (('water', 1002.2), ('air', 998.0)):
        open(f'{root}/constant/physicalProperties.{name}', 'w').write(f'nu 1e-06;\nrho {rho};\n')
    open(f'{root}/constant/g', 'w').write('value (0 -9.81 0);\n')
    yc = 0.5 * (yn[:-1] + yn[1:])
    f = np.repeat((yc > 0.25).astype(float)[:, None], nx, axis=1)
    os.makedirs(f'{root}/1')
    open(f'{root}/1/alpha.water', 'w').write(
        f'internalField   nonuniform List<scalar>\n{f.size}\n(\n' + '\n'.join(f'{v:g}' for v in f.ravel()) + '\n)\n;\n')
    return f


def test_openfoam_2d_roundtrip(tmp_path):
    f = _write_foam_case(str(tmp_path))
    c = open_case(str(tmp_path))
    assert (c.format, c.dim, c.nx, c.ny) == ('openfoam', 2, 8, 10)
    assert c.physics.A == pytest.approx(4.2 / 2000.2)
    d = c.read_2d(c.snapshots[0])
    assert np.array_equal(d['f'], f.T)
    assert d['x'][1, 0] - d['x'][0, 0] == pytest.approx(0.05)


def test_legacy_vtk_mesh_reader_rounds_float_points(tmp_path):
    p = tmp_path / 'interface.vtk'
    p.write_text('# vtk DataFile Version 2.0\nt\nASCII\nDATASET POLYDATA\nPOINTS 4 float\n'
                 '0 0 0 0.1 0 0 0.1 0.1 0 0 0.1 0.300000001\nPOLYGONS 1 5\n4 0 1 2 3\n')
    m = load_mesh(str(p))
    assert m.n_triangles == 2
    assert m.vertices[3, 2] == float(np.float32(0.300000001))


def test_solavof_input_parser(tmp_path):
    p = tmp_path / 'int200.in'
    p.write_text('NU (F15.6)   =  0.000001044\nGY(F10.4)    =  -9.81\nRHOF(F11.5)  =  1002.2\n'
                 'RHOFC(F11.5) =  998.0\n-XC-!-NXL!-NXR!-DXMN----!\n 0.20   80   80  2.50e-3\n'
                 '-YC-!-NYL!-NYR!-DYMN----!\n 0.25  100  100  2.50e-3\n'
                 '!BELOW ARE THE VALUES FOR XL.\n----!\n   0.  0.4\n!BELOW ARE THE VALUES FOR YL.\n----!\n   0.  0.5\n'
                 '!Enter the disturbance parameters R,H,ILOW,IHIGH\n!--------!\n       0.4       .25    1   90\n')
    phys = solavof.derive_physics(solavof.parse_input(str(p)))
    assert (phys['NX'], phys['NY'], phys['H0'], phys['L'], phys['H']) == (160, 200, 0.25, 0.4, 0.5)
