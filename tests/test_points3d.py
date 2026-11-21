"""The 3D point-cloud box counter.

Measured here, so the numbers quoted in the module and the README are not
assertions of faith.

Exact recovery, on a ladder commensurate with the construction:

    3D Cantor dust, level 6, factor 3   D = 1.8928   exact 3 log2/log3 = 1.8928
                                        R^2 = 1.0000, counts 8, 64, 512, 4096

That one also shows why the ladder factor is not cosmetic.  The same set on
factor 1.5 reads +0.069 and on factor 2.0 reads **+0.469**, because a strictly
self-similar set has a staircase box count: it jumps by 8 each time delta
crosses a factor of 3 and is flat between, so an incommensurate ladder
aliases.  Physical interfaces are not strictly self-similar and the default
1.5 is fine for them, but a textbook set wants a commensurate ladder.

Other sets, default ladder (2x spacing to extent/8, factor 1.5):

    Sierpinski tetrahedron L7   32770 pts   exact 2.0000   2.0038  +0.004
    Menger sponge L5          3200000 pts   exact 2.7268   2.7918  +0.065  (slow, not run here)
    regularly sampled plane    160000 pts   exact 2.0000   2.0087  +0.009
    sphere surface             200000 pts   exact 2.0000   1.9402  -0.060
    uniform cube               300000 pts   exact 3.0000   2.8981  -0.102

The near-planar history: the legacy D = 5 to 9 at R^2 ~ 0.8 belongs to the
LEGACY OFFSETS, not to the triangle measure.  A flat surface lies below every
shifted origin and the counts come out zero, which is asserted below.  With
covering offsets the triangle counter handles a flat plane fine (2.018), so
the point counter does not rescue a broken case: it is a second, independent
measure that agrees with the corrected one where both are valid.
"""
import numpy as np
import pytest

from rtfractal import backend, fractal
from rtfractal.boxcount import points3d, scaling
from rtfractal.io.mesh import TriangleMesh

needs_cuda = pytest.mark.skipif(not backend.cuda_available(), reason='no CUDA device')


def cantor_dust(level):
    """Product of three middle-thirds Cantor sets.  D = 3 log2 / log3."""
    c = np.array([0.0])
    for L in range(level):
        c = np.concatenate([c, c + 2.0 / 3.0 ** (L + 1)])
    g = c + 0.5 / 3.0 ** level
    X, Y, Z = np.meshgrid(g, g, g, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def sierpinski_tetrahedron(level):
    v = np.array([[0., 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0],
                  [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3]])
    pts = v.copy()
    for _ in range(level):
        pts = np.vstack([(pts + c) / 2 for c in v])
    return np.unique(pts, axis=0)


def plane(n, amp=0.0, L=0.4, y=0.25):
    """-> (points, triangle mesh) of the same surface, for comparing measures."""
    g = np.linspace(0, L, n)
    X, Z = np.meshgrid(g, g)
    Y = y + amp * np.sin(2 * np.pi * X / 0.1) * np.cos(2 * np.pi * Z / 0.1)
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    tris = np.vstack([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return P, TriangleMesh.from_arrays(P, tris)


# ------------------------------------------------------------ exact recovery
def test_cantor_dust_is_exact_on_a_commensurate_ladder():
    pts = cantor_dust(6)
    d, c, _ = points3d.box_counting_3d_points(pts, initial=1 / 3, factor=3.0, num_steps=30,
                                              backend='cpu')
    assert list(c) == [8, 64, 512, 4096]                 # exactly 8^k
    r = scaling.dimension_3d(d, c)
    assert r['dimension'] == pytest.approx(3 * np.log(2) / np.log(3), abs=1e-6)
    assert r['r_squared'] == pytest.approx(1.0, abs=1e-9)


def test_an_incommensurate_ladder_aliases_a_self_similar_set():
    """Why the ladder factor is not cosmetic.  The same set, exact 1.8928."""
    pts = cantor_dust(6)
    got = {}
    for f in (1.5, 2.0, 3.0):
        d, c, _ = points3d.box_counting_3d_points(pts, initial=1 / 3, factor=f, num_steps=30,
                                                  backend='cpu')
        got[f] = scaling.dimension_3d(d, c)['dimension']
    exact = 3 * np.log(2) / np.log(3)
    assert abs(got[3.0] - exact) < 1e-6
    assert abs(got[1.5] - exact) < 0.1
    assert abs(got[2.0] - exact) > 0.3                   # aliased badly


def test_sierpinski_tetrahedron():
    r = fractal.dimension_of_points_3d(sierpinski_tetrahedron(7), backend='cpu')
    assert r['dimension'] == pytest.approx(2.0, abs=0.02)
    assert r['r_squared'] > 0.99


# --------------------------------------------------------- surfaces and solids
@pytest.mark.parametrize('amp', [0.0, 1e-4, 1e-3])
def test_a_flat_or_nearly_flat_plane_reads_two(amp):
    pts, _ = plane(200, amp)
    r = fractal.dimension_of_points_3d(pts, min_delta=2 * 0.4 / 200, backend='cpu')
    assert r['dimension'] == pytest.approx(2.0, abs=0.02)
    assert r['r_squared'] > 0.99


def test_a_sphere_surface_reads_two():
    rng = np.random.default_rng(0)
    u, v = rng.random(200000), rng.random(200000)
    th, ph = np.arccos(1 - 2 * u), 2 * np.pi * v
    pts = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)
    r = fractal.dimension_of_points_3d(pts, backend='cpu')
    assert r['dimension'] == pytest.approx(2.0, abs=0.08)


def test_a_uniform_cube_approaches_three():
    rng = np.random.default_rng(0)
    r = fractal.dimension_of_points_3d(rng.random((300000, 3)), backend='cpu')
    assert 2.8 < r['dimension'] < 3.02


# -------------------------------------------- the near-planar history, measured
def test_the_legacy_offsets_are_what_fail_on_a_flat_surface():
    """The documented D = 5 to 9 at R^2 ~ 0.8 is the legacy offsets, not the
    triangle measure: a flat surface lies below every shifted origin and the
    counts come out zero."""
    _, mesh = plane(200, 0.0)
    with pytest.raises(ValueError, match='non-zero cube counts|scale range'):
        fractal.dimension_3d(mesh, initial=0.4 / 8, factor=1.5, num_steps=20,
                             min_delta=2 * 0.4 / 200, backend='cpu', offsets='legacy')


def test_covering_offsets_already_fix_the_flat_surface():
    _, mesh = plane(200, 0.0)
    r = fractal.dimension_3d(mesh, initial=0.4 / 8, factor=1.5, num_steps=20,
                             min_delta=2 * 0.4 / 200, backend='cpu', offsets='covering')
    assert r['dimension'] == pytest.approx(2.0, abs=0.05)
    assert r['r_squared'] > 0.97              # passes the admission gate


def test_point_and_triangle_measures_agree_on_a_nearly_flat_surface():
    """Agreement is a check, not a redundancy: the two measure different sets
    once the surface carries structure near the sampling scale."""
    pts, mesh = plane(200, 1e-3)
    kw = dict(initial=0.4 / 8, factor=1.5, num_steps=20, min_delta=2 * 0.4 / 200,
              backend='cpu', offsets='covering')
    d_tri = fractal.dimension_3d(mesh, **kw)['dimension']
    d_pts = fractal.dimension_of_points_3d(pts, **kw)['dimension']
    assert d_pts == pytest.approx(d_tri, abs=0.02)


# ------------------------------------------------------------- the hard bound
def test_a_ladder_below_the_spacing_is_refused():
    pts = sierpinski_tetrahedron(6)
    sp = points3d.nn_spacing(pts)
    with pytest.raises(ValueError, match='below the nearest-neighbour spacing'):
        points3d.box_counting_3d_points(pts, initial=0.125, min_delta=sp / 4, backend='cpu')


def test_below_the_spacing_the_count_saturates_and_there_is_no_dimension():
    pts = sierpinski_tetrahedron(6)
    sp = points3d.nn_spacing(pts)
    d, c, _ = points3d.box_counting_3d_points(pts, initial=sp / 4, factor=1.5, num_steps=10,
                                              min_delta=sp / 100, allow_below_spacing=True,
                                              backend='cpu')
    assert (c == len(pts)).all()
    assert np.isnan(scaling.dimension_3d(d, c)['dimension'])


def test_a_known_spacing_can_be_supplied():
    pts = sierpinski_tetrahedron(6)
    d, c, _ = points3d.box_counting_3d_points(pts, initial=0.125, min_delta=0.02,
                                              spacing=0.01, backend='cpu')
    assert len(d) >= 3


# ------------------------------------------------------------------- plumbing
def test_spacing_of_a_regular_3d_lattice():
    g = np.linspace(0, 1, 11)
    pts = np.stack(np.meshgrid(g, g, g), axis=-1).reshape(-1, 3)
    assert points3d.nn_spacing(pts) == pytest.approx(0.1)


def test_shape_validation():
    with pytest.raises(ValueError, match=r'\(n, 3\)'):
        points3d.PointCounter3D(np.zeros((5, 2)))
    with pytest.raises(ValueError, match=r'\(n, 3\)'):
        points3d.box_counting_3d_points(np.zeros((5, 2)))
    with pytest.raises(ValueError, match='at least 2 points'):
        points3d.box_counting_3d_points(np.zeros((1, 3)))


def test_counts_never_exceed_the_point_count():
    pts = sierpinski_tetrahedron(6)
    counter = points3d.PointCounter3D(pts, 'cpu')
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([1.0, 0.9, 0.9])
    for delta in (0.1, 0.03, 0.01):
        assert 0 < counter.count(delta, lo, hi) <= len(pts)


def test_dimension_of_points_3d_handles_a_degenerate_set():
    r = fractal.dimension_of_points_3d(np.zeros((1, 3)))
    assert np.isnan(r['dimension']) and r['n_points'] == 1


@needs_cuda
def test_point_counts_3d_cpu_equals_cuda():
    pts = sierpinski_tetrahedron(7)
    cpu = points3d.PointCounter3D(pts, 'cpu')
    gpu = points3d.PointCounter3D(pts, 'cuda')
    hi = np.array([1.05, 0.9, 0.9])
    for d in (0.1, 0.03, 0.01, 0.004):
        for o in (0.0, 0.37):
            lo = np.array([-0.01 + o * d, -0.01 + o * d, -0.01 + o * d])
            assert cpu.count(d, lo, hi) == gpu.count(d, lo, hi)
