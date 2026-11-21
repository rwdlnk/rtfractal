"""The 2D point-cloud box counter.

A point set is not a curve, and the difference is not cosmetic: a finite
point set has box dimension 0 below the spacing between its points, because
every box then holds at most one point and the count saturates at n.  So the
ladder has a hard lower bound a curve's does not, and the counter refuses to
cross it.

Measured here, so the numbers in the module docstring and the CLI notes are
not assertions of faith:

  point samples of the shapes, min_box = 2 x spacing, max_box = L/8
    koch        L8  65537 pts   exact 1.2619   1.2717   +0.010
    sierpinski  L8   9843 pts   exact 1.5850   1.5853   +0.000
    minkowski   L6 218793 pts   exact 1.5000   1.4676   -0.032
    hilbert     L9 262144 pts   exact 2.0000   1.9337   -0.066
    dragon      L16 72680 pts   exact 2.0000   1.7745   -0.226

  a uniform cloud in the unit square, exact 2, max_box = extent/8
    n = 20000    1.8188 at 2x spacing, 1.8920 at 4x
    n = 200000   1.9239                1.9320
    n = 1000000  1.9430                1.9494

The plane-filling pair read low for the same finite-level reason as in
test_shapes.py.  The uniform cloud approaches 2 as the sampling becomes
dense relative to the ladder, which is a property of finite sampling and not
of the counter.
"""
import numpy as np
import pytest

from rtfractal import backend, fractal, shapes
from rtfractal.boxcount import points2d, scaling

needs_cuda = pytest.mark.skipif(not backend.cuda_available(), reason='no CUDA device')

LADDER = dict(factor=1.5, backend='cpu', offsets='covering')


def vertices(name, level):
    """The distinct vertices of a shape, as an unordered cloud."""
    s = shapes.SHAPES[name](level)
    return np.unique(np.vstack([s[:, :2], s[:, 2:]]), axis=0)


# --------------------------------------------------------------- the spacing
def test_nn_spacing_of_a_regular_lattice_is_the_lattice_step():
    g = np.linspace(0, 1, 11)
    pts = np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2)
    assert points2d.nn_spacing(pts) == pytest.approx(0.1)


def test_nn_spacing_ignores_exact_duplicates():
    pts = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert points2d.nn_spacing(pts) == pytest.approx(1.0)


def test_nn_spacing_needs_two_distinct_points():
    with pytest.raises(ValueError, match='at least 2 points'):
        points2d.nn_spacing(np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError, match='every point is a duplicate'):
        points2d.nn_spacing(np.zeros((5, 2)))


def test_spacing_stats_are_ordered():
    pts = np.random.default_rng(1).random((2000, 2))
    assert (points2d.nn_spacing(pts, 'min') <= points2d.nn_spacing(pts, 'median')
            <= points2d.nn_spacing(pts, 'mean') * 1.5)


# ------------------------------------------------------------- the hard bound
def test_a_ladder_below_the_spacing_is_refused():
    pts = vertices('koch', 6)
    sp = points2d.nn_spacing(pts)
    with pytest.raises(ValueError, match='below the nearest-neighbour spacing'):
        points2d.box_counting_2d_points(pts, sp / 4, 0.05, **LADDER)


def test_the_refusal_can_be_overridden_deliberately():
    pts = vertices('koch', 6)
    sp = points2d.nn_spacing(pts)
    s, c, _ = points2d.box_counting_2d_points(pts, sp / 4, 0.05, allow_below_spacing=True, **LADDER)
    assert len(s) > 2


def test_below_the_spacing_the_count_saturates_and_there_is_no_dimension():
    """Why the bound exists, demonstrated rather than argued.

    Taken two decades below the spacing, every box holds exactly one point:
    the count is exactly n at all ten sizes, the log-log slope is exactly 0,
    and dimension_2d returns NaN because a flat ladder has no scaling range to
    fit.  A finite point set has box dimension 0, so NaN is the right answer
    and reporting one would be the bug."""
    pts = vertices('sierpinski', 7)
    sp = points2d.nn_spacing(pts)
    s, c, _ = points2d.box_counting_2d_points(pts, sp / 200, sp / 4,
                                              allow_below_spacing=True, **LADDER)
    assert (c == len(pts)).all()                                  # one box per point, every size
    assert np.polyfit(np.log10(s), np.log10(c), 1)[0] == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(scaling.dimension_2d(s, c)['dimension'])


def test_a_known_spacing_can_be_supplied_instead_of_measured():
    pts = vertices('koch', 6)
    s, c, _ = points2d.box_counting_2d_points(pts, 0.01, 0.05, spacing=0.005, **LADDER)
    assert len(s) >= 2
    with pytest.raises(ValueError, match='0.02'):
        points2d.box_counting_2d_points(pts, 0.01, 0.05, spacing=0.02, **LADDER)


# ------------------------------------------------------- recovered dimensions
@pytest.mark.parametrize('name,level,tol', [('koch', 8, 0.02),
                                            ('sierpinski', 8, 0.01),
                                            ('minkowski', 6, 0.05)])
def test_point_samples_of_the_shapes_recover_their_dimensions(name, level, tol):
    pts = vertices(name, level)
    r = fractal.dimension_of_points(pts, max_box=0.4 / 8, backend='cpu')
    assert r['dimension'] == pytest.approx(shapes.DIMENSION[name], abs=tol)
    assert r['r_squared'] > 0.99
    assert r['n_points'] == len(pts)


@pytest.mark.parametrize('name,level,floor', [('hilbert', 9, 1.85), ('dragon', 16, 1.70)])
def test_plane_filling_point_samples_approach_two_from_below(name, level, floor):
    r = fractal.dimension_of_points(vertices(name, level), max_box=0.4 / 8, backend='cpu')
    assert floor < r['dimension'] < 2.02


def test_a_uniform_cloud_approaches_two_with_density():
    """Exact 2.  A finite sample reads low, and converges as it densifies."""
    rng = np.random.default_rng(0)
    ds = []
    for n in (20000, 200000):
        pts = rng.random((n, 2))
        sp = points2d.nn_spacing(pts)
        s, c, _ = points2d.box_counting_2d_points(pts, 4 * sp, 1 / 16, **LADDER)
        ds.append(scaling.dimension_2d(s, c)['dimension'])
    assert 1.85 < ds[0] < 2.0
    assert ds[1] > ds[0]
    assert ds[1] < 2.0


def test_the_point_count_agrees_with_the_segment_count_on_one_curve():
    """A dense point sample of the Koch curve and the curve itself are the same
    set measured two ways, so the two counters must agree to within the
    estimators' own bias."""
    segs = shapes.koch(8)
    pts = np.unique(np.vstack([segs[:, :2], segs[:, 2:]]), axis=0)
    sp = points2d.nn_spacing(pts)
    d_seg = fractal.dimension_of_segments(segs, min_box=4 * sp, max_box=0.4 / 8, backend='cpu')
    d_pts = fractal.dimension_of_points(pts, min_box=4 * sp, max_box=0.4 / 8, backend='cpu')
    assert d_pts['dimension'] == pytest.approx(d_seg['dimension'], abs=0.03)


# ------------------------------------------------------------------- plumbing
def test_shape_validation():
    with pytest.raises(ValueError, match=r'\(n, 2\)'):
        points2d.PointCounter(np.zeros((5, 3)))
    with pytest.raises(ValueError, match=r'\(n, 2\)'):
        points2d.box_counting_2d_points(np.zeros((5, 4)), 0.1, 0.5)
    with pytest.raises(ValueError, match='at least 2 points'):
        points2d.box_counting_2d_points(np.zeros((1, 2)), 0.1, 0.5)


def test_counts_never_exceed_the_point_count_or_the_grid():
    pts = vertices('koch', 6)
    counter = points2d.PointCounter(pts, 'cpu')
    for delta in (0.05, 0.01, 0.002):
        n = counter.count(delta, -0.01, 0.24, 0.41, 0.37)
        nx = int(np.ceil(0.42 / delta))
        ny = int(np.ceil(0.13 / delta))
        assert 0 < n <= min(len(pts), nx * ny)


def test_dimension_of_points_handles_a_degenerate_set():
    r = fractal.dimension_of_points(np.zeros((1, 2)))
    assert np.isnan(r['dimension']) and r['n_points'] == 1


@needs_cuda
def test_point_counts_cpu_equals_cuda():
    pts = vertices('koch', 7)
    cpu, gpu = points2d.PointCounter(pts, 'cpu'), points2d.PointCounter(pts, 'cuda')
    for d in (0.05, 0.01, 0.003, 0.0011, 0.0004):
        for o in (0.0, 0.37):
            assert cpu.count(d, -0.01 + o * d, 0.24 + o * d, 0.41, 0.37) == \
                   gpu.count(d, -0.01 + o * d, 0.24 + o * d, 0.41, 0.37)
