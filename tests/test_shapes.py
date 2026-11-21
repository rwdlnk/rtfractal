"""The theoretical fractals, and what the estimator recovers from them.

These are the validations the 2025 box-counting article rests on.  Before the
generators were ported (2026-11-21) only Koch was available here, as a helper
duplicated in two test files, so four of the five could not be re-run against
the code that now produces the published numbers.

Measured values at the levels used below, default ladder (min_box = 2L/640,
max_box = L/8, factor 1.5, covering offsets):

    koch        L7  1.2668   exact 1.2619   +0.005
    sierpinski  L7  1.5938   exact 1.5850   +0.009
    minkowski   L5  1.4408   exact 1.5000   -0.059
    hilbert     L7  1.8471   exact 2.0000   -0.153
    dragon      L13 1.7260   exact 2.0000   -0.274

The last three read low for one reason, not three: a finite level has a
smallest construction feature, and below it the shape is a collection of
straight lines, which the small-box end of the ladder sees as D = 1.  All
five converge monotonically from below as the level rises, which is what
test_dimension_converges_with_level asserts, and is a stronger statement
about the estimator than any single number.
"""
import numpy as np
import pytest

from rtfractal import fractal, shapes

L = 0.4
LADDER = dict(min_box=2 * L / 640, max_box=L / 8, factor=1.5, backend='cpu',
              offsets='covering')


def measure(segs):
    return fractal.dimension_of_segments(segs, **LADDER)


# ------------------------------------------------------------ the port itself
def test_koch_is_identical_to_the_helper_it_replaced():
    """The former helper, inline, byte-for-byte as it stood in two test files."""
    def former(level, L=0.4, y0=0.25):
        pts = np.array([[0.0, y0], [L, y0]])
        rot = np.array([[0.5, -np.sqrt(3) / 2], [np.sqrt(3) / 2, 0.5]])
        for _ in range(level):
            new = [pts[0]]
            for a, b in zip(pts[:-1], pts[1:]):
                d = (b - a) / 3
                new += [a + d, a + d + rot @ d, a + 2 * d, b]
            pts = np.array(new)
        return np.hstack([pts[:-1], pts[1:]])

    for level in (0, 1, 4, 6):
        assert np.array_equal(shapes.koch(level), former(level))


@pytest.mark.parametrize('name', sorted(shapes.SHAPES))
def test_segments_follow_the_package_convention(name):
    segs = shapes.SHAPES[name](4)
    assert segs.ndim == 2 and segs.shape[1] == 4
    assert segs.dtype == np.float64
    assert len(segs) > 0
    assert np.isfinite(segs).all()


@pytest.mark.parametrize('name', ['koch', 'minkowski', 'hilbert', 'dragon'])
def test_polyline_shapes_are_connected(name):
    """Each segment starts where the previous one ended.  Sierpinski is
    excluded: it is a set of closed triangles, not one open polyline."""
    segs = shapes.SHAPES[name](4)
    assert np.allclose(segs[1:, :2], segs[:-1, 2:])


@pytest.mark.parametrize('name', sorted(shapes.SHAPES))
def test_shapes_are_placed_in_the_requested_box(name):
    segs = shapes.SHAPES[name](4, L=0.4)
    x = np.r_[segs[:, 0], segs[:, 2]]
    y = np.r_[segs[:, 1], segs[:, 3]]
    assert max(np.ptp(x), np.ptp(y)) == pytest.approx(0.4, rel=1e-9)


# ------------------------------------------------------- recovered dimensions
def test_koch_dimension():
    assert measure(shapes.koch(7))['dimension'] == pytest.approx(
        shapes.DIMENSION['koch'], abs=0.01)


def test_sierpinski_dimension():
    assert measure(shapes.sierpinski(7))['dimension'] == pytest.approx(
        shapes.DIMENSION['sierpinski'], abs=0.02)


def test_minkowski_dimension():
    """Tolerance 0.08: at level 5 the measured value is 1.441 against 1.5, and
    the deficit is the finite construction scale, not estimator error.  It
    halves from level 3 to 5 and keeps shrinking (-0.214, -0.115, -0.059)."""
    assert measure(shapes.minkowski(5))['dimension'] == pytest.approx(
        shapes.DIMENSION['minkowski'], abs=0.08)


@pytest.mark.parametrize('name,level,floor', [('hilbert', 7, 1.75),
                                              ('dragon', 13, 1.65)])
def test_plane_filling_shapes_approach_two_from_below(name, level, floor):
    d = measure(shapes.SHAPES[name](level))['dimension']
    assert floor < d < 2.02


@pytest.mark.parametrize('name,levels', [('koch', (5, 6, 7)),
                                         ('sierpinski', (4, 5, 6, 7)),
                                         ('minkowski', (3, 4, 5)),
                                         ('hilbert', (4, 5, 6, 7)),
                                         ('dragon', (9, 11, 13))])
def test_dimension_converges_with_level(name, levels):
    """D rises monotonically toward the exact value as the construction
    resolves, and never overshoots it by more than the estimator's own bias."""
    ds = [measure(shapes.SHAPES[name](lv))['dimension'] for lv in levels]
    assert all(b > a - 1e-9 for a, b in zip(ds, ds[1:])), ds
    assert ds[-1] <= shapes.DIMENSION[name] + 0.02, ds


@pytest.mark.parametrize('name', sorted(shapes.SHAPES))
def test_fits_pass_the_admission_gate(name):
    """R^2 >= 0.97 is the project's own admission gate, the one that rejects
    the 3D count on a near-planar surface.  All five shapes clear it at level
    5.  Hilbert is the closest, at 0.980: at that level it is still far from
    plane-filling and its log-log plot is visibly curved."""
    assert measure(shapes.SHAPES[name](5))['r_squared'] >= 0.97
