"""The field-based 3D counter, and what it recovers from surfaces of known dimension.

`fractal.dimension_3d_field` marching-cubes a volume-fraction field at the mesh
scale and counts cubes on the triangles, which is the three-dimensional sibling
of `dimension_2d` (marching squares, then boxes on the segments).  It exists so
that the Marstrand codimension test can be made without crossing the two
representations an interface has here, the solver's extracted surface and the
field it came from; `interface`'s module docstring has the full account.

Measured values below, D = 2 exactly for both shapes:

    plane   128^3                     2.0186
    plane   192^3                     2.0301
    sphere  192^3, R/box 6..36        2.0216
    sphere  256^3, R/box 6..48        2.0186

So the counter reads high by about 2 per cent on a smooth surface, which is the
complement of the 2D counter's 0.989 for a straight line.

THE TRAP, and why the sphere tests pin the ladder explicitly: with the default
largest cube of L/8 and a sphere of radius L/8, the top of the ladder is the
whole object and the count returns 2.31, not 2.  A sphere is D = 2 only where
the boxes are small against its radius.  That is a finite-size effect and not a
defect of the estimator, and it is the same hazard as asking a finite-level
Koch curve for its dimension below its smallest feature (see test_shapes).
"""
import numpy as np
import pytest

from rtfractal import fractal, interface
from rtfractal.io.case import Grid3D

L = 0.4


def cube_grid(n):
    d = L / n
    return Grid3D(n, n, n, 0.0, L, 0.0, n * d, 0.0, n * d)


def plane_field(g):
    """alpha crossing 0.5 on the horizontal mid-plane, over a four-cell band."""
    y = (np.arange(g.ny) + 0.5) * g.dy
    a = np.empty((g.nz, g.ny, g.nx))
    a[:] = np.clip(0.5 + (0.5 * g.ny * g.dy - y)[None, :, None] / (4 * g.dy), 0.0, 1.0)
    return a


def sphere_field(g, R):
    c = (np.arange(g.nx) + 0.5) * g.dx
    Z, Y, X = np.meshgrid(c, c, c, indexing='ij')
    rad = np.sqrt((X - 0.5 * L) ** 2 + (Y - 0.5 * L) ** 2 + (Z - 0.5 * L) ** 2)
    return np.clip(0.5 + (R - rad) / (4 * g.dx), 0.0, 1.0)


@pytest.mark.parametrize('n', [128, 192])
def test_plane_is_two(n):
    g = cube_grid(n)
    r = fractal.dimension_3d_field(plane_field(g), g, initial=0.1, min_delta=2 * g.dx)
    assert r['r_squared'] > 0.999
    assert abs(r['dimension'] - 2.0) < 0.05


@pytest.mark.parametrize('n,top', [(192, 0.025), (256, 0.025)])
def test_sphere_is_two_when_boxes_are_small_against_it(n, top):
    g = cube_grid(n)
    r = fractal.dimension_3d_field(sphere_field(g, 0.15), g, initial=top, min_delta=2 * g.dx)
    assert r['r_squared'] > 0.999
    assert abs(r['dimension'] - 2.0) < 0.05


def test_sphere_at_its_own_scale_is_not_two():
    """The finite-size trap, asserted so it cannot be rediscovered as a bug."""
    g = cube_grid(192)
    r = fractal.dimension_3d_field(sphere_field(g, 0.15), g, initial=0.15, min_delta=2 * g.dx)
    assert r['dimension'] > 2.03


def test_surface_sits_in_the_grid_frame():
    """Vertices must land on cell centres, not on the index origin."""
    g = cube_grid(64)
    m = interface.isosurface(plane_field(g), g)
    assert np.allclose(m.vertices[:, 1], 0.5 * g.ny * g.dy)
    assert m.vertices[:, 0].min() == pytest.approx(0.5 * g.dx)
    assert m.vertices[:, 0].max() == pytest.approx((g.nx - 0.5) * g.dx)


def test_no_crossing_raises():
    g = cube_grid(16)
    with pytest.raises(ValueError, match='does not cross'):
        interface.isosurface(np.zeros((16, 16, 16)), g)
    with pytest.raises(ValueError, match='must be'):
        interface.isosurface(np.zeros((16, 16)), g)


def test_offset_grid_shifts_the_surface_with_it():
    """x_min, y_min, z_min are honoured, so a field surface and a VTK surface
    read from the same case land in one frame."""
    n = 64
    d = L / n
    g0 = Grid3D(n, n, n, 0.0, L, 0.0, n * d, 0.0, n * d)
    g1 = Grid3D(n, n, n, 1.0, 1.0 + L, 2.0, 2.0 + n * d, 3.0, 3.0 + n * d)
    a = plane_field(g0)
    m0 = interface.isosurface(a, g0)
    m1 = interface.isosurface(a, g1)
    assert np.allclose(m1.vertices - m0.vertices, [1.0, 2.0, 3.0])
