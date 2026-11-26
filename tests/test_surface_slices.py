"""Plane sections of a triangulated surface, the counterpart of the field route.

`slices.surface_slice_dimension` cuts the extracted iso-surface that
`fractal.dimension_3d` counts, so a codimension taken against it crosses no
representations.  See `interface`'s module docstring for why that matters:
field sections read 0.061 higher, and the codimension test holds on the
surface throughout and fails on the field throughout.

Validated against vtk's own cutter on two real cases (2026-11-25), same planes
and same ladder:

    MPLIC 320x400x160, t = 12, eleven planes   both 1.6146, max |dD| 1.6e-04
    Cao A = 0 400x400x200, t = 14, nine planes  both 1.1930, max |dD| 0

The segment counts differ by about a factor of two because vtk chains
crossings into polylines while this emits one segment per crossed triangle.
Both cover the same curve, so the box count is the same.

THE BUG THESE TESTS GUARD: an iso-surface vertex sits on a cell edge, so a
cell-aligned plane passes exactly through vertices.  A strict `(da * db) < 0`
crossing test drops every triangle that touches the plane at a vertex, which
silently returned NO SECTION AT ALL for the centre plane of the Cao grid while
the eight planes either side were correct.  Zero distances now count as
positive.
"""
import numpy as np
import pytest

from rtfractal import fractal, interface, slices
from rtfractal.io.case import Grid3D

L = 0.4


def cube_grid(n):
    d = L / n
    return Grid3D(n, n, n, 0.0, L, 0.0, n * d, 0.0, n * d)


def sphere_mesh(n, R):
    g = cube_grid(n)
    c = (np.arange(n) + 0.5) * g.dx
    Z, Y, X = np.meshgrid(c, c, c, indexing='ij')
    rad = np.sqrt((X - 0.5 * L) ** 2 + (Y - 0.5 * L) ** 2 + (Z - 0.5 * L) ** 2)
    a = np.clip(0.5 + (R - rad) / (4 * g.dx), 0.0, 1.0)
    return interface.isosurface(a, g), g


def plane_mesh(n):
    g = cube_grid(n)
    y = (np.arange(g.ny) + 0.5) * g.dy
    a = np.empty((g.nz, g.ny, g.nx))
    a[:] = np.clip(0.5 + (0.5 * g.ny * g.dy - y)[None, :, None] / (4 * g.dy), 0.0, 1.0)
    return interface.isosurface(a, g), g


def test_centre_plane_is_not_empty():
    """Regression: a cell-aligned plane passes through iso-surface vertices."""
    m, g = sphere_mesh(96, 0.12)
    r = slices.surface_slice_dimension(m, g, 0.0, 0.0, max_box=0.05, min_box=2 * g.dx)
    assert r['n_segments'] > 0
    assert np.isfinite(r['dimension'])


def test_section_of_a_sphere_is_a_circle():
    """A great circle is D = 1, where the boxes are small against its radius."""
    m, g = sphere_mesh(96, 0.12)
    r = slices.surface_slice_dimension(m, g, 0.0, 0.0, max_box=0.03, min_box=2 * g.dx)
    assert abs(r['dimension'] - 1.0) < 0.05


def test_circle_at_its_own_scale_is_not_one():
    """The finite-size trap again, pinned so it is not rediscovered as a bug:
    with the largest box at R/2.4 the top of the ladder is the whole circle."""
    m, g = sphere_mesh(96, 0.12)
    r = slices.surface_slice_dimension(m, g, 0.0, 0.0, max_box=0.05, min_box=2 * g.dx)
    assert r['dimension'] > 1.03


def test_section_of_a_plane_is_a_line():
    m, g = plane_mesh(96)
    r = slices.surface_slice_dimension(m, g, 0.0, 0.0, max_box=0.05, min_box=2 * g.dx)
    assert r['n_segments'] > 0
    assert abs(r['dimension'] - 1.0) < 0.05


def test_offset_cuts_a_smaller_circle():
    m, g = sphere_mesh(96, 0.12)
    mid = slices.surface_slice_dimension(m, g, 0.0, 0.0, max_box=0.05, min_box=2 * g.dx)
    off = slices.surface_slice_dimension(m, g, 0.0, 0.09, max_box=0.05, min_box=2 * g.dx)
    assert off['n_segments'] < mid['n_segments']
    assert off['n_segments'] > 0


def test_plane_beyond_the_surface_is_empty_not_an_error():
    m, g = sphere_mesh(96, 0.12)
    r = slices.surface_slice_dimension(m, g, 0.0, 0.19, max_box=0.05, min_box=2 * g.dx)
    assert r['n_segments'] == 0
    assert np.isnan(r['dimension'])


@pytest.mark.parametrize('theta', [0.0, 90.0])
def test_grid_aligned_extents(theta):
    m, g = sphere_mesh(64, 0.12)
    segs, extent = slices.surface_plane_section(m, g, theta, 0.0)
    expect = (g.nx - 1) * g.dx if theta == 0.0 else (g.nz - 1) * g.dz
    assert extent == pytest.approx(expect)
    assert len(segs) > 0


def test_the_two_routes_agree_on_a_smooth_surface():
    """Field and surface sections differ on a rough interface, not a smooth one."""
    g = cube_grid(96)
    c = (np.arange(96) + 0.5) * g.dx
    Z, Y, X = np.meshgrid(c, c, c, indexing='ij')
    rad = np.sqrt((X - 0.5 * L) ** 2 + (Y - 0.5 * L) ** 2 + (Z - 0.5 * L) ** 2)
    a = np.clip(0.5 + (0.12 - rad) / (4 * g.dx), 0.0, 1.0)
    m = interface.isosurface(a, g)
    s = slices.surface_slice_dimension(m, g, 0.0, 0.0, max_box=0.05, min_box=2 * g.dx)
    f = slices.slice_dimension(a, g, 0.0, 0.0, max_box=0.05, min_box=2 * g.dx)
    assert abs(s['dimension'] - f['dimension']) < 0.05
