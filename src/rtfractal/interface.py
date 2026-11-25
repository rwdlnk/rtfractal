"""Interface extraction from volume-fraction fields, in two and three dimensions.

Two-dimensional fields follow the legacy layout: f, x, y are (nx, ny) with
x[:, 0] the cell-centre x coordinates and y[0, :] the cell-centre y
coordinates.  `contours` marching-squares them at the mesh scale, and
`fractal.dimension_2d` counts boxes on the segments that come out.

`isosurface` is the three-dimensional sibling of `contours`: marching cubes on
an (nz, ny, nx) field at the mesh scale, returning the triangulation in
physical coordinates.  It exists because THE SAME INTERFACE HAS TWO
REPRESENTATIONS HERE and they do not count the same.

A three-dimensional interface reaches this package either as a volume-fraction
field or as a surface the flow solver has already extracted and written out
(OpenFOAM's `extractInterface` / `surfaces` function object; see
`io.mesh.load_mesh`).  Counting the extracted surface and counting a surface
marched out of the field at the mesh scale give different dimensions.
Measured on the Dalziel configuration over six cases, plane sections of the
field read higher than plane sections of the extracted surface by 0.061 on
average, between 0.033 and 0.079, with the same sign in every case and
shrinking under refinement.  At 320x400x160 one plane yields about 15,100
contour segments from the field against 12,800 from the extracted surface,
and in three dimensions the gap is larger still, 0.150 at that grid.

That difference is larger than most effects worth measuring, so the first rule
is: NEVER CROSS THE TWO REPRESENTATIONS IN ONE COMPARISON.  The Marstrand
codimension test, D_3D - D_slice, is a set against its own sections, so both
numbers must come from the same representation.  Pairing a surface-counted
D_3D with field sections produced a codimension deficit of 0.06, read for a
while as a property of the interface, that was the representation difference
and nothing else.

The second rule is narrower and was settled by measurement once this function
existed, so it is not a matter of taste: FOR THE CODIMENSION TEST, USE THE
EXTRACTED SURFACE ON BOTH SIDES.  Run end to end on the field over the same
six cases and the same 216 snapshots, the test fails.  Codimensions, with the
surface route alongside:

                        field                     surface
    MULES 160x200x80    1.110 +/- 0.038           0.986
    MULES 320x400x160   1.084 +/- 0.005           0.998
    MULES 400x500x200   1.074 +/- 0.007           0.980
    MPLIC 160x200x80    1.155 +/- 0.031           1.024
    MPLIC 320x400x160   1.095 +/- 0.004           1.004
    MPLIC 400x500x200   1.085 +/- 0.007           0.994
    ----------------------------------------------------
    mean                1.100 +/- 0.029           0.998 +/- 0.016

Seven to sixteen per cent above unity, falling monotonically with refinement
in both codes and still seven to nine per cent high at the finest 3D grid that
exists.  The per-snapshot scatter at the fine grids is 0.004 to 0.007, so the
overshoot is a systematic and not a sampling artefact.  The reason is that
marching cubes on a field whose transition band is one to two cells wide
carries cell-scale staircase structure, and a surface has two tangential
directions to stair-step in where a plane section has one, so the artefact
inflates the 3D count by more than it inflates the sections -- by 0.124 to
0.210 against 0.033 to 0.079, a factor of two and a half to four.  The
codimension falls toward unity because that gap narrows with refinement (0.210,
0.148, 0.124 for MPLIC), not because either count is settling: the field D_3D
runs 2.664, 2.727, 2.751 for MULES and 2.785, 2.769, 2.786 for MPLIC, against
2.464, 2.576, 2.605 and 2.575, 2.621, 2.662 on the surface.

(A probe over two or three snapshots per case suggested the field D_3D FELL
with refinement.  Over the full window it does not.  Do not read a grid trend
off a handful of snapshots when the full window is affordable.)

What the field route is still for: anything compared against a 2D run, since
the 2D dimensions of these papers are counts of the F = 0.5 contour of a field
and a field section is the commensurable quantity (over the six cases a field
section estimates the matched 2D dimension with a mean error of -0.0005).  It
is also the only route available when no surface was written out.
"""
import numpy as np
from skimage import measure

from .io import mesh as _mesh


def contours(f, x, y, level=0.5):
    """Marching-squares iso-lines of f at `level`, in physical coordinates:
    a list of (m, 2) arrays of (x, y) vertices."""
    out = []
    xi, yi = np.arange(f.shape[0]), np.arange(f.shape[1])
    for c in measure.find_contours(f.T, level):
        out.append(np.column_stack([np.interp(c[:, 1], xi, x[:, 0]),
                                    np.interp(c[:, 0], yi, y[0, :])]))
    return out


def segments(cs):
    """Consecutive-vertex segments of every contour -> (n, 4) float64 (x1, y1, x2, y2)."""
    parts = [np.hstack([c[:-1], c[1:]]) for c in cs if len(c) > 1]
    if not parts:
        return np.empty((0, 4))
    return np.ascontiguousarray(np.vstack(parts), dtype=np.float64)


def isosurface(alpha, grid, level=0.5):
    """Marching-cubes iso-surface of a 3D volume-fraction field -> TriangleMesh.

    `alpha` is the (nz, ny, nx) field as `io.case.Case.read_alpha_3d` returns
    it, `grid` its `Grid3D`.  Vertices come back in physical (x, y, z) with the
    grid's own origin and the half-cell shift that puts index i at the centre
    of cell i, so the result sits in the same frame as a surface read from a
    VTK file and the two are directly comparable.

    Triangle winding is not preserved (the axis relabelling from the
    (z, y, x) volume order flips it), which nothing here depends on: the
    cube-occupancy test is orientation-free and `surface_area` takes a
    magnitude.  Raises ValueError when the field does not cross `level`,
    rather than returning an empty mesh whose bounding box is undefined.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim != 3:
        raise ValueError('alpha must be (nz, ny, nx), got shape %r' % (alpha.shape,))
    lo, hi = float(alpha.min()), float(alpha.max())
    if not (lo < level < hi):
        raise ValueError('field does not cross level=%g (range %g to %g)' % (level, lo, hi))
    verts, faces, _, _ = measure.marching_cubes(alpha, level=level,
                                                spacing=(grid.dz, grid.dy, grid.dx))
    # verts columns are (z, y, x) in cell-index units scaled by spacing.
    v = np.column_stack([verts[:, 2] + grid.x_min + 0.5 * grid.dx,
                         verts[:, 1] + grid.y_min + 0.5 * grid.dy,
                         verts[:, 0] + grid.z_min + 0.5 * grid.dz])
    return _mesh.TriangleMesh.from_arrays(v, faces)
