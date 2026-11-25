"""Box-counting fractal dimensions of 2D contours and 3D surfaces."""
import numpy as np

from . import interface
from .boxcount import scaling
from .boxcount.offsets import offset_fractions
from .boxcount import points2d, points3d
from .boxcount.segments2d import box_counting_2d
from .boxcount.triangles3d import box_counting_3d

NAN_RESULT = dict(dimension=np.nan, error=np.nan, r_squared=np.nan)


def _ladder_diagnostics(sizes, counts, n_points):
    """What the ladder looked like before anything was fitted to it.

    `saturation` is the fraction of the cloud's own points that the count has
    reached, N/n.  A point set cannot be counted below its own spacing, and it
    stops being countable well above it: at N/n = 0.72 most points are already
    alone in their box and the local slope is on its way to 0, which is the
    dimension of a finite set and not of anything it samples.
    """
    ms, sl = scaling.local_slope(sizes, counts)
    return dict(sizes=np.asarray(sizes, dtype=np.float64),
                counts=np.asarray(counts),
                saturation=np.asarray(counts, dtype=np.float64) / max(n_points, 1),
                local_slope_sizes=ms, local_slope=sl)


def dimension_of_segments(segs, min_box, max_box=None, factor=1.5, backend=None, offsets='covering',
                          trim='r2', num_sizes=None, sizes=None, min_span=0.0):
    if len(segs) == 0:
        return dict(NAN_RESULT)
    if max_box is None:
        ext = max(np.ptp(np.r_[segs[:, 0], segs[:, 2]]), np.ptp(np.r_[segs[:, 1], segs[:, 3]]))
        max_box = ext / 8
    sizes, counts, bbox = box_counting_2d(segs, min_box, max_box, factor, backend, offsets,
                                          num_sizes, sizes)
    res = scaling.dimension_2d(sizes, counts, trim, min_span)
    res['bounding_box'] = bbox
    return res


def dimension_of_points(pts, min_box=None, max_box=None, factor=1.5, backend=None, offsets='covering',
                        trim='r2', spacing=None, allow_below_spacing=False,
                        num_sizes=None, sizes=None, min_span=0.0):
    """Box-counting dimension of a 2D POINT SET.  -> the dimension_2d dict plus
    `spacing` and `n_points`.

    A finite point set has box dimension 0 below the spacing between its
    points, so the ladder has a hard lower bound that a curve's does not;
    `box_counting_2d_points` refuses to cross it.  min_box defaults to twice
    the measured nearest-neighbour spacing, max_box to extent/8 as elsewhere.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return dict(NAN_RESULT, spacing=np.nan, n_points=len(pts))
    if spacing is None:
        spacing = points2d.nn_spacing(pts)
    if min_box is None:
        min_box = 2.0 * spacing
    if max_box is None:
        ext = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
        max_box = ext / 8
    ladder_sizes, ladder_counts, bbox = points2d.box_counting_2d_points(
        pts, min_box, max_box, factor, backend, offsets, spacing, allow_below_spacing,
        num_sizes, sizes)
    res = scaling.dimension_2d(ladder_sizes, ladder_counts, trim, min_span)
    res['bounding_box'] = bbox
    res['spacing'] = spacing
    res['n_points'] = len(pts)
    res['ladder'] = _ladder_diagnostics(ladder_sizes, ladder_counts, len(pts))
    return res


def dimension_of_points_3d(pts, initial=None, factor=1.5, num_steps=20, min_delta=None,
                           max_delta=None, backend=None, log=None, offsets='covering',
                           trim='r2', spacing=None, allow_below_spacing=False,
                           num_sizes=None, sizes=None, min_span=0.0):
    """Box-counting dimension of a 3D POINT SET.  -> the dimension_3d dict plus
    `spacing` and `n_points`.

    Same hard lower bound as in 2D: below the nearest-neighbour spacing the
    count saturates at n and the dimension collapses toward 0.  For a cloud
    sampling a surface the nearest-neighbour distance IS the surface sampling
    density, which is the scale that matters; nothing is derived from the
    bounding-box volume.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return dict(NAN_RESULT, spacing=np.nan, n_points=len(pts))
    deltas, counts, bbox = points3d.box_counting_3d_points(
        pts, initial, factor, num_steps, min_delta, max_delta, backend, log, offsets,
        spacing, allow_below_spacing, num_sizes, sizes)
    res = scaling.dimension_3d(deltas, counts, trim, min_span)
    res['bounding_box'] = bbox
    res['spacing'] = spacing if spacing is not None else points3d.nn_spacing(pts)
    res['n_points'] = len(pts)
    res['ladder'] = _ladder_diagnostics(deltas, counts, len(pts))
    return res


def dimension_2d(field, level=0.5, min_box=0.001, max_box=None, factor=1.5, backend=None, offsets='covering',
                 trim='r2', min_span=0.0):
    """Dimension of the `level` contour of field['f'] (legacy RTAnalyzer.compute_fractal_dimension)."""
    segs = interface.segments(interface.contours(field['f'], field['x'], field['y'], level))
    res = dimension_of_segments(segs, min_box, max_box, factor, backend, offsets, trim,
                                min_span=min_span)
    res['n_segments'] = len(segs)
    return res


def dimension_3d(mesh, initial, factor=1.5, num_steps=20, min_delta=None, backend=None, log=None, offsets='covering',
                 trim='r2', min_span=0.0):
    deltas, counts = box_counting_3d(mesh, initial, factor, num_steps, min_delta, backend=backend, log=log,
                                     offsets=offsets)
    return scaling.dimension_3d(deltas, counts, trim, min_span)


def dimension_3d_field(alpha, grid, level=0.5, initial=None, factor=1.5, num_steps=20, min_delta=None,
                       backend=None, log=None, offsets='covering', trim='r2', min_span=0.0):
    """Dimension of the `level` iso-surface of a 3D field, marched out of the field.

    The three-dimensional sibling of `dimension_2d`.  That function
    marching-squares the field at the mesh scale and counts boxes on the
    segments; this one marching-cubes it at the mesh scale and counts cubes on
    the triangles, through the same ladder, the same covering offsets and the
    same end-point trim, so the only difference from `dimension_3d` is where
    the surface came from.

    Use `dimension_3d` on `io.mesh.load_mesh` output, not this, for the
    Marstrand codimension test.  Run end to end on the field that test
    overshoots unity by seven to sixteen per cent, falling with refinement but
    still seven to nine per cent high at the finest grid, because marching
    cubes on a one-to-two-cell transition band inflates a 3D count by two and a
    half to four times what it inflates a plane section.  `interface` carries
    the numbers.  What this function is for is a 3D dimension commensurable with
    something else measured on the field, and the case where no surface was
    written out at all.

    Defaults match what the pipeline uses for the extracted surface: a largest
    cube of one eighth of the x extent and a smallest of twice the coarsest
    mesh spacing.  Returns the `dimension_3d` dict plus `n_triangles` and
    `surface_area`, the latter being the area of the marched surface and not
    of the solver's.
    """
    m = interface.isosurface(alpha, grid, level)
    if initial is None:
        initial = grid.nx * grid.dx / 8.0
    if min_delta is None:
        min_delta = 2.0 * max(grid.dx, grid.dy, grid.dz)
    res = dimension_3d(m, initial=initial, factor=factor, num_steps=num_steps, min_delta=min_delta,
                       backend=backend, log=log, offsets=offsets, trim=trim, min_span=min_span)
    res['n_triangles'] = m.n_triangles
    res['surface_area'] = m.surface_area
    return res


def projected_dimension(mesh, dx, dy):
    """Dimension of the surface's x-y projection (depth axis z collapsed).

    Every unique edge is sampled at its end points and mid-point (edges are
    shorter than the smallest box), points are binned with origin offsets
    (minimum count kept), and the legacy scaling window is applied.  Binning
    floor((p - min + f*box)/box) puts the origin at min - f*box, so this was
    already covering-style in the legacy code (despite its comment) and
    legacy D_proj values are unbiased.  Still uses max box L/2.
    -> (D, R^2)."""
    tris = mesh.triangles
    e = np.unique(np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0)
    if len(e) < 10:
        return np.nan, np.nan
    v = mesh.vertices
    x1, y1, x2, y2 = v[e[:, 0], 0], v[e[:, 0], 1], v[e[:, 1], 0], v[e[:, 1], 1]
    px = np.concatenate([x1, 0.5 * (x1 + x2), x2])
    py = np.concatenate([y1, 0.5 * (y1 + y2), y2])
    x_min, y_min = float(px.min()), float(py.min())
    L = max(float(px.max()) - x_min, float(py.max()) - y_min)
    min_box, sizes, bs = max(dx, dy) * 2, [], L / 2
    while bs >= min_box:
        sizes.append(bs)
        bs /= 1.5
    if len(sizes) < 4:
        return np.nan, np.nan
    counts = []
    for bs in sizes:
        best = None
        for fx in offset_fractions(bs):
            for fy in offset_fractions(bs):
                bx = np.floor((px - x_min + fx * bs) / bs).astype(np.int64)
                by = np.floor((py - y_min + fy * bs) / bs).astype(np.int64)
                n = int(np.unique(bx * np.int64(1 << 32) + by).size)
                best = n if best is None else min(best, n)
        counts.append(best)
    sl, _, r2, _, _ = scaling._best_window(np.log10(sizes), np.log10(np.array(counts, dtype=float)))
    return -sl, r2
