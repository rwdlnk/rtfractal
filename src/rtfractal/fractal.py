"""Box-counting fractal dimensions of 2D contours and 3D surfaces."""
import numpy as np

from . import interface
from .boxcount import scaling
from .boxcount.offsets import offset_fractions
from .boxcount.segments2d import box_counting_2d
from .boxcount.triangles3d import box_counting_3d

NAN_RESULT = dict(dimension=np.nan, error=np.nan, r_squared=np.nan)


def dimension_of_segments(segs, min_box, max_box=None, factor=1.5, backend=None, offsets='covering'):
    if len(segs) == 0:
        return dict(NAN_RESULT)
    if max_box is None:
        ext = max(np.ptp(np.r_[segs[:, 0], segs[:, 2]]), np.ptp(np.r_[segs[:, 1], segs[:, 3]]))
        max_box = ext / 2
    sizes, counts, bbox = box_counting_2d(segs, min_box, max_box, factor, backend, offsets)
    res = scaling.dimension_2d(sizes, counts)
    res['bounding_box'] = bbox
    return res


def dimension_2d(field, level=0.5, min_box=0.001, max_box=None, factor=1.5, backend=None, offsets='covering'):
    """Dimension of the `level` contour of field['f'] (legacy RTAnalyzer.compute_fractal_dimension)."""
    segs = interface.segments(interface.contours(field['f'], field['x'], field['y'], level))
    res = dimension_of_segments(segs, min_box, max_box, factor, backend, offsets)
    res['n_segments'] = len(segs)
    return res


def dimension_3d(mesh, initial, factor=1.5, num_steps=20, min_delta=None, backend=None, log=None, offsets='covering'):
    deltas, counts = box_counting_3d(mesh, initial, factor, num_steps, min_delta, backend=backend, log=log,
                                     offsets=offsets)
    return scaling.dimension_3d(deltas, counts)


def projected_dimension(mesh, dx, dy):
    """Dimension of the surface's x-y projection (depth axis z collapsed).

    Every unique edge is sampled at its end points and mid-point (edges are
    shorter than the smallest box), points are binned with the usual origin
    offsets (minimum count kept), and the legacy scaling window is applied.
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
