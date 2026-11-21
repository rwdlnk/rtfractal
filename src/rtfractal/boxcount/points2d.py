"""Occupied-box counting for a 2D point set.

Points are an (n, 2) float64 array.  A point occupies exactly one box, so
there is no clipping test as for segments and no separating-axis test as for
triangles: the cell index is floor((p - origin) / delta) and the count is the
number of distinct cells.  The CPU path takes the distinct count of the cell
keys, which costs O(n) memory and never allocates the grid; the CUDA path
marks an occupancy array, as the segment and triangle kernels do.  Both count
the same integer cells, so they agree exactly rather than approximately.

**A finite point set has box dimension zero.**  Below the spacing between
points every box holds at most one point, the count saturates at n, and the
apparent dimension falls to 0.  Above the spacing the count follows the
structure the points sample.  So a point ladder has a hard lower bound that a
segment ladder does not, and `box_counting_2d_points` refuses to go below the
median nearest-neighbour distance unless told to.  That is the difference
between a point measure and a curve measure, and it is not a tuning
parameter.

`box_counting_2d_points()` otherwise matches `segments2d.box_counting_2d`:
1% bounding-box margin, ladder from max_box down by `factor`, minimum count
over the origin offsets.
"""
import numpy as np

from .. import backend as _backend
from .ladder import box_ladder
from .offsets import offset_fractions, origin

MARGIN_FACTOR = 0.01


def nn_spacing(pts, stat='median'):
    """Nearest-neighbour distance of the cloud: the ladder's hard lower bound.

    'median' by default, being insensitive to a few coincident or isolated
    points; 'mean' and 'min' are also accepted."""
    from scipy.spatial import cKDTree

    pts = np.ascontiguousarray(pts, dtype=np.float64)
    if len(pts) < 2:
        raise ValueError('nearest-neighbour spacing needs at least 2 points')
    d, _ = cKDTree(pts).query(pts, k=2)
    nn = d[:, 1]
    nn = nn[nn > 0]                      # drop exact duplicates
    if len(nn) == 0:
        raise ValueError('every point is a duplicate: no spacing to measure')
    return float({'median': np.median, 'mean': np.mean, 'min': np.min}[stat](nn))


def _cell_keys(pts, delta, dmin_x, dmin_y, nx, ny):
    """Distinct-cell keys of the points, as int64.  Points outside the grid are
    clamped, matching the kernels."""
    i = np.floor((pts[:, 0] - dmin_x) / delta).astype(np.int64)
    j = np.floor((pts[:, 1] - dmin_y) / delta).astype(np.int64)
    np.clip(i, 0, nx - 1, out=i)
    np.clip(j, 0, ny - 1, out=j)
    return i * ny + j


_cuda_kernel = None


def _get_cuda_kernel():
    global _cuda_kernel
    if _cuda_kernel is None:
        from numba import cuda

        @cuda.jit
        def kern(pts, delta, inv_delta, dmin_x, dmin_y, nx, ny, occ):
            p = cuda.grid(1)
            if p >= pts.shape[0]:
                return
            i = int(np.float64((pts[p, 0] - dmin_x) * inv_delta) // 1.0)
            j = int(np.float64((pts[p, 1] - dmin_y) * inv_delta) // 1.0)
            if i < 0:
                i = 0
            if j < 0:
                j = 0
            if i >= nx:
                i = nx - 1
            if j >= ny:
                j = ny - 1
            occ[i * ny + j] = 1          # idempotent: races are benign

        _cuda_kernel = kern
    return _cuda_kernel


class PointCounter:
    """Counts occupied boxes for one point set on a chosen backend
    (points stay resident on the GPU across calls)."""

    def __init__(self, pts, backend=None):
        pts = np.ascontiguousarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f'points must be (n, 2), got {pts.shape}')
        self.pts = pts
        self.backend = _backend.resolve(backend)
        if self.backend == 'cuda':
            from numba import cuda
            self._d_pts = cuda.to_device(self.pts)

    def count(self, delta, dmin_x, dmin_y, dmax_x, dmax_y):
        nx = max(1, int(np.ceil((dmax_x - dmin_x) / delta)))
        ny = max(1, int(np.ceil((dmax_y - dmin_y) / delta)))
        if self.backend == 'cpu':
            keys = _cell_keys(self.pts, delta, dmin_x, dmin_y, nx, ny)
            return int(np.unique(keys).size)
        from numba import cuda
        occ = cuda.to_device(np.zeros(nx * ny, dtype=np.uint8))
        n = self.pts.shape[0]
        _get_cuda_kernel()[(n + 255) // 256, 256](
            self._d_pts, delta, 1.0 / delta, dmin_x, dmin_y, nx, ny, occ)
        return int(occ.copy_to_host().sum())


def box_counting_2d_points(pts, min_box, max_box, factor=1.5, backend=None, offsets='covering',
                           spacing=None, allow_below_spacing=False, num_sizes=None, sizes=None):
    """Grid-optimised box counting of a point set -> (box_sizes, box_counts, bbox).

    Raises ValueError if the smallest box is below the nearest-neighbour
    spacing, where the count saturates and the dimension collapses toward 0.
    Pass allow_below_spacing=True to override, or `spacing` to supply a known
    value (the particle diameter, the extraction stride) instead of measuring
    it.

    `num_sizes` or `sizes` replace the factor ladder; see `ladder.box_ladder`.
    The published MD counts use a fixed window with a fixed number of boxes,
    which is `num_sizes`, and on those clouds the ladder is worth fifteen
    times what the offset convention is.
    """
    pts = np.ascontiguousarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f'points must be (n, 2), got {pts.shape}')
    if len(pts) < 2:
        raise ValueError('a point cloud needs at least 2 points')

    rungs = box_ladder(min_box, max_box, factor, num_sizes, sizes)
    if spacing is None:
        spacing = nn_spacing(pts)
    smallest = float(rungs.min())
    if smallest < spacing and not allow_below_spacing:
        raise ValueError(
            f'smallest box {smallest:.6g} is below the nearest-neighbour spacing '
            f'{spacing:.6g}. Below the spacing every box holds at most one point, the '
            f'count saturates at n = {len(pts)} and the dimension collapses toward 0, '
            f'so the fit would be measuring the sampling and not the structure. '
            f'Raise the smallest box to at least {spacing:.6g}, or pass '
            f'allow_below_spacing=True deliberately.')

    min_x, max_x = float(pts[:, 0].min()), float(pts[:, 0].max())
    min_y, max_y = float(pts[:, 1].min()), float(pts[:, 1].max())
    if offsets != 'single':
        margin = max(max_x - min_x, max_y - min_y) * MARGIN_FACTOR
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin

    counter = PointCounter(pts, backend)
    kept_sizes, kept_counts = [], []
    for box in rungs:
        fr = offset_fractions(box, offsets)
        best = None
        for fx in fr:
            for fy in fr:
                c = counter.count(box, origin(min_x, fx, box, offsets),
                                  origin(min_y, fy, box, offsets), max_x, max_y)
                best = c if best is None else min(best, c)
        if best > 0:
            kept_sizes.append(box)
            kept_counts.append(best)
    if len(kept_sizes) < 2:
        raise ValueError('not enough occupied box sizes for a fractal dimension')
    return np.array(kept_sizes), np.array(kept_counts), (min_x, min_y, max_x, max_y)
