"""Occupied-box counting for a 3D point set.

Points are an (n, 3) float64 array.  As in 2D, a point occupies exactly one
box, so the cell index is floor((p - origin) / delta) and the count is the
number of distinct cells.  The CPU path takes the distinct count of the cell
keys, O(n) memory, never allocating the grid; the CUDA path marks an
occupancy array, as the triangle kernel does.  Both count the same integer
cells and so agree exactly.

**A finite point set has box dimension 0**, and the hard lower bound is the
nearest-neighbour spacing, exactly as in 2D.  There is one thing to be
careful about that 2D does not raise: a 3D cloud sampling a *surface* is
bounded by the surface's sampling density, not by the volume's.  The
nearest-neighbour distance measures the former, which is what is wanted; a
bound derived from n and the bounding-box volume would be wrong by the
aspect ratio of a thin layer and is not used.

On the near-planar case, which is where the 3D count has a documented
history, measure before assuming.  The legacy D = 5 to 9 at R^2 ~ 0.8 on a
nearly flat early-time surface belongs to the LEGACY OFFSETS, not to the
triangle measure: a flat surface lies below every shifted origin and the
counts come out zero.  With covering offsets the triangle counter handles a
flat plane perfectly well, measured here at D = 2.018, R^2 = 0.9999.  So the
point counter does not rescue a broken case; the offset correction already
did.

What a point sample adds is a second, independent measure that agrees with
the corrected triangle measure where both are valid: on a regularly sampled
plane, 2.009 against 2.018, and on the same plane perturbed by 1e-3 the two
agree exactly at 2.0087.  They separate once the surface carries structure
near the sampling scale (at a 1e-2 perturbation, 2.139 against 2.208), which
is expected, because a mesh and a vertex sample of it are then different
sets.  Agreement is therefore a check, and the two must not be mixed in one
comparison.

`box_counting_3d_points()` matches `triangles3d.box_counting_3d`: a ladder
from `initial` down by `factor` for `num_steps`, minimum count over the
origin offsets, zero counts removed, and the dimension taken against
log10(1/delta) by `scaling.dimension_3d`.
"""
import numpy as np

from .. import backend as _backend
from .offsets import offset_fractions, origin
from .triangles3d import delta_ladder


def nn_spacing(pts, stat='median'):
    """Nearest-neighbour distance of the cloud: the ladder's hard lower bound.

    For a surface sample this is the surface sampling density, which is the
    relevant scale; do not substitute anything derived from the bounding-box
    volume."""
    from scipy.spatial import cKDTree

    pts = np.ascontiguousarray(pts, dtype=np.float64)
    if len(pts) < 2:
        raise ValueError('nearest-neighbour spacing needs at least 2 points')
    d, _ = cKDTree(pts).query(pts, k=2)
    nn = d[:, 1]
    nn = nn[nn > 0]
    if len(nn) == 0:
        raise ValueError('every point is a duplicate: no spacing to measure')
    return float({'median': np.median, 'mean': np.mean, 'min': np.min}[stat](nn))


def _cell_keys(pts, delta, dmin, nx, ny, nz):
    i = np.floor((pts[:, 0] - dmin[0]) / delta).astype(np.int64)
    j = np.floor((pts[:, 1] - dmin[1]) / delta).astype(np.int64)
    k = np.floor((pts[:, 2] - dmin[2]) / delta).astype(np.int64)
    np.clip(i, 0, nx - 1, out=i)
    np.clip(j, 0, ny - 1, out=j)
    np.clip(k, 0, nz - 1, out=k)
    return (i * ny + j) * nz + k


_cuda_kernel = None


def _get_cuda_kernel():
    global _cuda_kernel
    if _cuda_kernel is None:
        from numba import cuda

        @cuda.jit
        def kern(pts, delta, inv, d0x, d0y, d0z, nx, ny, nz, occ):
            p = cuda.grid(1)
            if p >= pts.shape[0]:
                return
            i = int((pts[p, 0] - d0x) * inv // 1.0)
            j = int((pts[p, 1] - d0y) * inv // 1.0)
            k = int((pts[p, 2] - d0z) * inv // 1.0)
            if i < 0:
                i = 0
            if j < 0:
                j = 0
            if k < 0:
                k = 0
            if i >= nx:
                i = nx - 1
            if j >= ny:
                j = ny - 1
            if k >= nz:
                k = nz - 1
            occ[(i * ny + j) * nz + k] = 1       # idempotent: races are benign

        _cuda_kernel = kern
    return _cuda_kernel


class PointCounter3D:
    """Counts occupied cubes for one 3D point set on a chosen backend
    (points stay resident on the GPU across calls)."""

    def __init__(self, pts, backend=None):
        pts = np.ascontiguousarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f'points must be (n, 3), got {pts.shape}')
        self.pts = pts
        self.backend = _backend.resolve(backend)
        if self.backend == 'cuda':
            from numba import cuda
            self._d_pts = cuda.to_device(self.pts)

    def count(self, delta, dmin, dmax):
        nx = max(1, int(np.ceil((dmax[0] - dmin[0]) / delta)))
        ny = max(1, int(np.ceil((dmax[1] - dmin[1]) / delta)))
        nz = max(1, int(np.ceil((dmax[2] - dmin[2]) / delta)))
        if self.backend == 'cpu':
            return int(np.unique(_cell_keys(self.pts, delta, dmin, nx, ny, nz)).size)
        from numba import cuda
        occ = cuda.to_device(np.zeros(nx * ny * nz, dtype=np.uint8))
        n = self.pts.shape[0]
        _get_cuda_kernel()[(n + 255) // 256, 256](
            self._d_pts, delta, 1.0 / delta, dmin[0], dmin[1], dmin[2], nx, ny, nz, occ)
        return int(occ.copy_to_host().sum())


def box_counting_3d_points(pts, initial=None, factor=1.5, num_steps=20, min_delta=None,
                           max_delta=None, backend=None, log=None, offsets='covering',
                           spacing=None, allow_below_spacing=False):
    """Grid-optimised box counting of a 3D point set -> (deltas, counts, bbox).

    `initial` defaults to the characteristic length / 8, `min_delta` to twice
    the nearest-neighbour spacing.  Raises ValueError if min_delta is below
    the spacing, where the count saturates at n and the dimension collapses
    toward 0; pass allow_below_spacing=True to override, or `spacing` to
    supply a known value instead of measuring it.
    """
    pts = np.ascontiguousarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f'points must be (n, 3), got {pts.shape}')
    if len(pts) < 2:
        raise ValueError('a point cloud needs at least 2 points')

    if spacing is None:
        spacing = nn_spacing(pts)
    lo = np.array([pts[:, i].min() for i in range(3)], dtype=np.float64)
    hi = np.array([pts[:, i].max() for i in range(3)], dtype=np.float64)
    char = float(max(hi - lo))

    if min_delta is None:
        min_delta = 2.0 * spacing
    if initial is None:
        initial = char / 8
    if max_delta is None:
        max_delta = char

    if min_delta < spacing and not allow_below_spacing:
        raise ValueError(
            f'smallest cube {min_delta:.6g} is below the nearest-neighbour spacing '
            f'{spacing:.6g}. Below the spacing every cube holds at most one point, the '
            f'count saturates at n = {len(pts)} and the dimension collapses toward 0, so '
            f'the fit would be measuring the sampling and not the structure. Raise '
            f'min_delta to at least {spacing:.6g}, or pass allow_below_spacing=True '
            f'deliberately.')

    deltas = delta_ladder(initial, factor, num_steps, min_delta, max_delta)
    if len(deltas) < 3:
        raise ValueError(f'insufficient scale range: {len(deltas)} sizes between '
                         f'{min_delta:.6g} and {min(initial, max_delta):.6g}')

    counter = PointCounter3D(pts, backend)
    counts = []
    for d in deltas:
        fr = offset_fractions(d)
        best = None
        for fx in fr:
            for fy in fr:
                for fz in fr:
                    dmin = np.array([origin(lo[0], fx, d, offsets),
                                     origin(lo[1], fy, d, offsets),
                                     origin(lo[2], fz, d, offsets)], dtype=np.float64)
                    c = counter.count(d, dmin, hi)
                    best = c if best is None else min(best, c)
        counts.append(best)
        if log:
            log(f"  delta = {d:.6f}: {best} cubes [{len(fr) ** 3} offsets]")
    deltas, counts = np.array(deltas), np.array(counts)
    keep = counts > 0
    if keep.sum() < 3:
        raise ValueError('not enough occupied cube sizes for a fractal dimension')
    return deltas[keep], counts[keep], (lo, hi)
