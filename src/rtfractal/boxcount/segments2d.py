"""Occupied-box counting for 2D line segments (Liang-Barsky segment/box test).

Segments are an (n, 4) float64 array of (x1, y1, x2, y2).  The CPU (numba)
and CUDA kernels are line-for-line the same algorithm as the legacy GPU kernel
(fast_counting_gpu_2d.py), which produced the reference results: one pass per
segment over the boxes its bounding box touches (padded by one box), testing
each with Liang-Barsky.

box_counting_2d() reproduces FractalAnalyzer.box_counting_unified with grid
optimisation: 1% bounding-box margin, ladder max -> min dividing by `factor`,
minimum count over the origin offsets.
"""
import numpy as np
from numba import njit, prange

from .. import backend as _backend
from .offsets import offset_fractions, origin


@njit(cache=True, inline='always')
def _lb_edge(p, q, t_min, t_max):
    # returns (t_min, t_max, reject)
    if p == 0.0:
        return t_min, t_max, q < 0.0
    t = q / p
    if p < 0.0:
        if t > t_min:
            t_min = t
    else:
        if t < t_max:
            t_max = t
    return t_min, t_max, t_min > t_max


@njit(cache=True)
def _liang_barsky(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    dx = x2 - x1
    dy = y2 - y1
    t0, t1, rej = _lb_edge(-dx, x1 - xmin, 0.0, 1.0)
    if rej:
        return False
    t0, t1, rej = _lb_edge(dx, xmax - x1, t0, t1)
    if rej:
        return False
    t0, t1, rej = _lb_edge(-dy, y1 - ymin, t0, t1)
    if rej:
        return False
    t0, t1, rej = _lb_edge(dy, ymax - y1, t0, t1)
    return not rej


@njit(parallel=True, cache=True)
def _occupied_cpu(segs, delta, inv_delta, dmin_x, dmin_y, nx, ny):
    occ = np.zeros(nx * ny, dtype=np.uint8)
    for s in prange(segs.shape[0]):
        x1, y1, x2, y2 = segs[s, 0], segs[s, 1], segs[s, 2], segs[s, 3]
        sx0, sx1 = (x1, x2) if x1 < x2 else (x2, x1)
        sy0, sy1 = (y1, y2) if y1 < y2 else (y2, y1)
        i0 = max(0, int((sx0 - dmin_x) * inv_delta) - 1)
        i1 = min(nx - 1, int((sx1 - dmin_x) * inv_delta) + 1)
        j0 = max(0, int((sy0 - dmin_y) * inv_delta) - 1)
        j1 = min(ny - 1, int((sy1 - dmin_y) * inv_delta) + 1)
        for i in range(i0, i1 + 1):
            bx0 = dmin_x + i * delta
            bx1 = dmin_x + (i + 1) * delta
            for j in range(j0, j1 + 1):
                if _liang_barsky(x1, y1, x2, y2, bx0, dmin_y + j * delta, bx1, dmin_y + (j + 1) * delta):
                    occ[i * ny + j] = 1     # idempotent: races are benign
    return occ.sum()


_cuda_kernel = None


def _get_cuda_kernel():
    global _cuda_kernel
    if _cuda_kernel is None:
        from numba import cuda

        lbe = cuda.jit(device=True)(_lb_edge.py_func)

        @cuda.jit(device=True)
        def lb(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
            dx = x2 - x1
            dy = y2 - y1
            t0, t1, rej = lbe(-dx, x1 - xmin, 0.0, 1.0)
            if rej:
                return False
            t0, t1, rej = lbe(dx, xmax - x1, t0, t1)
            if rej:
                return False
            t0, t1, rej = lbe(-dy, y1 - ymin, t0, t1)
            if rej:
                return False
            t0, t1, rej = lbe(dy, ymax - y1, t0, t1)
            return not rej

        @cuda.jit
        def kern(segs, delta, inv_delta, dmin_x, dmin_y, nx, ny, occ):
            s = cuda.grid(1)
            if s >= segs.shape[0]:
                return
            x1, y1, x2, y2 = segs[s, 0], segs[s, 1], segs[s, 2], segs[s, 3]
            if x1 < x2:
                sx0, sx1 = x1, x2
            else:
                sx0, sx1 = x2, x1
            if y1 < y2:
                sy0, sy1 = y1, y2
            else:
                sy0, sy1 = y2, y1
            i0 = int((sx0 - dmin_x) * inv_delta) - 1
            i1 = int((sx1 - dmin_x) * inv_delta) + 1
            j0 = int((sy0 - dmin_y) * inv_delta) - 1
            j1 = int((sy1 - dmin_y) * inv_delta) + 1
            if i0 < 0:
                i0 = 0
            if j0 < 0:
                j0 = 0
            if i1 >= nx:
                i1 = nx - 1
            if j1 >= ny:
                j1 = ny - 1
            for i in range(i0, i1 + 1):
                bx0 = dmin_x + i * delta
                bx1 = dmin_x + (i + 1) * delta
                for j in range(j0, j1 + 1):
                    if lb(x1, y1, x2, y2, bx0, dmin_y + j * delta, bx1, dmin_y + (j + 1) * delta):
                        occ[i * ny + j] = 1

        _cuda_kernel = kern
    return _cuda_kernel


class SegmentCounter:
    """Counts occupied boxes for one segment set on a chosen backend
    (segments stay resident on the GPU across calls)."""

    def __init__(self, segs, backend=None):
        self.segs = np.ascontiguousarray(segs, dtype=np.float64)
        self.backend = _backend.resolve(backend)
        if self.backend == 'cuda':
            from numba import cuda
            self._d_segs = cuda.to_device(self.segs)

    def count(self, delta, dmin_x, dmin_y, dmax_x, dmax_y):
        nx = max(1, int(np.ceil((dmax_x - dmin_x) / delta)))
        ny = max(1, int(np.ceil((dmax_y - dmin_y) / delta)))
        inv = 1.0 / delta
        if self.backend == 'cpu':
            return int(_occupied_cpu(self.segs, delta, inv, dmin_x, dmin_y, nx, ny))
        from numba import cuda
        occ = cuda.to_device(np.zeros(nx * ny, dtype=np.uint8))
        n = self.segs.shape[0]
        _get_cuda_kernel()[(n + 255) // 256, 256](self._d_segs, delta, inv, dmin_x, dmin_y, nx, ny, occ)
        return int(occ.copy_to_host().sum())


MARGIN_FACTOR = 0.01


def box_counting_2d(segs, min_box, max_box, factor=1.5, backend=None, offsets='covering'):
    """Grid-optimised box counting -> (box_sizes, box_counts, bbox).

    Sizes with zero occupied boxes are skipped (legacy behaviour)."""
    segs = np.ascontiguousarray(segs, dtype=np.float64)
    xs = np.concatenate([segs[:, 0], segs[:, 2]])
    ys = np.concatenate([segs[:, 1], segs[:, 3]])
    min_x, max_x, min_y, max_y = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    margin = max(max_x - min_x, max_y - min_y) * MARGIN_FACTOR
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin
    counter = SegmentCounter(segs, backend)
    sizes, counts = [], []
    box = max_box
    while box >= min_box:
        best = None
        for fx in offset_fractions(box):
            for fy in offset_fractions(box):
                c = counter.count(box, origin(min_x, fx, box, offsets), origin(min_y, fy, box, offsets), max_x, max_y)
                best = c if best is None else min(best, c)
        if best > 0:
            sizes.append(box)
            counts.append(best)
        box /= factor
    if len(sizes) < 2:
        raise ValueError('not enough occupied box sizes for a fractal dimension')
    return np.array(sizes), np.array(counts), (min_x, min_y, max_x, max_y)
