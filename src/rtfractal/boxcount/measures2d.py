"""Per-box segment counts for the 2D multifractal measure.

The measure of a box is the number of interface segments that intersect it
(legacy RTAnalyzer.compute_multifractal_spectrum, GPU path
fast_counting_gpu_2d.count_segments_per_box_gpu, which produced the
reference results).  The candidate boxes of a segment are those its bounding
box touches, padded by one box, tested with the same Liang-Barsky test as the
occupancy counter in segments2d.py.

Box corners are dmin + i*delta.  NVVM contracts a*b + c into a fused
multiply-add by default (numba exposes no switch), which rounds differently
from the CPU; the kernels therefore multiply through an injected `mul`
(plain a*b on the CPU, libdevice.dmul_rn on CUDA, which is never fused), so
both backends test exactly the same boxes.  (The legacy GPU kernel was
contracted; it can differ from these only for a segment touching a box edge
to the last bit.)

Each backend lists, for every segment, the flat indices (i * ny + j) of the
boxes it hits (count pass, prefix sum, write pass), so CPU and CUDA give the
same integer counts.  Integer counts don't depend on summation order.

grid_optimized_measure() keeps, as the legacy code does, the distribution of
the FIRST origin offset that reaches the minimum number of occupied boxes;
the occupied boxes are listed in flat (i * ny + j) order, as the legacy GPU
path returns them.
"""
import numpy as np
from numba import njit, prange

from .. import backend as _backend
from .offsets import offset_fractions, origin
from .segments2d import _lb_edge, _liang_barsky


def _make_seg_hits(lb, mul):
    def seg_hits(segs, s, delta, inv, dmx, dmy, nx, ny, out, start):
        """Number of boxes segment s hits; if start >= 0 also write their flat
        indices to out[start:]."""
        x1 = segs[s, 0]
        y1 = segs[s, 1]
        x2 = segs[s, 2]
        y2 = segs[s, 3]
        if x1 < x2:
            sx0, sx1 = x1, x2
        else:
            sx0, sx1 = x2, x1
        if y1 < y2:
            sy0, sy1 = y1, y2
        else:
            sy0, sy1 = y2, y1
        i0 = int((sx0 - dmx) * inv) - 1
        i1 = int((sx1 - dmx) * inv) + 1
        j0 = int((sy0 - dmy) * inv) - 1
        j1 = int((sy1 - dmy) * inv) + 1
        if i0 < 0:
            i0 = 0
        if j0 < 0:
            j0 = 0
        if i1 >= nx:
            i1 = nx - 1
        if j1 >= ny:
            j1 = ny - 1
        n = 0
        for i in range(i0, i1 + 1):
            bx0 = dmx + mul(i, delta)
            bx1 = dmx + mul(i + 1, delta)
            for j in range(j0, j1 + 1):
                if lb(x1, y1, x2, y2, bx0, dmy + mul(j, delta), bx1, dmy + mul(j + 1, delta)):
                    if start >= 0:
                        out[start + n] = i * ny + j
                    n += 1
        return n
    return seg_hits


def _mul_py(a, b):
    return a * b


_seg_hits = njit(_make_seg_hits(_liang_barsky, njit(_mul_py)))


@njit(parallel=True)
def _hits_cpu(segs, delta, inv, dmx, dmy, nx, ny):
    n = segs.shape[0]
    cnt = np.empty(n, dtype=np.int64)
    dummy = np.empty(1, dtype=np.int64)
    for s in prange(n):
        cnt[s] = _seg_hits(segs, s, delta, inv, dmx, dmy, nx, ny, dummy, -1)
    off = np.zeros(n + 1, dtype=np.int64)
    for s in range(n):
        off[s + 1] = off[s] + cnt[s]
    out = np.empty(off[n], dtype=np.int64)
    for s in prange(n):
        _seg_hits(segs, s, delta, inv, dmx, dmy, nx, ny, out, off[s])
    return out, off


_cuda_kernels = None


def _get_cuda_kernels():
    global _cuda_kernels
    if _cuda_kernels is None:
        from numba import cuda
        dev = cuda.jit(device=True)
        lbe = dev(_lb_edge.py_func)

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

        from numba.cuda import libdevice

        @cuda.jit(device=True)
        def mul(a, b):
            return libdevice.dmul_rn(a, b)      # never fused into an FMA: same rounding as the CPU

        hits = dev(_make_seg_hits(lb, mul))

        @cuda.jit
        def kcount(segs, delta, inv, dmx, dmy, nx, ny, cnt, dummy):
            s = cuda.grid(1)
            if s < segs.shape[0]:
                cnt[s] = hits(segs, s, delta, inv, dmx, dmy, nx, ny, dummy, -1)

        @cuda.jit
        def kwrite(segs, delta, inv, dmx, dmy, nx, ny, off, out):
            s = cuda.grid(1)
            if s < segs.shape[0]:
                hits(segs, s, delta, inv, dmx, dmy, nx, ny, out, off[s])

        _cuda_kernels = (kcount, kwrite)
    return _cuda_kernels


class SegmentMeasure:
    """Per-box segment counts for one segment set on a chosen backend."""

    def __init__(self, segs, backend=None):
        self.segs = np.ascontiguousarray(segs, dtype=np.float64)
        self.backend = _backend.resolve(backend)
        if self.backend == 'cuda':
            from numba import cuda
            self._d_segs = cuda.to_device(self.segs)

    def hits(self, delta, dmin_x, dmin_y, nx, ny):
        """-> (hits, off): flat box indices hit by segment s are hits[off[s]:off[s+1]]."""
        inv = 1.0 / delta
        if self.backend == 'cpu':
            return _hits_cpu(self.segs, delta, inv, dmin_x, dmin_y, nx, ny)
        from numba import cuda
        n = self.segs.shape[0]
        kcount, kwrite = _get_cuda_kernels()
        blocks = (n + 255) // 256
        d_cnt = cuda.device_array(n, dtype=np.int64)
        d_dummy = cuda.device_array(1, dtype=np.int64)
        kcount[blocks, 256](self._d_segs, delta, inv, dmin_x, dmin_y, nx, ny, d_cnt, d_dummy)
        off = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(d_cnt.copy_to_host(), out=off[1:])
        d_out = cuda.device_array(max(1, int(off[-1])), dtype=np.int64)
        kwrite[blocks, 256](self._d_segs, delta, inv, dmin_x, dmin_y, nx, ny, cuda.to_device(off), d_out)
        return d_out.copy_to_host()[:off[-1]], off

    def box_counts(self, delta, dmin_x, dmin_y, dmax_x, dmax_y):
        """Segments per box, flat (nx * ny, int32; index i * ny + j) -> (counts, nx, ny)."""
        nx = max(1, int(np.ceil((dmax_x - dmin_x) / delta)))
        ny = max(1, int(np.ceil((dmax_y - dmin_y) / delta)))
        h, _ = self.hits(delta, dmin_x, dmin_y, nx, ny)
        return np.bincount(h, minlength=nx * ny).astype(np.int32), nx, ny


def grid_optimized_measure(counter, box, bbox, offsets='covering'):
    """Box measure at one size, minimised over the origin offsets.

    bbox = (min_x, min_y, max_x, max_y), margin already applied.
    -> (occupied box counts in flat order, n_occupied, n_offsets_tested)."""
    min_x, min_y, max_x, max_y = bbox
    best_n, best, tests = None, np.array([], dtype=np.int32), 0
    for fx in offset_fractions(box):
        for fy in offset_fractions(box):
            tests += 1
            c, _, _ = counter.box_counts(box, origin(min_x, fx, box, offsets), origin(min_y, fy, box, offsets),
                                         max_x, max_y)
            n = int(np.count_nonzero(c))
            if best_n is None or n < best_n:
                best_n, best = n, c[c > 0]
    return best, best_n, tests
