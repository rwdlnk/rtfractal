"""Per-cube measures of a triangulated surface for the 3D multifractal analysis.

Legacy: fractal3d.multifractal_3d.MultifractalAnalyzer3D (measure='area' in
the pipeline).  Each triangle's area is split equally among the cubes it
intersects (SAT test, triangles3d.py); 'count' counts triangles per cube.
Triangles of zero area contribute nothing to the area measure, so a cube hit
only by such triangles is not occupied.  The candidate cubes of a triangle
are those its bounding box touches (no padding), as in the legacy measure
kernel.

Same tests on every backend.  NVVM contracts a*b + c into a fused
multiply-add by default (numba exposes no switch), so a CUDA build of the SAT
test rounds differently from the CPU build and disagrees for triangles lying
on a cube face to the last bit (common for LBM isosurfaces).  The kernels
here take every product that feeds a sum through an injected `mul` (plain
a*b on the CPU, libdevice.dmul_rn on CUDA, which is never fused).  The
legacy CPU path was not contracted either.

Determinism.  The legacy GPU kernel accumulated cube areas with float
atomics, so its last bits depend on thread scheduling and its results are not
reproducible bit for bit, even run to run.  Here every backend lists the
cubes each triangle hits (count pass, prefix sum, write pass, in parallel),
and the areas are then summed serially in triangle order, with the occupied
cubes listed in first-touch order.  That is exactly the legacy CPU path
(compute_cube_measures: a dict filled triangle by triangle), so CPU and CUDA
give identical measures, identical to the legacy CPU code.
"""
import numpy as np
from numba import njit, prange

from .. import backend as _backend
from .offsets import offset_fractions, origin
from .triangles3d import _max3, _max3_py, _min3, _min3_py


def _make_sat(min3, max3, mul):
    """Triangle/cube separating-axis test: triangles3d's legacy SAT, with every
    product that feeds a sum taken through `mul` (see module note on FMA)."""
    def sat(v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, c0x, c0y, c0z, c1x, c1y, c1z):
        cx = (c0x + c1x) * 0.5
        cy = (c0y + c1y) * 0.5
        cz = (c0z + c1z) * 0.5
        hx = (c1x - c0x) * 0.5
        hy = (c1y - c0y) * 0.5
        hz = (c1z - c0z) * 0.5
        t0x = v0x - cx
        t0y = v0y - cy
        t0z = v0z - cz
        t1x = v1x - cx
        t1y = v1y - cy
        t1z = v1z - cz
        t2x = v2x - cx
        t2y = v2y - cy
        t2z = v2z - cz
        if min3(t0x, t1x, t2x) > hx or max3(t0x, t1x, t2x) < -hx:
            return False
        if min3(t0y, t1y, t2y) > hy or max3(t0y, t1y, t2y) < -hy:
            return False
        if min3(t0z, t1z, t2z) > hz or max3(t0z, t1z, t2z) < -hz:
            return False
        e0x = t1x - t0x
        e0y = t1y - t0y
        e0z = t1z - t0z
        e1x = t2x - t1x
        e1y = t2y - t1y
        e1z = t2z - t1z
        e2x = t0x - t2x
        e2y = t0y - t2y
        e2z = t0z - t2z
        nx_ = mul(e0y, e1z) - mul(e0z, e1y)
        ny_ = mul(e0z, e1x) - mul(e0x, e1z)
        nz_ = mul(e0x, e1y) - mul(e0y, e1x)
        d = -(mul(nx_, t0x) + mul(ny_, t0y) + mul(nz_, t0z))
        if abs(d) > mul(hx, abs(nx_)) + mul(hy, abs(ny_)) + mul(hz, abs(nz_)):
            return False
        for k in range(3):
            if k == 0:
                ex, ey, ez = e0x, e0y, e0z
            elif k == 1:
                ex, ey, ez = e1x, e1y, e1z
            else:
                ex, ey, ez = e2x, e2y, e2z
            r = mul(hy, abs(ez)) + mul(hz, abs(ey))
            p0 = mul(t0y, ez) - mul(t0z, ey)
            p1 = mul(t1y, ez) - mul(t1z, ey)
            p2 = mul(t2y, ez) - mul(t2z, ey)
            if min3(p0, p1, p2) > r or max3(p0, p1, p2) < -r:
                return False
            r = mul(hx, abs(ez)) + mul(hz, abs(ex))
            p0 = mul(t0z, ex) - mul(t0x, ez)
            p1 = mul(t1z, ex) - mul(t1x, ez)
            p2 = mul(t2z, ex) - mul(t2x, ez)
            if min3(p0, p1, p2) > r or max3(p0, p1, p2) < -r:
                return False
            r = mul(hx, abs(ey)) + mul(hy, abs(ex))
            p0 = mul(t0x, ey) - mul(t0y, ex)
            p1 = mul(t1x, ey) - mul(t1y, ex)
            p2 = mul(t2x, ey) - mul(t2y, ex)
            if min3(p0, p1, p2) > r or max3(p0, p1, p2) < -r:
                return False
        return True
    return sat


def _make_tri_hits(min3, max3, sat, mul):
    def tri_hits(tv, t, delta, inv, d0x, d0y, d0z, nx, ny, nz, out, start):
        """Number of cubes triangle t hits; if start >= 0 also write their flat
        indices (i * ny * nz + j * nz + k, loop order i, j, k) to out[start:]."""
        v0x, v0y, v0z = tv[t, 0, 0], tv[t, 0, 1], tv[t, 0, 2]
        v1x, v1y, v1z = tv[t, 1, 0], tv[t, 1, 1], tv[t, 1, 2]
        v2x, v2y, v2z = tv[t, 2, 0], tv[t, 2, 1], tv[t, 2, 2]
        i0 = int((min3(v0x, v1x, v2x) - d0x) * inv)
        i1 = int((max3(v0x, v1x, v2x) - d0x) * inv)
        j0 = int((min3(v0y, v1y, v2y) - d0y) * inv)
        j1 = int((max3(v0y, v1y, v2y) - d0y) * inv)
        k0 = int((min3(v0z, v1z, v2z) - d0z) * inv)
        k1 = int((max3(v0z, v1z, v2z) - d0z) * inv)
        if i0 < 0:
            i0 = 0
        if j0 < 0:
            j0 = 0
        if k0 < 0:
            k0 = 0
        if i1 >= nx:
            i1 = nx - 1
        if j1 >= ny:
            j1 = ny - 1
        if k1 >= nz:
            k1 = nz - 1
        n = 0
        for i in range(i0, i1 + 1):
            cx0 = d0x + mul(i, delta)
            cx1 = cx0 + delta
            for j in range(j0, j1 + 1):
                cy0 = d0y + mul(j, delta)
                cy1 = cy0 + delta
                for k in range(k0, k1 + 1):
                    cz0 = d0z + mul(k, delta)
                    if sat(v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, cx0, cy0, cz0, cx1, cy1, cz0 + delta):
                        if start >= 0:
                            out[start + n] = i * (ny * nz) + j * nz + k
                        n += 1
        return n
    return tri_hits


def _mul_py(a, b):
    return a * b


_mul = njit(_mul_py)
_tri_hits = njit(_make_tri_hits(_min3, _max3, njit(_make_sat(_min3, _max3, _mul)), _mul))


@njit(parallel=True)
def _hits_cpu(tv, delta, inv, d0x, d0y, d0z, nx, ny, nz):
    n = tv.shape[0]
    cnt = np.empty(n, dtype=np.int64)
    dummy = np.empty(1, dtype=np.int64)
    for t in prange(n):
        cnt[t] = _tri_hits(tv, t, delta, inv, d0x, d0y, d0z, nx, ny, nz, dummy, -1)
    off = np.zeros(n + 1, dtype=np.int64)
    for t in range(n):
        off[t + 1] = off[t] + cnt[t]
    out = np.empty(off[n], dtype=np.int64)
    for t in prange(n):
        _tri_hits(tv, t, delta, inv, d0x, d0y, d0z, nx, ny, nz, out, off[t])
    return out, off


@njit
def _accumulate(hits, off, areas, ncell, area_measure):
    """Serial, triangle-order accumulation (the legacy CPU dict) -> measures of
    the occupied cubes in first-touch order."""
    slot = np.full(ncell, -1, dtype=np.int32)
    vals = np.empty(hits.shape[0], dtype=np.float64)
    m = 0
    for t in range(off.shape[0] - 1):
        nh = off[t + 1] - off[t]
        if area_measure:
            a = areas[t]
            if a <= 0:
                continue
            if nh == 0:
                continue
            v = a / nh
        else:
            v = 1.0
        for h in range(off[t], off[t + 1]):
            c = hits[h]
            k = slot[c]
            if k < 0:
                k = m
                slot[c] = m
                vals[m] = 0.0
                m += 1
            vals[k] = vals[k] + v
    return vals[:m]


_cuda_kernels = None


def _get_cuda_kernels():
    global _cuda_kernels
    if _cuda_kernels is None:
        from numba import cuda
        dev = cuda.jit(device=True)
        min3, max3 = dev(_min3_py), dev(_max3_py)
        from numba.cuda import libdevice

        @cuda.jit(device=True)
        def mul(a, b):
            return libdevice.dmul_rn(a, b)      # never fused into an FMA: same rounding as the CPU

        hits = dev(_make_tri_hits(min3, max3, dev(_make_sat(min3, max3, mul)), mul))

        @cuda.jit
        def kcount(tv, delta, inv, d0x, d0y, d0z, nx, ny, nz, cnt, dummy):
            t = cuda.grid(1)
            if t < tv.shape[0]:
                cnt[t] = hits(tv, t, delta, inv, d0x, d0y, d0z, nx, ny, nz, dummy, -1)

        @cuda.jit
        def kwrite(tv, delta, inv, d0x, d0y, d0z, nx, ny, nz, off, out):
            t = cuda.grid(1)
            if t < tv.shape[0]:
                hits(tv, t, delta, inv, d0x, d0y, d0z, nx, ny, nz, out, off[t])

        _cuda_kernels = (kcount, kwrite)
    return _cuda_kernels


MEASURES = ('area', 'count')


class TriangleMeasure:
    """Per-cube measures of one surface on a chosen backend."""

    def __init__(self, mesh, measure='area', backend=None):
        if measure not in MEASURES:
            raise ValueError(f"measure must be one of {MEASURES}")
        self.measure = measure
        self.tv = np.ascontiguousarray(mesh.triangle_vertices(), dtype=np.float64)
        self.areas = np.ascontiguousarray(mesh.triangle_areas(), dtype=np.float64)
        self.backend = _backend.resolve(backend)
        if self.backend == 'cuda':
            from numba import cuda
            self._d_tv = cuda.to_device(self.tv)

    def hits(self, delta, dmin, nx, ny, nz):
        """-> (hits, off): flat cube indices hit by triangle t are hits[off[t]:off[t+1]]."""
        inv = 1.0 / delta
        if self.backend == 'cpu':
            return _hits_cpu(self.tv, delta, inv, dmin[0], dmin[1], dmin[2], nx, ny, nz)
        from numba import cuda
        n = self.tv.shape[0]
        kcount, kwrite = _get_cuda_kernels()
        blocks = (n + 255) // 256
        d_cnt = cuda.device_array(n, dtype=np.int64)
        d_dummy = cuda.device_array(1, dtype=np.int64)
        kcount[blocks, 256](self._d_tv, delta, inv, dmin[0], dmin[1], dmin[2], nx, ny, nz, d_cnt, d_dummy)
        off = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(d_cnt.copy_to_host(), out=off[1:])
        d_out = cuda.device_array(max(1, int(off[-1])), dtype=np.int64)
        kwrite[blocks, 256](self._d_tv, delta, inv, dmin[0], dmin[1], dmin[2], nx, ny, nz, cuda.to_device(off), d_out)
        return d_out.copy_to_host()[:off[-1]], off

    def measures(self, delta, dmin, dmax):
        """Non-zero cube measures (first-touch order) for the grid with origin dmin."""
        nx = max(1, int(np.ceil((dmax[0] - dmin[0]) / delta)))
        ny = max(1, int(np.ceil((dmax[1] - dmin[1]) / delta)))
        nz = max(1, int(np.ceil((dmax[2] - dmin[2]) / delta)))
        ncell = nx * ny * nz
        if ncell >= 2 ** 31:
            raise ValueError(f'cube grid too large ({ncell} cells) at delta={delta}')
        h, off = self.hits(delta, dmin, nx, ny, nz)
        return _accumulate(h, off, self.areas, ncell, self.measure == 'area')


def grid_optimized_measure(counter, delta, bbox, offsets='covering'):
    """Cube measures at one size, minimised over the origin offsets (first
    offset reaching the minimum number of occupied cubes wins, as in legacy).

    bbox: the mesh bounding box (BBox).  -> (measures, n_occupied, n_offsets_tested)."""
    dmax = np.array([bbox.max_x, bbox.max_y, bbox.max_z], dtype=np.float64)
    fr = offset_fractions(delta)
    best_n, best, tests = None, np.array([]), 0
    for fx in fr:
        for fy in fr:
            for fz in fr:
                tests += 1
                dmin = np.array([origin(bbox.min_x, fx, delta, offsets), origin(bbox.min_y, fy, delta, offsets),
                                 origin(bbox.min_z, fz, delta, offsets)], dtype=np.float64)
                m = counter.measures(delta, dmin, dmax)
                if best_n is None or len(m) < best_n:
                    best_n, best = len(m), m
    return best, best_n, tests
