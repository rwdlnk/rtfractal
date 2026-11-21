"""Occupied-cube counting for triangulated surfaces (separating-axis test).

The SAT test and the per-triangle cube range are the legacy fractal3d GPU
kernel (fast_counting_gpu.py), which produced the reference results.  The
CPU path marks the same occupancy grid in parallel.  (The legacy numba CPU
path kept at most 27 cubes per triangle and could undercount; it is not
reproduced.)

The kernels are written once as plain Python and compiled for each target;
min3/max3 are the legacy conditional forms (builtin min/max do not compile
in CUDA device code on all numba versions).

box_counting_3d() reproduces fractal3d.compute_fractal_dimension_3d's
ladder: start at `initial`, divide by `factor`, at most `num_steps` sizes,
stop below `min_delta`, skip sizes above `max_delta`; minimum count over the
origin offsets; zero counts dropped.
"""
import numpy as np
from numba import njit, prange

from .. import backend as _backend
from .offsets import offset_fractions, origin


def _min3_py(a, b, c):
    t = a if a < b else b
    return t if t < c else c


def _max3_py(a, b, c):
    t = a if a > b else b
    return t if t > c else c


def _make_sat(min3, max3):
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
        # box-axis (AABB) tests
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
        # triangle-normal test
        nx_ = e0y * e1z - e0z * e1y
        ny_ = e0z * e1x - e0x * e1z
        nz_ = e0x * e1y - e0y * e1x
        d = -(nx_ * t0x + ny_ * t0y + nz_ * t0z)
        if abs(d) > hx * abs(nx_) + hy * abs(ny_) + hz * abs(nz_):
            return False
        # nine edge x box-axis tests, in the legacy order e0, e1, e2 x (X, Y, Z)
        for k in range(3):
            if k == 0:
                ex, ey, ez = e0x, e0y, e0z
            elif k == 1:
                ex, ey, ez = e1x, e1y, e1z
            else:
                ex, ey, ez = e2x, e2y, e2z
            r = hy * abs(ez) + hz * abs(ey)
            p0 = t0y * ez - t0z * ey
            p1 = t1y * ez - t1z * ey
            p2 = t2y * ez - t2z * ey
            if min3(p0, p1, p2) > r or max3(p0, p1, p2) < -r:
                return False
            r = hx * abs(ez) + hz * abs(ex)
            p0 = t0z * ex - t0x * ez
            p1 = t1z * ex - t1x * ez
            p2 = t2z * ex - t2x * ez
            if min3(p0, p1, p2) > r or max3(p0, p1, p2) < -r:
                return False
            r = hx * abs(ey) + hy * abs(ex)
            p0 = t0x * ey - t0y * ex
            p1 = t1x * ey - t1y * ex
            p2 = t2x * ey - t2y * ex
            if min3(p0, p1, p2) > r or max3(p0, p1, p2) < -r:
                return False
        return True
    return sat


def _make_mark(min3, max3, sat):
    """Mark every cube triangle t intersects (legacy per-triangle cube range)."""
    def mark(tv, t, delta, inv, d0x, d0y, d0z, nx, ny, nz, occ):
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
        for i in range(i0, i1 + 1):
            cx0 = d0x + i * delta
            cx1 = cx0 + delta
            for j in range(j0, j1 + 1):
                cy0 = d0y + j * delta
                cy1 = cy0 + delta
                for k in range(k0, k1 + 1):
                    cz0 = d0z + k * delta
                    if sat(v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, cx0, cy0, cz0, cx1, cy1, cz0 + delta):
                        occ[i * (ny * nz) + j * nz + k] = 1     # idempotent: races are benign
    return mark


_min3, _max3 = njit(cache=True)(_min3_py), njit(cache=True)(_max3_py)
_sat = njit(_make_sat(_min3, _max3))
_mark = njit(_make_mark(_min3, _max3, _sat))


@njit(parallel=True)
def _occupied_cpu(tv, delta, inv, d0x, d0y, d0z, nx, ny, nz):
    occ = np.zeros(nx * ny * nz, dtype=np.uint8)
    for t in prange(tv.shape[0]):
        _mark(tv, t, delta, inv, d0x, d0y, d0z, nx, ny, nz, occ)
    return occ.sum()


_cuda_kernel = None


def _get_cuda_kernel():
    global _cuda_kernel
    if _cuda_kernel is None:
        from numba import cuda
        dev = cuda.jit(device=True)
        min3, max3 = dev(_min3_py), dev(_max3_py)
        mark = dev(_make_mark(min3, max3, dev(_make_sat(min3, max3))))

        @cuda.jit
        def kern(tv, delta, inv, d0x, d0y, d0z, nx, ny, nz, occ):
            t = cuda.grid(1)
            if t < tv.shape[0]:
                mark(tv, t, delta, inv, d0x, d0y, d0z, nx, ny, nz, occ)

        _cuda_kernel = kern
    return _cuda_kernel


class TriangleCounter:
    def __init__(self, tri_verts, backend=None):
        self.tv = np.ascontiguousarray(tri_verts, dtype=np.float64)
        self.backend = _backend.resolve(backend)
        if self.backend == 'cuda':
            from numba import cuda
            self._d_tv = cuda.to_device(self.tv)

    def count(self, delta, dmin, dmax):
        nx = max(1, int(np.ceil((dmax[0] - dmin[0]) / delta)))
        ny = max(1, int(np.ceil((dmax[1] - dmin[1]) / delta)))
        nz = max(1, int(np.ceil((dmax[2] - dmin[2]) / delta)))
        inv = 1.0 / delta
        if self.backend == 'cpu':
            return int(_occupied_cpu(self.tv, delta, inv, dmin[0], dmin[1], dmin[2], nx, ny, nz))
        from numba import cuda
        occ = cuda.to_device(np.zeros(nx * ny * nz, dtype=np.uint8))
        n = self.tv.shape[0]
        _get_cuda_kernel()[(n + 255) // 256, 256](self._d_tv, delta, inv, dmin[0], dmin[1], dmin[2],
                                                  nx, ny, nz, occ)
        return int(occ.copy_to_host().sum())


def delta_ladder(initial, factor, num_steps, min_delta, max_delta):
    deltas, d = [], initial
    for _ in range(num_steps):
        if d < min_delta:
            break
        if d > max_delta:
            d = d / factor
            continue
        deltas.append(d)
        d = d / factor
    return deltas


def box_counting_3d(mesh, initial, factor=1.5, num_steps=20, min_delta=None, max_delta=None,
                    backend=None, log=None, offsets='covering'):
    """-> (deltas, counts) with zero counts removed."""
    bb = mesh.bbox
    char = bb.characteristic_length
    min_delta = char / 500 if min_delta is None else min_delta
    max_delta = char if max_delta is None else max_delta
    deltas = delta_ladder(initial, factor, num_steps, min_delta, max_delta)
    if len(deltas) < 3:
        raise ValueError(f'insufficient scale range: {len(deltas)} sizes')
    counter = TriangleCounter(mesh.triangle_vertices(), backend)
    dmax = np.array([bb.max_x, bb.max_y, bb.max_z], dtype=np.float64)
    counts = []
    for d in deltas:
        fr = offset_fractions(d, offsets)
        best = None
        for fx in fr:
            for fy in fr:
                for fz in fr:
                    dmin = np.array([origin(bb.min_x, fx, d, offsets), origin(bb.min_y, fy, d, offsets),
                                     origin(bb.min_z, fz, d, offsets)], dtype=np.float64)
                    c = counter.count(d, dmin, dmax)
                    best = c if best is None else min(best, c)
        counts.append(best)
        if log:
            log(f"  delta = {d:.6f}: {best} cubes [{len(fr) ** 3} offsets]")
    deltas, counts = np.array(deltas), np.array(counts)
    keep = counts > 0
    if keep.sum() < 3:
        raise ValueError('insufficient non-zero cube counts')
    return deltas[keep], counts[keep]
