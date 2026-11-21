"""Theoretical fractals with known dimensions, for validating the estimator.

Each generator returns segments in the package convention, an (N, 4) float64
array of rows [x1, y1, x2, y2], so they feed `box_counting_2d` and
`fractal.dimension_of_segments` directly.

Ported 2026-11-21 from the pre-unification generator package, which survived
only inside archived trees
(`FractalParameterAI/legacy/fractal_analyzer_pkg/fractal_analyzer/generators/`).
Before the port this module's five shapes existed here as a single Koch
helper, duplicated byte-for-byte in two test files, so four of the five
validations behind the 2025 box-counting article could not be reproduced
against the code that now produces the numbers.

`koch` reproduces that former test helper exactly, including its default
placement, so the tests written against it are unchanged by the port.

    from rtfractal import shapes
    segs = shapes.koch(7)
    shapes.SHAPES['sierpinski'](5)
    shapes.DIMENSION['koch']          # log 4 / log 3
"""
import numpy as np

__all__ = ['koch', 'sierpinski', 'minkowski', 'hilbert', 'dragon',
           'SHAPES', 'DIMENSION']

# Exact Hausdorff dimensions.  hilbert and dragon are plane-filling, so a
# finite level approaches 2 from below and a box count over a finite ladder
# will read low; see tests/test_shapes.py for the measured values.
DIMENSION = {
    'koch': np.log(4) / np.log(3),        # 1.2619
    'sierpinski': np.log(3) / np.log(2),  # 1.5850
    'minkowski': np.log(8) / np.log(4),   # 1.5
    'hilbert': 2.0,
    'dragon': 2.0,
}


def _as_segments(pts):
    """(M, 2) vertices of an open polyline -> (M-1, 4) segments."""
    pts = np.asarray(pts, dtype=np.float64)
    return np.hstack([pts[:-1], pts[1:]])


def _place(pts, L, origin):
    """Scale a shape spanning its own bounding box into a box of width L."""
    pts = np.asarray(pts, dtype=np.float64)
    span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
    if span > 0:
        pts = (pts - pts.min(axis=0)) * (L / span)
    return pts + np.asarray(origin, dtype=np.float64)


def koch(level, L=0.4, y0=0.25):
    """Koch curve, D = log 4 / log 3.

    Identical to the helper this replaced: a horizontal base from (0, y0) to
    (L, y0), each segment replaced by four with the peak rotated +60 degrees."""
    pts = np.array([[0.0, y0], [L, y0]])
    rot = np.array([[0.5, -np.sqrt(3) / 2], [np.sqrt(3) / 2, 0.5]])
    for _ in range(level):
        new = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            d = (b - a) / 3
            new += [a + d, a + d + rot @ d, a + 2 * d, b]
        pts = np.array(new)
    return _as_segments(pts)


def sierpinski(level, L=0.4, origin=(0.0, 0.0)):
    """Sierpinski gasket as triangle outlines, D = log 3 / log 2.

    Unlike the others this is not one polyline but a set of closed triangles,
    so it is assembled as segments rather than vertices."""
    def tri(p1, p2, p3, n):
        if n == 0:
            return [(p1, p2), (p2, p3), (p3, p1)]
        m1 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        m2 = ((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2)
        m3 = ((p3[0] + p1[0]) / 2, (p3[1] + p1[1]) / 2)
        return (tri(p1, m1, m3, n - 1) + tri(m1, p2, m2, n - 1)
                + tri(m3, m2, p3, n - 1))

    segs = tri((0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3) / 2), level)
    arr = np.array([[a[0], a[1], b[0], b[1]] for a, b in segs], dtype=np.float64)
    # place both endpoints with one transform, so segments stay connected
    ends = _place(np.vstack([arr[:, :2], arr[:, 2:]]), L, origin)
    n = len(arr)
    return np.hstack([ends[:n], ends[n:]])


def minkowski(level, L=0.4, origin=(0.0, 0.25)):
    """Minkowski sausage, D = log 8 / log 4 = 3/2.

    Each segment is replaced by eight of a quarter the length."""
    def rec(x1, y1, x2, y2, n):
        if n == 0:
            return [(x1, y1), (x2, y2)]
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        q = length / 4.0
        p = [(x1, y1)]
        for sx, sy in ((ux, uy), (-uy, ux), (ux, uy), (uy, -ux),
                       (ux, uy), (-uy, ux), (ux, uy)):
            p.append((p[-1][0] + sx * q, p[-1][1] + sy * q))
        p.append((x2, y2))
        out = []
        for a, b in zip(p[:-1], p[1:]):
            out.extend(rec(a[0], a[1], b[0], b[1], n - 1)[:-1])
        out.append((x2, y2))
        return out

    return _as_segments(_place(rec(0.0, 0.0, 1.0, 0.0, level), L, origin))


def hilbert(level, L=0.4, origin=(0.0, 0.0)):
    """Hilbert curve.  Plane-filling in the limit, so D -> 2 from below."""
    pts = []

    def rec(x0, y0, xi, xj, yi, yj, n):
        if n <= 0:
            pts.append((x0 + (xi + yi) / 2, y0 + (xj + yj) / 2))
            return
        rec(x0, y0, yi / 2, yj / 2, xi / 2, xj / 2, n - 1)
        rec(x0 + xi / 2, y0 + xj / 2, xi / 2, xj / 2, yi / 2, yj / 2, n - 1)
        rec(x0 + xi / 2 + yi / 2, y0 + xj / 2 + yj / 2,
            xi / 2, xj / 2, yi / 2, yj / 2, n - 1)
        rec(x0 + xi / 2 + yi, y0 + xj / 2 + yj,
            -yi / 2, -yj / 2, -xi / 2, -xj / 2, n - 1)

    rec(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, level)
    return _as_segments(_place(pts, L, origin))


def dragon(level, L=0.4, origin=(0.0, 0.0)):
    """Heighway dragon curve.  Plane-filling in the limit, so D -> 2.

    Turn sequence: S(n) = S(n-1), R, reverse(flip(S(n-1)))."""
    seq = [1]
    for _ in range(level):
        seq = seq + [1] + [0 if t == 1 else 1 for t in reversed(seq)]

    step = ((1, 0), (0, 1), (-1, 0), (0, -1))
    x = y = 0
    d = 0
    pts = [(0.0, 0.0)]
    for turn in seq:
        d = (d + 1) % 4 if turn == 1 else (d - 1) % 4
        x += step[d][0]
        y += step[d][1]
        pts.append((float(x), float(y)))
    return _as_segments(_place(pts, L, origin))


SHAPES = {
    'koch': koch,
    'sierpinski': sierpinski,
    'minkowski': minkowski,
    'hilbert': hilbert,
    'dragon': dragon,
}
