"""Interface extraction from 2D volume-fraction fields.

Fields follow the legacy layout: f, x, y are (nx, ny) with x[:, 0] the cell-
centre x coordinates and y[0, :] the cell-centre y coordinates.
"""
import numpy as np
from skimage import measure


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
