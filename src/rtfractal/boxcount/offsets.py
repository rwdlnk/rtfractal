"""Grid-origin offsets for occupied-box counting.

At each box size the grid origin is shifted by these fractions of the box
size along every axis and the MINIMUM occupied count is kept, suppressing
grid-alignment bias (2D: 4/9/16 origins; 3D: 8/27/64).

NOTE (legacy, kept for reproducibility): the thresholds are absolute lengths
in the case's units (metres), not fractions of the domain.
"""
import numpy as np


def offset_fractions(box_size):
    if box_size < 0.005:
        return np.linspace(0, 0.75, 4)
    if box_size < 0.02:
        return np.linspace(0, 0.5, 3)
    return np.linspace(0, 0.5, 2)


OFFSET_MODES = ('legacy', 'covering')


def origin(lo, frac, box, mode):
    """Grid origin for one offset.

    legacy:   lo + frac*box -- the grid starts INSIDE the bounding box, so the
              part of the set in [lo, lo + frac*box) is never counted and the
              minimum over offsets undercounts.  This inflates D (Koch curve:
              1.35-1.39 instead of 1.26 with FractalAnalyzer's own generator).
              Kept as the default only so legacy results reproduce exactly.
    covering: lo - frac*box -- every grid covers the whole set.
    """
    if mode == 'legacy':
        return lo + frac * box
    if mode == 'covering':
        return lo - frac * box
    raise ValueError(f'offset mode must be one of {OFFSET_MODES}')
