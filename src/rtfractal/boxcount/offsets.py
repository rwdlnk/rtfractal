"""Grid-origin offsets for occupied-box counting.

At each box size the grid origin is shifted by these fractions of the box
size along every axis and the MINIMUM occupied count is kept, suppressing
grid-alignment bias (2D: 4/9/16 origins; 3D: 8/27/64).

NOTE (legacy, kept for reproducibility): the thresholds are absolute lengths
in the case's units (metres), not fractions of the domain.
"""
import numpy as np

OFFSET_MODES = ('legacy', 'covering', 'single')


def offset_fractions(box_size, mode='covering'):
    """Fractions of the box size to shift the grid origin by, per axis.

    `single` returns one fraction, so no minimum is taken over anything; the
    other modes return 4, 9 or 16 origins in 2D and the cube of that in 3D.
    """
    if mode == 'single':
        return np.zeros(1)
    if box_size < 0.005:
        return np.linspace(0, 0.75, 4)
    if box_size < 0.02:
        return np.linspace(0, 0.5, 3)
    return np.linspace(0, 0.5, 2)


def origin(lo, frac, box, mode):
    """Grid origin for one offset.

    legacy:   lo + frac*box -- the grid starts INSIDE the bounding box, so the
              part of the set in [lo, lo + frac*box) is never counted and the
              minimum over offsets undercounts.  This inflates D (Koch curve:
              1.35-1.39 instead of 1.26 with FractalAnalyzer's own generator).
              Kept only so legacy results reproduce exactly.
    covering: lo - frac*box -- every grid covers the whole set (default).
    single:   lo -- one grid, aligned on the set's own minimum corner, with no
              minimum taken over offsets at all and no bounding-box margin, so
              it reproduces `floor((x - x.min()) / box)` exactly.  This is what a box counter
              does before anyone thinks about the grid, and it is the
              convention behind most published counts, including the
              Cao-reanalysis MD figure.  It is not biased the way `legacy` is,
              since its grid does cover the set, but it keeps whatever
              alignment luck it happened to draw.  On the MD midpoint clouds
              it sits 0.001-0.005 below `covering`, which is small next to the
              0.078 that the choice of LADDER is worth on the same data; see
              `ladder.py`.
    """
    if mode == 'legacy':
        return lo + frac * box
    if mode == 'covering':
        return lo - frac * box
    if mode == 'single':
        return lo
    raise ValueError(f'offset mode must be one of {OFFSET_MODES}')
