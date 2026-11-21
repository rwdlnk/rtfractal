"""Box-size ladders.

Three ways to lay the boxes out, kept separate because the choice is not
neutral and has already been mistaken for an estimator difference.  On the
Cao MD midpoint clouds, over one fixed window of 0.8-2.5 sigma, the
published ten-size ladder reads D = 1.359 and a factor-1.5 ladder reads
1.437.  The grid-origin convention over the same two counts is worth 0.005.
The ladder was worth fifteen times as much as the convention, because that
window is a crossover rather than a scaling range and a ladder that stops
higher up reports the steeper part of the bend.  `scaling.local_slope` is
the test for which of the two a window is.
"""
import numpy as np


def box_ladder(min_box=None, max_box=None, factor=1.5, num_sizes=None, sizes=None):
    """-> box sizes, largest first.

    sizes:     an explicit ladder, taken as given (sorted descending).
               min_box and max_box are then ignored.
    num_sizes: exactly this many sizes, geometrically spaced, with max_box and
               min_box as the two end points.  This is how a fixed window with
               a fixed number of boxes is specified.  A factor ladder cannot
               express it, because it hits neither end exactly unless the
               factor happens to divide the window.
    neither:   start at max_box and divide by `factor` while >= min_box, the
               default everywhere else in rtfractal.
    """
    if sizes is not None and num_sizes is not None:
        raise ValueError('pass sizes or num_sizes, not both')
    if sizes is not None:
        out = np.sort(np.asarray(sizes, dtype=np.float64))[::-1]
        if len(out) < 2:
            raise ValueError(f'an explicit ladder needs at least 2 sizes, got {len(out)}')
        if out.min() <= 0:
            raise ValueError('box sizes must be positive')
        return np.ascontiguousarray(out)
    if min_box is None or max_box is None:
        raise ValueError('min_box and max_box are required unless sizes is given')
    if min_box <= 0:
        raise ValueError(f'min_box must be positive, got {min_box}')
    if max_box < min_box:
        raise ValueError(f'max_box {max_box:.6g} is below min_box {min_box:.6g}')
    if num_sizes is not None:
        if num_sizes < 2:
            raise ValueError(f'num_sizes must be at least 2, got {num_sizes}')
        return np.exp(np.linspace(np.log(max_box), np.log(min_box), int(num_sizes)))
    if factor <= 1.0:
        raise ValueError(f'factor must exceed 1, got {factor}')
    out, box = [], float(max_box)
    while box >= min_box:
        out.append(box)
        box /= factor
    return np.array(out, dtype=np.float64)
