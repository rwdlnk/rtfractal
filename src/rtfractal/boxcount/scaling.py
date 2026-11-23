"""Scaling-range selection for box-counting dimensions (2D and 3D share this).

1. Boundary removal (ladders with > 8 sizes): fit the first, middle and last
   quarter (>= 3 points each); trim one end point if its slope deviates from
   the middle by > 15 %, or if its correlation test fails.
2. Optimal window: over every contiguous window of >= 5 points, keep the one
   maximising R^2 * span among those with R^2 >= 0.97 and span >= 1 decade.
3. Fallback: fit the whole ladder; NaN if R^2 < 0.97.
4. Refusal: NaN if the fitted window holds fewer than MIN_FIT_POINTS points,
   or is shorter than `min_span` decades.  The point count is the guard that
   is always on, because R^2 cannot do that job: two points are collinear by
   construction, so a two-rung ladder reports R^2 = 1.0000 for whatever slope
   its two ends happen to give.  It is set at 3 and not higher on purpose: a
   three-rung fit is weak but it is a fit, and one is quoted -- a factor-1.5
   ladder across the Cao atomic window holds exactly three rungs and is how
   the sensitivity of D to the ladder, about +0.078, was measured.  Measured on Koch level 5 (true D = 1.2619),
   max_box = extent/4: 12 rungs give D = 1.2787 at R^2 = 0.9992 over 1.94
   decades, and 2 rungs give **D = 1.0000 at R^2 = 1.0000** over 0.18.

   `min_span` defaults to 0, so a short window still returns its dimension
   and is flagged rather than refused.  That is a decision taken 2026-11-22,
   after measuring what the strict setting would cost, and it is reversible
   per call with `min_span=SPAN_MIN`:

   - 493 of the 4684 stored RT snapshots fit a window under a decade, and
     they are not noise.  Each of the five `MPLIC-3D/*/160x200x80` cases fits
     its whole time series at 0.70 decades, worst case D = 1.8692 at
     R^2 = 0.9996 over 5 points.  A 3D ladder runs from two cells to L/8,
     which is exactly 1.0 decades at that grid before boundary trimming, so
     the strict setting deletes a paper's coarsest 3D grid rather than
     flagging it.
   - A point set has a hard lower bound at its nearest-neighbour spacing, so
     a short ladder is its normal condition, not a defect.  The Cao atomic
     window, ell = 0.8-2.5 sigma, is 0.49 decades wide by construction.
   - Short is not the same as wrong.  On Minkowski level 3 the sub-decade
     window is the better answer: 0.88 decades gives D = 1.414 against a true
     1.5, while the 1.41-decade ladder, which reaches below the geometry's own
     feature size, gives 1.248.

LEGACY QUIRK (trim='legacy' reproduces it; default trim='r2' fixes it): the end-point test
compares the correlation coefficient r -- not R^2 -- against 0.95.  The 2D
analyzer fits ln(count) vs ln(size), where r ~ -1, so `r < 0.95` is always
true and both end points are always trimmed.  The 3D analyzer fits
log10(count) vs log10(1/size), where r ~ +1, so the test works as intended.
"""
import numpy as np
from scipy import stats

R2_MIN = 0.97
SPAN_MIN = 1.0          # decades (log10)
MIN_FIT_POINTS = 3      # two points are collinear, so R^2 tests nothing at all
SLOPE_DEV = 0.15
TRIM_R_MIN = 0.95


def _trim_ends(x, y, trim='r2'):
    """Return (start, end) slice bounds after boundary removal on (x, y).

    trim='r2' (default) tests R^2 = r^2 < 0.95 as intended; trim='legacy'
    tests the correlation coefficient r < 0.95 (see module note); trim='none'
    removes no end point at all, which is what reproducing a published bare
    polyfit over a fixed ladder requires."""
    n = len(x)
    if trim == 'none':
        return 0, n
    seg = max(3, n // 4)
    if n <= 8 or n < 3 * seg:
        return 0, n
    try:
        sl_f, _, r_f, _, _ = stats.linregress(x[:seg], y[:seg])
        sl_m, _, _, _, _ = stats.linregress(x[seg:2 * seg], y[seg:2 * seg])
        sl_l, _, r_l, _, _ = stats.linregress(x[-seg:], y[-seg:])
    except Exception:
        return 0, n
    if trim == 'r2':
        r_f, r_l = r_f ** 2, r_l ** 2
    elif trim != 'legacy':
        raise ValueError("trim must be 'none', 'legacy' or 'r2'")
    ts = te = 0
    if sl_m != 0:
        if abs(sl_f - sl_m) / abs(sl_m) > SLOPE_DEV or r_f < TRIM_R_MIN:
            ts = 1
        if abs(sl_l - sl_m) / abs(sl_m) > SLOPE_DEV or r_l < TRIM_R_MIN:
            te = 1
    if (ts or te) and n > ts + te + 5:
        return ts, n - te
    return 0, n


def _best_window(x, y):
    """-> (slope, stderr, r2, start, end) of the optimal window, or the
    whole-ladder fallback (slope NaN if its R^2 < R2_MIN)."""
    best = (-1.0, np.nan, np.nan, 0.0, 0, len(x))
    for n_pts in range(min(5, len(x)), len(x) + 1):
        for s in range(len(x) - n_pts + 1):
            e = s + n_pts
            sl, _, rv, _, se = stats.linregress(x[s:e], y[s:e])
            r2 = rv ** 2
            span = abs(x[e - 1] - x[s])
            if r2 >= R2_MIN and span >= SPAN_MIN and r2 * span > best[0]:
                best = (r2 * span, sl, se, r2, s, e)
    if not np.isnan(best[1]):
        return best[1:]
    sl, _, rv, _, se = stats.linregress(x, y)
    return (sl if rv ** 2 >= R2_MIN else np.nan), se, rv ** 2, 0, len(x)


def local_slope(sizes, counts):
    """-> (midpoint sizes, -d log N / d log size) between consecutive rungs.

    A scaling range is where this is flat.  A crossover is where it is not,
    and a straight line fitted across one returns a number set by where the
    ladder stops rather than by the geometry.  The Cao MD atomic window is
    the second kind, falling from about 1.50 to 0.96 across itself as the
    boxes approach the point spacing; the 3-50 sigma window above it is the
    first kind, flat to about +/-0.08 over 1.2 decades.
    """
    sizes = np.asarray(sizes, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if len(sizes) < 2:
        return np.empty(0), np.empty(0)
    ls, lc = np.log(sizes), np.log(counts)
    return np.exp(0.5 * (ls[:-1] + ls[1:])), -np.diff(lc) / np.diff(ls)


def _window_flag(fitted_sizes, r2):
    """-> (span in decades, 'selected' | 'fallback').

    `_best_window` falls back to a whole-ladder regression when no window
    clears R2_MIN and SPAN_MIN, and says nothing about having done so.  A
    deliberately narrow window is a legitimate choice, but it should be
    recorded as one: every fit over the Cao atomic window of 0.49 decades is
    a fallback, and the paper's case for that window has to be made in words
    because the estimator did not make it.
    """
    span = (float(np.log10(fitted_sizes.max() / fitted_sizes.min()))
            if len(fitted_sizes) > 1 else 0.0)
    return span, ('selected' if (r2 >= R2_MIN and span >= SPAN_MIN) else 'fallback')


def _refuse(dim, err, span, flag, n_pts, min_span):
    """-> (dimension, error, window) with a too-short window refused.

    The dimension goes to NaN and the flag to 'refused'; `r_squared`,
    `span_decades` and the fitted ladder are left alone, because the point of
    refusing is to be able to see what was refused.
    """
    if span < min_span or n_pts < MIN_FIT_POINTS:
        return np.nan, np.nan, 'refused'
    return dim, err, flag


def dimension_2d(sizes, counts, trim='r2', min_span=0.0):
    """Legacy 2D protocol.  -> dict(dimension, error, r_squared, box_sizes,
    box_counts, span_decades, window) where box_sizes/box_counts are the
    fitted window and `window` is 'selected', 'fallback' or 'refused'.

    `min_span` is the shortest fitted window, in decades, that may return a
    dimension; below it the dimension is NaN and `window` is 'refused'.  It
    defaults to 0, accepting any window of at least MIN_FIT_POINTS points,
    because a deliberately narrow window is a real case and a short window is
    not necessarily a wrong one (see the module note).  `min_span=SPAN_MIN`
    is the strict setting: cover a decade or return NaN."""
    sizes = np.asarray(sizes, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    s, e = _trim_ends(np.log(sizes), np.log(counts), trim)
    sizes, counts = sizes[s:e], counts[s:e]
    sl, se, r2, a, b = _best_window(np.log10(sizes), np.log10(counts))
    span, flag = _window_flag(sizes[a:b], r2)
    dim, err, flag = _refuse(-sl, se, span, flag, b - a, min_span)
    return dict(dimension=dim, error=err, r_squared=r2, box_sizes=sizes[a:b], box_counts=counts[a:b],
                span_decades=span, window=flag)


def dimension_3d(deltas, counts, trim='r2', min_span=0.0):
    """Legacy 3D protocol (log10 of 1/delta).  Same return keys as dimension_2d,
    and the same `min_span` refusal, defaulting to 0 for a reason this
    protocol feels hardest: a 3D ladder runs from two cells to L/8, so at
    `160x200x80` it spans exactly 1.0 decades before boundary trimming and
    0.70 after, and `min_span=SPAN_MIN` would refuse every snapshot of every
    such case."""
    deltas = np.asarray(deltas, dtype=np.float64)
    counts = np.asarray(counts)
    x = np.log10(1.0 / deltas)
    y = np.log10(counts)
    s, e = _trim_ends(x, y, trim)
    deltas, counts, x, y = deltas[s:e], counts[s:e], x[s:e], y[s:e]
    sl, se, r2, a, b = _best_window(x, y)
    span, flag = _window_flag(deltas[a:b], r2)
    sl, se, flag = _refuse(sl, se, span, flag, b - a, min_span)
    return dict(dimension=sl, error=se, r_squared=r2, box_sizes=deltas[a:b], box_counts=counts[a:b],
                span_decades=span, window=flag)
