"""Scaling-range selection for box-counting dimensions (2D and 3D share this).

1. Boundary removal (ladders with > 8 sizes): fit the first, middle and last
   quarter (>= 3 points each); trim one end point if its slope deviates from
   the middle by > 15 %, or if its correlation test fails.
2. Optimal window: over every contiguous window of >= 5 points, keep the one
   maximising R^2 * span among those with R^2 >= 0.97 and span >= 1 decade.
3. Fallback: fit the whole ladder; NaN if R^2 < 0.97.

LEGACY QUIRK (reproduced on purpose, see TRIM_R_TEST): the end-point test
compares the correlation coefficient r -- not R^2 -- against 0.95.  The 2D
analyzer fits ln(count) vs ln(size), where r ~ -1, so `r < 0.95` is always
true and both end points are always trimmed.  The 3D analyzer fits
log10(count) vs log10(1/size), where r ~ +1, so the test works as intended.
"""
import numpy as np
from scipy import stats

R2_MIN = 0.97
SPAN_MIN = 1.0          # decades (log10)
SLOPE_DEV = 0.15
TRIM_R_MIN = 0.95


def _trim_ends(x, y):
    """Return (start, end) slice bounds after boundary removal on (x, y)."""
    n = len(x)
    seg = max(3, n // 4)
    if n <= 8 or n < 3 * seg:
        return 0, n
    try:
        sl_f, _, r_f, _, _ = stats.linregress(x[:seg], y[:seg])
        sl_m, _, _, _, _ = stats.linregress(x[seg:2 * seg], y[seg:2 * seg])
        sl_l, _, r_l, _, _ = stats.linregress(x[-seg:], y[-seg:])
    except Exception:
        return 0, n
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


def dimension_2d(sizes, counts):
    """Legacy 2D protocol.  -> dict(dimension, error, r_squared, box_sizes, box_counts)
    where box_sizes/box_counts are the fitted window."""
    sizes = np.asarray(sizes, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    s, e = _trim_ends(np.log(sizes), np.log(counts))
    sizes, counts = sizes[s:e], counts[s:e]
    sl, se, r2, a, b = _best_window(np.log10(sizes), np.log10(counts))
    return dict(dimension=-sl, error=se, r_squared=r2, box_sizes=sizes[a:b], box_counts=counts[a:b])


def dimension_3d(deltas, counts):
    """Legacy 3D protocol (log10 of 1/delta).  Same return keys as dimension_2d."""
    deltas = np.asarray(deltas, dtype=np.float64)
    counts = np.asarray(counts)
    x = np.log10(1.0 / deltas)
    y = np.log10(counts)
    s, e = _trim_ends(x, y)
    deltas, counts, x, y = deltas[s:e], counts[s:e], x[s:e], y[s:e]
    sl, se, r2, a, b = _best_window(x, y)
    return dict(dimension=sl, error=se, r_squared=r2, box_sizes=deltas[a:b], box_counts=counts[a:b])
