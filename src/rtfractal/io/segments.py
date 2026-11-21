"""Read a plain text or CSV file of line segments, or of an ordered polyline.

    segs = read_segments('interface.txt')        # -> (N, 4) [x1, y1, x2, y2]

Four numeric columns per line are taken as segments, two as the vertices of
an **ordered** polyline, which is then differenced into segments.  Blank
lines, and lines beginning # or %, are ignored, as is a single header line
whose first field is not a number.  Whitespace, comma, semicolon and tab all
work as delimiters, mixed freely.

Two columns are ambiguous: an ordered polyline and an unordered point cloud
look identical in a file and must be counted differently.  This module will
not guess.  `kind='polyline'` is the explicit reading; `kind='points'`
raises, because rtfractal has no point-cloud measure yet (`boxcount/` holds
segments2d and triangles3d only).

`suggest_min_box` exists because the smallest box is the one parameter a
bare file cannot supply, and getting it wrong is silent.  It returns a
geometric default and a list of warnings; it does not choose a method.
"""
import re

import numpy as np

__all__ = ['read_segments', 'read_points', 'read_columns', 'suggest_min_box',
           'describe', 'describe_points']

_SPLIT = re.compile(r'[\s,;]+')


def read_columns(path):
    """-> (M, C) float array of the numeric rows of a text file.

    Raises ValueError naming the first offending line if the columns are
    ragged or a value will not parse."""
    rows, ncol, skipped_header = [], None, False
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            s = line.strip()
            if not s or s[0] in '#%':
                continue
            parts = [p for p in _SPLIT.split(s) if p]
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                if not rows and not skipped_header:
                    skipped_header = True        # one header line, silently
                    continue
                raise ValueError(f'{path}:{lineno}: not numeric: {s[:60]!r}')
            if ncol is None:
                ncol = len(vals)
            elif len(vals) != ncol:
                raise ValueError(f'{path}:{lineno}: {len(vals)} columns, '
                                 f'{ncol} on the first data line')
            rows.append(vals)
    if not rows:
        raise ValueError(f'{path}: no numeric data')
    return np.asarray(rows, dtype=np.float64)


def read_segments(path, kind='auto'):
    """-> (N, 4) segments [x1, y1, x2, y2].

    kind='auto'     4 columns -> segments; 2 columns -> error, say which
    kind='segments' require 4 columns (extra columns are ignored)
    kind='polyline' require 2 columns, consecutive vertices joined
    kind='points'   not supported yet; raises with the reason
    """
    if kind == 'points':
        raise ValueError(
            'a point set is not segments. Read it with read_points() and measure it '
            'with fractal.dimension_of_points() or boxcount.points2d, which count '
            'occupied cells rather than clipping a curve against them.')
    if kind not in ('auto', 'segments', 'polyline'):
        raise ValueError("kind must be 'auto', 'segments', 'polyline' or 'points'")

    data = read_columns(path)
    ncol = data.shape[1]

    if kind == 'polyline':
        if ncol < 2:
            raise ValueError(f'{path}: a polyline needs 2 columns, found {ncol}')
        pts = data[:, :2]
        if len(pts) < 2:
            raise ValueError(f'{path}: a polyline needs at least 2 vertices')
        return np.hstack([pts[:-1], pts[1:]])

    if ncol >= 4:
        return np.ascontiguousarray(data[:, :4])

    if ncol == 2:
        raise ValueError(
            f'{path}: 2 columns is ambiguous. An ordered polyline and an '
            'unordered point cloud look the same in a file and are counted '
            "differently. Pass kind='polyline' (--as polyline) if the rows are "
            "in order along a curve, or kind='points' (--as points) if they are "
            'a cloud.')
    raise ValueError(f'{path}: need 4 columns (x1 y1 x2 y2) or 2 '
                     f'(polyline vertices), found {ncol}')


def describe(segs):
    """-> dict of the geometry a reader cannot assume: extent, segment lengths."""
    xs = np.concatenate([segs[:, 0], segs[:, 2]])
    ys = np.concatenate([segs[:, 1], segs[:, 3]])
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return dict(n=len(segs),
                x_min=float(xs.min()), x_max=float(xs.max()),
                y_min=float(ys.min()), y_max=float(ys.max()),
                extent=float(max(xs.max() - xs.min(), ys.max() - ys.min())),
                len_min=float(lengths.min()), len_p10=float(np.percentile(lengths, 10)),
                len_median=float(np.median(lengths)), len_max=float(lengths.max()))


def suggest_min_box(segs, divisor=512):
    """-> (min_box, [warnings]).

    extent/divisor, a geometric choice that asserts nothing about method.
    The warnings are the two ways a bare file goes wrong silently:

    1. a ladder reaching below the data's own resolution.  Below the shortest
       segments the set is a collection of straight lines, which the count
       reads as D = 1 and which pulls the fit down.  Measured on the
       Minkowski sausage at construction level 3: -0.214 against exact on a
       ladder four steps too fine, -0.086 with the smallest box at the
       feature size.
    2. units.  `boxcount/offsets.py:offset_fractions` switches on ABSOLUTE
       box sizes (0.005 and 0.02), so the number of covering offsets, and
       therefore the bias, depends on whether the file is in metres or in
       Lennard-Jones sigma.  A known issue, recorded and not yet changed.
    """
    d = describe(segs)
    min_box = d['extent'] / float(divisor)
    warn = []
    if min_box < d['len_p10']:
        warn.append(
            'smallest box %.4g is below the 10th-percentile segment length %.4g: '
            'the ladder reaches below the data\'s own resolution and D will read low. '
            'Consider --min-box %.4g.' % (min_box, d['len_p10'], d['len_p10']))
    if d['extent'] > 1.0 or d['extent'] < 0.01:
        warn.append(
            'extent is %.4g. The covering-offset count switches on absolute box '
            'sizes (0.005, 0.02), so results are not scale invariant; compare only '
            'with runs at a similar extent, or rescale to match.' % d['extent'])
    return min_box, warn


def read_points(path):
    """-> (M, 2) point set from the first two numeric columns of a text file.

    Unlike a polyline, the order of the rows does not matter and is not used.
    """
    data = read_columns(path)
    if data.shape[1] < 2:
        raise ValueError(f'{path}: a point set needs 2 columns (x y), '
                         f'found {data.shape[1]}')
    pts = np.ascontiguousarray(data[:, :2])
    if len(pts) < 2:
        raise ValueError(f'{path}: a point set needs at least 2 points')
    return pts


def describe_points(pts):
    """-> dict of the geometry a point ladder needs: extent and spacing."""
    from ..boxcount import points2d

    return dict(n=len(pts),
                x_min=float(pts[:, 0].min()), x_max=float(pts[:, 0].max()),
                y_min=float(pts[:, 1].min()), y_max=float(pts[:, 1].max()),
                extent=float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))),
                spacing=points2d.nn_spacing(pts),
                spacing_min=points2d.nn_spacing(pts, 'min'),
                spacing_mean=points2d.nn_spacing(pts, 'mean'))
