"""LAMMPS dump files, and Cao's midpoint construction of a mixing interface.

A dump carries atom positions, not an interface.  The interface the Cao
(1992) reanalysis counts is built from them: the midpoints of cross-type
nearest-neighbour pairs, thinned by the angular criterion of that paper's
appendix III so that an atom hidden behind a nearer one of the same
direction contributes nothing.

The raw result is a POINT CLOUD; counting it with `boxcount.points2d`
measures the cloud density (D -> 2), not interface geometry.  To get a
fractal dimension, reduce the cloud to a 1D curve with `reduce_to_curve`
(median y per x-bin), convert to segments with `curve_to_segments`, and
count with `boxcount.segments2d`.
"""
import os
import re
from dataclasses import dataclass

import numpy as np

ITEM = 'ITEM: '
DEFAULT_CUTOFF = 1.8          # sigma, the plan's value for the Cao cases
DEFAULT_ANGLE = 30.0          # degrees, Cao appendix III
DEFAULT_MAX_NEIGHBOURS = 10
DEFAULT_DEDUP = 0.1           # sigma; the two passes produce each midpoint twice


@dataclass
class Frame:
    """One dump frame: the timestep, the box, and the atom table as read."""
    timestep: int
    box_lo: np.ndarray
    box_hi: np.ndarray
    columns: tuple
    data: np.ndarray
    path: str = ''

    @property
    def n_atoms(self):
        return len(self.data)

    def column(self, name):
        """One named column, by the name the dump's own ATOMS line gives it."""
        try:
            return self.data[:, self.columns.index(name)]
        except ValueError:
            raise ValueError(
                f'{self.path or "dump"}: no column {name!r}; it has '
                f'{", ".join(self.columns)}') from None

    def types(self):
        return self.column('type').astype(np.int64)

    def positions(self, dim=2):
        """-> (n, dim) coordinates from the x/y[/z] columns."""
        if dim not in (2, 3):
            raise ValueError(f'dim must be 2 or 3, got {dim}')
        return np.ascontiguousarray(
            np.column_stack([self.column(c) for c in 'xyz'[:dim]]))

    def select(self, types=None, exclude=None):
        """-> a Frame holding only these atom types, in the same order.

        `exclude` removes types instead, which is how the wall atoms (type 3
        in the Cao cases) are kept out of an interface.
        """
        t = self.types()
        keep = np.ones(len(t), dtype=bool)
        if types is not None:
            keep &= np.isin(t, np.atleast_1d(types))
        if exclude is not None:
            keep &= ~np.isin(t, np.atleast_1d(exclude))
        return Frame(self.timestep, self.box_lo, self.box_hi, self.columns,
                     np.ascontiguousarray(self.data[keep]), self.path)


TILT = ('xy', 'xz', 'yz')


def _n_box_lines(item):
    """How many bound lines follow `BOX BOUNDS ...`.

    One per dimension, and the dump says which it has through its per-axis
    boundary flags (`ff ff pp`).  A triclinic box names its tilt factors
    first; those are not axes.  A dump old enough to carry no flags at all
    gets the LAMMPS default of three.
    """
    flags = [w for w in item.split()[2:] if w not in TILT]
    return len(flags) if flags else 3


def _read_frame(fh, path):
    """-> a Frame, or None at end of file.  Leaves fh at the next frame."""
    timestep = n_atoms = None
    lo, hi = [], []
    for line in fh:
        line = line.rstrip('\n')
        if not line.startswith(ITEM):
            continue
        item = line[len(ITEM):].strip()
        if item == 'TIMESTEP':
            timestep = int(next(fh).split()[0])
        elif item == 'NUMBER OF ATOMS':
            n_atoms = int(next(fh).split()[0])
        elif item.startswith('BOX BOUNDS'):
            lo, hi = [], []
            for _ in range(_n_box_lines(item)):
                parts = next(fh).split()
                lo.append(float(parts[0]))
                hi.append(float(parts[1]))
        elif item.startswith('ATOMS'):
            columns = tuple(item.split()[1:])
            if n_atoms is None:
                raise ValueError(f'{path}: ATOMS before NUMBER OF ATOMS')
            data = np.loadtxt(fh, max_rows=n_atoms, ndmin=2)
            if len(data) != n_atoms:
                raise ValueError(f'{path}: the frame at timestep {timestep} declares '
                                 f'{n_atoms} atoms and holds {len(data)}')
            if data.shape[1] != len(columns):
                raise ValueError(f'{path}: {len(columns)} column names and '
                                 f'{data.shape[1]} columns of data')
            return Frame(timestep, np.array(lo), np.array(hi), columns, data, path)
    if timestep is not None:
        raise ValueError(f'{path}: a frame header with no ATOMS section')
    return None


def read_frames(path):
    """Iterate over every frame in a dump file."""
    with open(path) as fh:
        while True:
            frame = _read_frame(fh, path)
            if frame is None:
                return
            yield frame


def read_dump(path):
    """-> the single Frame in this dump file.

    Raises if the file holds several, since taking the first silently is how
    a multi-frame dump gets analysed as one snapshot.  Use `read_frames`.
    """
    it = read_frames(path)
    first = next(it, None)
    if first is None:
        raise ValueError(f'{path}: no dump frames found')
    if next(it, None) is not None:
        raise ValueError(f'{path}: holds more than one frame; use read_frames(path) '
                         f'and say which timestep you mean')
    return first


def timestep_of(path):
    """-> the timestep in a `dump.name.STEP` filename, or None."""
    m = re.search(r'\.(\d+)$', os.path.basename(path))
    return int(m.group(1)) if m else None


def _angular_keep(unit, valid, cos_min):
    """Cao's angular filter, vectorised over reference atoms.

    `unit` is (n, k, d) unit vectors to the k candidate neighbours, ordered by
    increasing distance as a KD-tree returns them.  A neighbour is kept when
    no NEARER kept neighbour lies within the threshold angle of it.  Walking
    outward and testing only against what is already kept is exactly the
    pairwise rule of the original: with the neighbours distance-sorted, the
    farther of any offending pair is always the later one.
    """
    n, k, _ = unit.shape
    keep = np.zeros((n, k), dtype=bool)
    if k == 0:
        return keep
    keep[:, 0] = valid[:, 0]
    for j in range(1, k):
        blocked = np.zeros(n, dtype=bool)
        for i in range(j):
            cos_ij = np.einsum('nd,nd->n', unit[:, i, :], unit[:, j, :])
            blocked |= keep[:, i] & (np.clip(cos_ij, -1.0, 1.0) > cos_min)
        keep[:, j] = valid[:, j] & ~blocked
    return keep


def _pass_midpoints(ref, other, cutoff, angle, max_neighbours):
    """Midpoints from every atom of `ref` to its kept neighbours in `other`."""
    from scipy.spatial import cKDTree

    if len(ref) == 0 or len(other) == 0:
        return np.zeros((0, ref.shape[1] if len(ref) else 2))
    k = min(max_neighbours, len(other))
    dists, idx = cKDTree(other).query(ref, k=k, distance_upper_bound=cutoff,
                                      workers=-1)
    if k == 1:
        dists, idx = dists[:, None], idx[:, None]
    valid = dists < cutoff
    if not valid.any():
        return np.zeros((0, ref.shape[1]))
    safe = np.where(valid, idx, 0)
    neigh = other[safe]                              # (n, k, d)
    vecs = neigh - ref[:, None, :]
    norms = np.linalg.norm(vecs, axis=2)
    norms[norms == 0] = 1e-10
    unit = vecs / norms[:, :, None]
    keep = (_angular_keep(unit, valid, np.cos(np.radians(angle)))
            if (k > 1 and angle > 0) else valid)
    rows, cols = np.nonzero(keep)                    # row-major: atom, then neighbour
    return 0.5 * (ref[rows] + neigh[rows, cols])


def _dedup(points, radius):
    """Drop a point that lies within `radius` of an earlier one."""
    from scipy.spatial import cKDTree

    if len(points) < 2 or radius <= 0:
        return points
    pairs = cKDTree(points).query_pairs(radius)
    if not pairs:
        return points
    keep = np.ones(len(points), dtype=bool)
    keep[[max(i, j) for i, j in pairs]] = False
    return points[keep]


def midpoint_interface(a, b, cutoff=DEFAULT_CUTOFF, angle=DEFAULT_ANGLE,
                       max_neighbours=DEFAULT_MAX_NEIGHBOURS, dedup=DEFAULT_DEDUP):
    """Cao's interface between two species -> (m, d) midpoints.

    Both directions are taken, a to b and b to a, because an atom with no
    cross-type neighbour of its own still lies on the interface as seen from
    the other side.  Each pair then yields its midpoint twice, which is what
    `dedup` removes; it is a coincidence threshold, not a smoothing length.
    """
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f'both species need (n, d) positions of equal d, '
                         f'got {a.shape} and {b.shape}')
    mid = np.vstack([_pass_midpoints(a, b, cutoff, angle, max_neighbours),
                     _pass_midpoints(b, a, cutoff, angle, max_neighbours)])
    return _dedup(mid, dedup)


def extract_interface(frame, types=(1, 2), exclude=(3,), dim=2,
                      cutoff=DEFAULT_CUTOFF, angle=DEFAULT_ANGLE,
                      max_neighbours=DEFAULT_MAX_NEIGHBOURS, dedup=DEFAULT_DEDUP):
    """Cao's midpoint interface between two atom types of one dump frame.

    `exclude` drops the wall atoms before anything else, so a wall of a type
    listed in `types` cannot contribute midpoints.
    """
    if len(types) != 2:
        raise ValueError(f'two species make an interface, got {types}')
    fluid = frame.select(exclude=exclude)
    t = fluid.types()
    pos = fluid.positions(dim)
    a, b = pos[t == types[0]], pos[t == types[1]]
    if len(a) == 0 or len(b) == 0:
        raise ValueError(
            f'{frame.path or "frame"}: timestep {frame.timestep} has '
            f'{len(a)} atoms of type {types[0]} and {len(b)} of type {types[1]}, '
            f'so there is no interface between them. Check --types against the '
            f'types present: {sorted(set(frame.types().tolist()))}.')
    return midpoint_interface(a, b, cutoff, angle, max_neighbours, dedup)


def reduce_to_curve(points, bin_width=None):
    """Bin a 2D midpoint cloud in x, take median y per bin -> (x, y) curve.

    The midpoint cloud has thermal scatter in y that makes it a 2D object
    (D -> 2 under point counting).  Median binning suppresses the scatter
    while preserving the large-scale corrugation, producing a 1D curve whose
    box-counting dimension measures interface geometry.

    Returns (x_curve, y_curve), one point per occupied bin.  Consecutive
    points form a natural polyline whose segments feed `boxcount.segments2d`.

    Parameters
    ----------
    points : (M, 2) array
        Midpoint cloud (x, y).
    bin_width : float or None
        Bin width in x.  None uses twice the median x-spacing (the PRA
        paper's 2 sigma default for the Cao cases).
    """
    x, y = points[:, 0], points[:, 1]
    order = np.argsort(x)
    x_s, y_s = x[order], y[order]

    if bin_width is None:
        dx = np.diff(x_s)
        bin_width = max(2.0 * np.median(dx[dx > 0]), 0.5)

    bins = ((x_s - x_s[0]) / bin_width).astype(int)
    unique_bins = np.unique(bins)
    xc = np.empty(len(unique_bins))
    yc = np.empty(len(unique_bins))
    for i, b in enumerate(unique_bins):
        mask = bins == b
        xc[i] = np.median(x_s[mask])
        yc[i] = np.median(y_s[mask])
    return xc, yc


def curve_to_segments(x, y):
    """Consecutive (x, y) points -> (N-1, 4) segment array [x1, y1, x2, y2]."""
    return np.column_stack([x[:-1], y[:-1], x[1:], y[1:]])


def write_interface(points, path, fmt='%.6f'):
    """Write midpoints as the x,y CSV the point counter reads."""
    header = ','.join('xyz'[:points.shape[1]])
    np.savetxt(path, points, delimiter=',', header=header, fmt=fmt, comments='')
    return path
