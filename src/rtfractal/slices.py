"""Plane sections of a three-dimensional interface: the Marstrand slice test.

Marstrand's theorem is a statement about ONE set and ITS OWN plane sections:
if a set of dimension D > 1 is isotropic, almost every plane section has
dimension D - 1.  Testing it therefore requires slicing the three-dimensional
field and counting the sections, which is what this module does.

It does NOT mean differencing the dimension of a three-dimensional interface
from that of a separate two-dimensional simulation.  Those are two different
sets produced by two different flows, and their difference being near one is
an empirical observation rather than an instance of the theorem.  Both numbers
are worth having, and this module keeps them apart: `slice_dimension` measures
sections of the 3D field, and the comparison against a 2D run is left to the
caller, who must say which it is reporting.

Three design decisions are fixed here deliberately, because getting any of
them wrong produces a number that looks like physics and is not.

1.  **Count each plane separately, then aggregate.**  Never pool the segments
    of several planes into one count.  A pooled set is not a section of
    anything -- it is a superposition of sections, and it counts higher
    because segments from different planes occupy boxes independently.  The
    spread across planes is not noise to be averaged away either: it is the
    measurement of how much the section dimension depends on where you cut.

2.  **Gravity makes the vertical special, so only vertical planes are swept.**
    A Rayleigh-Taylor interface is not isotropic between a vertical and a
    horizontal cut, so "almost every orientation" cannot be taken literally
    for this flow.  The sub-family the symmetry does permit is the azimuthal
    one: vertical planes rotated about the vertical axis.  A flat D(theta)
    over that family measures horizontal isotropy, which the laboratory
    experiments assume and never measure.

3.  **Every section in a sweep gets the same extent, hence the same ladder.**
    The largest box is a fixed fraction of the section's horizontal extent, so
    an extent that varies with angle would move the fit range with the angle
    and manufacture a dependence on orientation out of nothing.  See
    `inscribed_chord` for how that is avoided.

The interpolation trap, which is the reason `theta` is reported alongside
every row rather than averaged over: a vertical plane at theta = 0 or 90
degrees lands exactly on grid lines and is sampled with no interpolation at
all, while every other angle must be interpolated and is therefore slightly
smoothed.  Smoothing lowers D, and the smoothing is largest near 45 degrees,
so D(theta) acquires a dip keyed to grid alignment that mimics anisotropy.
This is the same hazard as the box-counting offset bias: a numerical
convention producing a trend that looks physical.  Calibrate it by running
the same sweep at an early time when the interface is still nearly flat and
D is known to be about 1; whatever theta-dependence appears there is
artefact, and it bounds what can be read from the late-time sweep.
"""
import numpy as np

from . import fractal, interface

__all__ = ['GRID_ALIGNED', 'is_exact', 'inscribed_chord', 'plane_section', 'slice_dimension',
           'slice_sweep', 'surface_plane_section', 'surface_slice_dimension']

# Angles at which a vertical plane lies on grid lines and needs no
# interpolation: theta = 0 is a plane of constant z, theta = 90 one of
# constant x.  Sections at these two angles are exact.
GRID_ALIGNED = (0.0, 90.0)


def inscribed_chord(grid, n_offsets, margin=0.5):
    """Chord length and plane offsets that are the same at every azimuth.

    The horizontal footprint of these runs is a rectangle (0.4 x 0.2 m for the
    Dalziel configuration), so the chord of a vertical plane through the box
    depends on the azimuth -- 0.4 m end to end, 0.2 m across.  Since the
    largest box is extent/8, that would tie the fit range to the orientation.

    The fix is to work inside the inscribed CIRCLE of the horizontal
    footprint, radius R = min(Lx, Lz)/2.  A chord of a circle at distance |p|
    from the centre has length 2*sqrt(R^2 - p^2), which depends on p but not
    on the angle.  Taking the offsets from a bounded range and then sampling
    every section over the SHORTEST of those chords makes the extent -- and so
    the ladder -- identical for every (theta, offset) pair in the sweep.

    `margin` sets how far out the offsets reach as a fraction of R; the
    default of 0.5 keeps the chord at 87% of the diameter.  Returns
    (chord, offsets).
    """
    Lx = grid.nx * grid.dx
    Lz = grid.nz * grid.dz
    R = 0.5 * min(Lx, Lz)
    p_max = margin * R
    offsets = np.linspace(-p_max, p_max, n_offsets) if n_offsets > 1 else np.zeros(1)
    chord = 2.0 * np.sqrt(R * R - p_max * p_max)   # the shortest chord in use
    return float(chord), offsets


def plane_section(alpha, grid, theta_deg, offset, chord=None):
    """Sample one vertical plane of `alpha` -> (f, x, y) for `interface.contours`.

    `alpha` is the (nz, ny, nx) volume fraction; y is the vertical axis.  The
    plane contains the vertical and the horizontal direction at azimuth
    `theta_deg` measured from +x, and sits at signed distance `offset` from
    the horizontal centre along the plane normal.  `chord` is the horizontal
    extent sampled, centred on that point; None takes the full grid extent,
    which is only meaningful at a grid-aligned angle.

    Returns `f` of shape (ns, ny) with `x` the in-plane horizontal coordinate
    and `y` the vertical, both as (ns, ny) meshes, matching the layout that
    `interface.contours` expects.

    At theta = 0 and 90 degrees the plane is taken straight off the grid, so
    the section is exact.  At any other angle the field is interpolated
    bilinearly in (x, z) at each exact y level -- the vertical never needs
    interpolating because the plane contains it.
    """
    nz, ny, nx = alpha.shape
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    yc = (np.arange(ny) + 0.5) * dy
    th = float(theta_deg) % 180.0

    # --- exact, grid-aligned sections ------------------------------------
    # Rounding to the nearest cell is honest here: the offset only names
    # which plane of cells to take, and taking a real one beats interpolating
    # between two of them.
    if chord is None and th in GRID_ALIGNED:
        if th == 0.0:                      # constant z; the plane is (x, y)
            k = int(round((0.5 * nz * dz + offset) / dz - 0.5))
            f = np.ascontiguousarray(alpha[np.clip(k, 0, nz - 1)].T)
            sc = (np.arange(nx) + 0.5) * dx
        else:                              # constant x; the plane is (z, y)
            i = int(round((0.5 * nx * dx + offset) / dx - 0.5))
            f = np.ascontiguousarray(alpha[:, :, np.clip(i, 0, nx - 1)])
            sc = (np.arange(nz) + 0.5) * dz
        x = sc[:, None] * np.ones((1, ny))
        y = np.ones((len(sc), 1)) * yc[None, :]
        return f, x, y

    # --- general azimuth: interpolate onto the plane ----------------------
    from scipy.ndimage import map_coordinates

    if chord is None:
        chord = min(nx * dx, nz * dz)
    ds = min(dx, dz)                        # keep the in-plane sampling at the
    ns = int(round(chord / ds))             # mesh scale so 2*ds stays the floor
    sc = (np.arange(ns) + 0.5) * ds
    s = sc - 0.5 * chord                    # in-plane horizontal, centred on 0

    c, sn = np.cos(np.radians(th)), np.sin(np.radians(th))
    # u = (cos, sin) along the plane, n = (-sin, cos) is its normal.
    xq = 0.5 * nx * dx + s * c - offset * sn
    zq = 0.5 * nz * dz + s * sn + offset * c

    # map_coordinates indexes alpha[z, y, x]; cell centres sit at (i+0.5)*d.
    zi = np.repeat((zq / dz - 0.5)[:, None], ny, axis=1)
    xi = np.repeat((xq / dx - 0.5)[:, None], ny, axis=1)
    yi = np.repeat(np.arange(ny, dtype=float)[None, :], ns, axis=0)
    f = map_coordinates(alpha, [zi, yi, xi], order=1, mode='nearest')

    x = sc[:, None] * np.ones((1, ny))
    y = np.ones((ns, 1)) * yc[None, :]
    return f, x, y


def is_exact(theta_deg, chord):
    """True when `plane_section` will take the plane straight off the grid.

    The fast path needs BOTH a grid-aligned azimuth and a full-extent section:
    asking for a trimmed chord routes even theta = 0 through the interpolator.
    Reporting this from the angle alone was a bug -- in a common-extent sweep
    every angle is interpolated, grid-aligned ones included.

    That is not a defect of such a sweep.  It is the reason one is comparable:
    when every section is interpolated the same way, the smoothing penalty is
    common to all of them and cancels in a comparison ACROSS angles.  The
    penalty only matters when an interpolated section is compared against an
    exact one.  Measured on developed structure at the worst case, a half-cell
    offset, it is -0.008 +/- 0.001 in D; see the module docstring.
    """
    return chord is None and float(theta_deg) % 180.0 in GRID_ALIGNED


def slice_dimension(alpha, grid, theta_deg, offset, chord=None, level=0.5,
                    max_box=None, min_box=None, factor=1.5, backend=None,
                    offsets='covering', trim='r2'):
    """Box-counting dimension of ONE plane section.

    Conventions default to the paper's: covering offsets, a largest box of
    one eighth of the section's horizontal extent, a smallest box of twice the
    mesh spacing, and the R^2 end-point trim.  The ladder is returned with the
    result because a dimension quoted without its scaling range is not a
    well-posed number, and sections taken at different extents are not
    comparable without it.
    """
    f, x, y = plane_section(alpha, grid, theta_deg, offset, chord)
    extent = float(x[-1, 0] - x[0, 0])
    if max_box is None:
        max_box = extent / 8.0
    if min_box is None:
        min_box = 2.0 * min(grid.dx, grid.dz)

    segs = interface.segments(interface.contours(f, x, y, level))
    if len(segs) == 0:
        return dict(theta=float(theta_deg), offset=float(offset), n_segments=0,
                    dimension=np.nan, error=np.nan, r_squared=np.nan,
                    max_box=max_box, min_box=min_box, extent=extent,
                    exact=is_exact(theta_deg, chord))

    r = fractal.dimension_of_segments(segs, min_box, max_box, factor, backend,
                                      offsets, trim)
    return dict(theta=float(theta_deg), offset=float(offset), n_segments=len(segs),
                dimension=float(r['dimension']), error=float(r['error']),
                r_squared=float(r['r_squared']),
                max_box=max_box, min_box=min_box, extent=extent,
                exact=is_exact(theta_deg, chord))


def slice_sweep(alpha, grid, thetas=(0.0, 90.0), n_offsets=9, margin=0.5,
                common_extent=True, **kw):
    """Every (azimuth, offset) section of one snapshot, as a list of rows.

    With `common_extent` (the default) all sections are sampled over the same
    chord through the inscribed circle, so the ladder is identical across the
    sweep and a change in D with theta cannot be an artefact of the fit range.
    Turn it off only to take full grid-aligned planes, which is what the
    comparison against a two-dimensional run wants -- there the section should
    match the 2D domain exactly rather than be trimmed to a circle.

    Rows carry `exact`, true where the plane needed no interpolation.  Compare
    like with like: the two grid-aligned angles are the trustworthy pair, and
    the interpolated angles between them are only as good as the calibration
    described in this module's docstring.
    """
    if common_extent:
        chord, offs = inscribed_chord(grid, n_offsets, margin)
    else:
        chord, offs = None, (np.linspace(-0.25, 0.25, n_offsets) *
                             min(grid.nx * grid.dx, grid.nz * grid.dz))
    rows = []
    for th in thetas:
        for p in offs:
            rows.append(slice_dimension(alpha, grid, th, float(p), chord, **kw))
    return rows


# ---------------------------------------------------------------------------
# Sections of an EXTRACTED SURFACE rather than of the field.
#
# Everything above cuts the volume-fraction field and re-contours it inside the
# plane.  These two cut the triangulated iso-surface the solver wrote out, and
# they exist because the codimension test must not mix the two
# representations: `fractal.dimension_3d` counts that same triangulated
# surface, so a section to difference against it has to come off the surface
# too.  `interface`'s module docstring has the measurements -- field sections
# read 0.061 higher on average, and the test holds on the surface throughout
# (0.998 +/- 0.016 over six cases) while it fails on the field throughout
# (1.100).
#
# The intersection is computed directly from the triangles, not through vtk,
# because vtk is an optional dependency here and a plane-triangle crossing is
# four lines of arithmetic.  A triangle whose three signed vertex distances to
# the plane are not all of one sign is crossed, and the two edges carrying the
# sign change give the end points of its segment.
#
# ONE CONVENTION TO KNOW: `plane_section` above places the box centre at
# 0.5 * nx * dx, ignoring `grid.x_min`, because it indexes the field array.
# These functions work in the mesh's own physical coordinates and so add the
# grid minima.  The two agree whenever the grid origin is zero, which is every
# case in these papers, and differ by the origin otherwise.  Pass the same
# (theta_deg, offset) to both and the planes coincide.


def surface_plane_section(mesh, grid, theta_deg, offset, chord=None):
    """Segments where a vertical plane cuts a triangulated surface.

    `mesh` is an `io.mesh.TriangleMesh`, `grid` its `Grid3D`.  The plane
    contains the vertical and the horizontal direction at azimuth `theta_deg`
    from +x, and sits at signed distance `offset` from the horizontal centre
    along its own normal, matching `plane_section`.

    Returns (segs, extent): `segs` is (n, 4) float64 of (s1, y1, s2, y2) with
    `s` the in-plane horizontal coordinate and `y` the vertical, ready for
    `fractal.dimension_of_segments`; `extent` is the horizontal extent the
    ladder should be built on.
    """
    import numpy as _np
    th = _np.radians(float(theta_deg) % 180.0)
    c, sn = _np.cos(th), _np.sin(th)
    u = _np.array([c, 0.0, sn])            # in-plane horizontal
    n = _np.array([-sn, 0.0, c])           # plane normal
    cx = grid.x_min + 0.5 * grid.nx * grid.dx
    cz = grid.z_min + 0.5 * grid.nz * grid.dz
    p0 = _np.array([cx, 0.0, cz]) + float(offset) * n

    tv = mesh.triangle_vertices()                       # (M, 3, 3)
    d = _np.einsum('mkj,j->mk', tv - p0, n)             # signed distances
    # A vertex exactly on the plane is counted as positive, so a triangle that
    # merely touches the plane at a vertex is not crossed while one straddling
    # it through a vertex still yields two crossings (one of them the vertex
    # itself, at t = 0).  A strict `(da * db) < 0` test instead drops both,
    # which silently empties any plane that coincides with the mesh: every
    # iso-surface vertex sits on a cell edge, so the centre plane of a
    # cell-aligned grid is exactly such a plane, and it returned no section at
    # all until this was fixed.
    pos = d >= 0.0
    crossed = pos.any(axis=1) & (~pos).any(axis=1)
    tv, d, pos = tv[crossed], d[crossed], pos[crossed]
    if len(tv) == 0:
        segs = _np.empty((0, 4))
    else:
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            da, db = d[:, a], d[:, b]
            hit = pos[:, a] != pos[:, b]
            den = da - db
            t = _np.where(hit, da / _np.where(den == 0.0, 1.0, den), 0.0)
            pts.append((hit, tv[:, a] + t[:, None] * (tv[:, b] - tv[:, a])))
        # every crossed triangle contributes exactly two edge crossings
        out = _np.full((len(tv), 2, 3), _np.nan)
        slot = _np.zeros(len(tv), dtype=int)
        for hit, pt in pts:
            idx = _np.flatnonzero(hit)
            out[idx, slot[idx]] = pt[idx]
            slot[idx] += 1
        ok = slot == 2
        out = out[ok]
        s_coord = _np.einsum('mkj,j->mk', out, u)
        y_coord = out[:, :, 1]
        segs = _np.ascontiguousarray(
            _np.column_stack([s_coord[:, 0], y_coord[:, 0], s_coord[:, 1], y_coord[:, 1]]),
            dtype=_np.float64)

    if chord is not None:
        extent = float(chord)
    elif float(theta_deg) % 180.0 == 0.0:
        extent = (grid.nx - 1) * grid.dx
    elif float(theta_deg) % 180.0 == 90.0:
        extent = (grid.nz - 1) * grid.dz
    else:
        extent = inscribed_chord(grid, 1)[0]
    return segs, extent


def surface_slice_dimension(mesh, grid, theta_deg, offset, chord=None, max_box=None,
                            min_box=None, factor=1.5, backend=None, offsets='covering',
                            trim='r2'):
    """Box-counting dimension of ONE plane section of a triangulated surface.

    The surface-representation counterpart of `slice_dimension`, with the same
    defaults (covering offsets, largest box one eighth of the section's
    horizontal extent, smallest twice the mesh spacing, R^2 end-point trim), so
    a codimension taken against `fractal.dimension_3d` on the same mesh crosses
    nothing.
    """
    import numpy as _np
    from . import fractal
    segs, extent = surface_plane_section(mesh, grid, theta_deg, offset, chord)
    if max_box is None:
        max_box = extent / 8.0
    if min_box is None:
        min_box = 2.0 * min(grid.dx, grid.dz)
    if len(segs) == 0:
        return dict(theta=float(theta_deg), offset=float(offset), n_segments=0,
                    dimension=_np.nan, error=_np.nan, r_squared=_np.nan,
                    max_box=max_box, min_box=min_box, extent=extent)
    r = fractal.dimension_of_segments(segs, min_box, max_box, factor, backend, offsets, trim)
    return dict(theta=float(theta_deg), offset=float(offset), n_segments=len(segs),
                dimension=float(r['dimension']), error=float(r['error']),
                r_squared=float(r['r_squared']),
                max_box=max_box, min_box=min_box, extent=extent)
