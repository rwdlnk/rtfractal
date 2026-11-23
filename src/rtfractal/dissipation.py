"""Dissipation, enstrophy and resolved-scale diagnostics (phase 3).

Ported from paper 1's ``scripts_paper1_revision/dissip.py`` (2D OpenFOAM),
``dissip3d.py`` (3D OpenFOAM) and ``dissip_svof.py`` (SOLA-VOF VTK), which
are one kernel behind three readers.  The arithmetic is kept operation for
operation, so a ported run reproduces the stored CSVs bit for bit.

    eps = 2 nu S_ij S_ij,   S = sym(grad u),   nu = a*nu_heavy + (1-a)*nu_light
    enstrophy = |omega|^2
    eta = (nu_mix^3 / eps_mix)^(1/4)

Averages are over the MIXING ZONE: the rows where the horizontally-averaged
volume fraction satisfies 0.01 <= abar <= 0.99, with a one-cell rim excluded
in every direction so no one-sided difference from np.gradient enters a mean.
`eps_dom` is the same dissipation averaged over the whole domain core instead,
which is what makes the mixing-zone enhancement readable.

Two things this module does NOT do, both deliberate:

- It does not orient the volume fraction.  `a` must already be the heavy
  phase, as OpenFOAM's alpha.water is; SOLA-VOF's F is the other way round in
  some runs and its reader flips it (dissip_svof.py did this by comparing the
  two half-domain means).  Getting it backwards swaps nu_heavy and nu_light,
  which moves eps by under 4% here and would be easy to miss.
- It does not decide which snapshots are worth measuring.  eta is reported
  against dx by the caller; whether eta/dx says the run resolves the
  dissipation scale is a judgement about the ladder, not about one snapshot.
"""
import numpy as np

EPS_LO, EPS_HI = 0.01, 0.99     # mixing-zone bounds on the mean profile
HINT_PREFACTOR = 6.0


def mixing_rows(abar, ny):
    """(j0, j1) bounding the mixing zone, or None if there is not one.

    j0 and j1 are clamped one cell inside the domain, so the mixing-zone
    slice never includes a row whose gradient is one-sided.  Fewer than three
    rows in the band is 'no mixing zone', not an error: it is what an
    undisturbed interface looks like, and the caller skips the snapshot.
    """
    rows = np.where((abar >= EPS_LO) & (abar <= EPS_HI))[0]
    if rows.size < 3:
        return None
    return max(rows[0], 1), min(rows[-1] + 1, ny - 1)


def integral_width(abar, dy):
    """Integral mixing width from the MEAN profile.

    The prefactor 6 is the linear-ramp calibration: a mean profile ramping
    0 -> 1 over width W returns exactly W.  This uses abar*(1-abar), the
    product of the means, NOT the mean of the product, so it is a pure
    profile-transition width with no sensitivity to the molecular-mixing
    fraction Theta (which varies 0.12-0.22 with grid and scheme here and
    would otherwise bias any matched-width comparison).

    NOT the same functional as the pipeline's h_10 or h_total_int.  Do not
    label it h_{1,0}.
    """
    return HINT_PREFACTOR * float(np.sum(abar * (1.0 - abar))) * dy


def _close(nu, eps, ens, comps, mz, core, j0, j1, dx, dy, abar):
    """The reductions, shared by the 2D and 3D kernels.

    `comps` is the velocity components themselves, not a precomputed |u|^2:
    the ported scripts slice first and square the slice, and summing a
    contiguous slice is not the same floating-point operation order as
    slicing a whole-domain sum, so doing it the other way costs bit-identity
    (and, at 40M cells, three full-size temporaries).
    """
    e_mix = float(eps[mz].mean())
    nu_mix = float(nu[mz].mean())
    urms = float(np.sqrt(sum(c[mz] ** 2 for c in comps).mean()))
    hint = integral_width(abar, dy)
    eta = (nu_mix ** 3 / e_mix) ** 0.25 if e_mix > 0 else np.nan
    return dict(h99=(j1 - j0) * dy, hint=hint, urms=urms, nu_mix=nu_mix,
                Re_h=urms * hint / nu_mix, eps_mix=e_mix,
                eps_dom=float(eps[core].mean()), ens_mix=float(ens[mz].mean()),
                eta=eta, eta_over_dx=eta / dx, Re_cell=urms * dx / nu_mix,
                j0=j0, j1=j1)


def dissipation_stats_2d(u, v, a, dx, dy, nu_heavy, nu_light):
    """eps, enstrophy and eta for a 2D snapshot.  All fields are (ny, nx).

    `a` is the HEAVY-phase volume fraction (see the module docstring).
    Returns None when the snapshot has no mixing zone.
    """
    for name, arr in (('v', v), ('a', a)):
        if arr.shape != u.shape:
            raise ValueError(f'{name} shape {arr.shape} != u shape {u.shape}')
    ny = u.shape[0]
    nu = a * nu_heavy + (1.0 - a) * nu_light

    duy, dux = np.gradient(u, dy, dx)
    dvy, dvx = np.gradient(v, dy, dx)
    sxy = 0.5 * (duy + dvx)
    eps = 2.0 * nu * (dux ** 2 + dvy ** 2 + 2.0 * sxy ** 2)
    ens = (dvx - duy) ** 2

    abar = a.mean(axis=1)
    jj = mixing_rows(abar, ny)
    if jj is None:
        return None
    j0, j1 = jj
    mz = (slice(j0, j1), slice(1, -1))
    core = (slice(1, -1), slice(1, -1))
    return _close(nu, eps, ens, (u, v), mz, core, j0, j1, dx, dy, abar)


def dissipation_stats_3d(u, v, w, a, dx, dy, dz, nu_heavy, nu_light):
    """eps, enstrophy and eta for a 3D snapshot.  All fields are (nz, ny, nx).

    `a` is the HEAVY-phase volume fraction (see the module docstring).
    Returns None when the snapshot has no mixing zone.

    This holds nine gradient arrays at once, so at 400x500x200 it needs about
    5 GB.  The intermediates are dropped as soon as they are consumed, in the
    order the ported script used.
    """
    for name, arr in (('v', v), ('w', w), ('a', a)):
        if arr.shape != u.shape:
            raise ValueError(f'{name} shape {arr.shape} != u shape {u.shape}')
    ny = u.shape[1]
    nu = a * nu_heavy + (1.0 - a) * nu_light

    duz, duy, dux = np.gradient(u, dz, dy, dx)
    dvz, dvy, dvx = np.gradient(v, dz, dy, dx)
    dwz, dwy, dwx = np.gradient(w, dz, dy, dx)
    sxy = 0.5 * (duy + dvx)
    sxz = 0.5 * (duz + dwx)
    syz = 0.5 * (dvz + dwy)
    eps = 2.0 * nu * (dux ** 2 + dvy ** 2 + dwz ** 2
                      + 2.0 * (sxy ** 2 + sxz ** 2 + syz ** 2))
    del sxy, sxz, syz
    ens = (dwy - dvz) ** 2 + (duz - dwx) ** 2 + (dvx - duy) ** 2
    del duz, duy, dux, dvz, dvy, dvx, dwz, dwy, dwx

    abar = a.mean(axis=(0, 2))
    jj = mixing_rows(abar, ny)
    if jj is None:
        return None
    j0, j1 = jj
    mz = (slice(1, -1), slice(j0, j1), slice(1, -1))
    core = (slice(1, -1), slice(1, -1), slice(1, -1))
    return _close(nu, eps, ens, (u, v, w), mz, core, j0, j1, dx, dy, abar)
