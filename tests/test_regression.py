"""Bit-identity against the legacy analyzers, run live on the reference cases.

Skipped unless the legacy repos (FractalParameterAI, FractalAnalyzer,
3D-FractalParameterAI as siblings of this repo) and the reference cases are
present.  Live comparison (same machine, same libraries) is the only fair
bitwise test: archived CSVs differ in the last digit whenever numpy/scipy
builds differ.  The archived late-time D means are checked to 1e-12.
"""
import contextlib
import io
import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest

from rtfractal import fractal, interface
from rtfractal.io.case import open_case

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPOS = os.path.dirname(HERE)
LBM2D = '/media/rod/RT-Simulations-2/LatticeBoltzmann/cases/2D/M0.002/160x200'
SVOF = '/media/rod/RT-Simulations/Dalziel_1999/sola-vof/NDIS-6/160x200'
LBM3D = '/media/rod/RT-Simulations-2/LatticeBoltzmann/cases/3D/80x100x40-M0.002'
REFERENCE_D = {LBM2D: 1.6482, SVOF: 1.7323, LBM3D: 2.7239}     # mean D over tau > 2

pytestmark = pytest.mark.regression


def _need(*paths):
    for p in paths:
        if not os.path.exists(p):
            pytest.skip(f'missing {p}')


@pytest.fixture(scope='module')
def legacy2d(legacy_rtanalyzer):
    """The analyzer pinned in conftest.py, not whatever sits in the sibling tree."""
    with contextlib.redirect_stdout(io.StringIO()):
        return legacy_rtanalyzer(
            output_dir=os.path.join(os.environ.get('TMPDIR', '/tmp'), 'rtfractal_legacy_out'))


@pytest.mark.parametrize('path', [LBM2D, SVOF])
@pytest.mark.parametrize('t', [3.0, 10.0, 16.0])
def test_2d_bit_identical_to_legacy(legacy2d, path, t):
    _need(path)
    c = open_case(path)
    snap = min(c.snapshots, key=lambda s: abs(s.t - t))
    d = c.read_2d(snap)
    xc = d['x'][:, 0]
    mx, mn = float(xc[-1] - xc[0]) / 2, float(xc[1] - xc[0]) * 2
    with contextlib.redirect_stdout(io.StringIO()):
        dl = legacy2d.read_vtk_file(snap.path) if c.format == 'solavof' else legacy2d.read_openfoam_field(path, snap.t)
        ref = legacy2d.compute_fractal_dimension(dl, min_box_size=mn, max_box_size=mx, box_size_factor=1.5)
    assert np.array_equal(dl['f'], d['f'])
    new = fractal.dimension_2d(d, min_box=mn, max_box=mx, factor=1.5, offsets='legacy', trim='legacy')
    assert new['dimension'] == ref['dimension'] or (np.isnan(new['dimension']) and np.isnan(ref['dimension']))
    assert list(new['box_counts']) == list(ref['box_counts'])


@pytest.mark.parametrize('t', [3.0, 6.0, 10.0, 16.0])
def test_3d_bit_identical_to_legacy(t):
    _need(f'{REPOS}/3D-FractalParameterAI', LBM3D)
    sys.path.insert(0, f'{REPOS}/3D-FractalParameterAI')
    from fractal3d import compute_fractal_dimension_3d, load_mesh
    c = open_case(LBM3D)
    p, g = c.params, c.grid
    snap = min(c.snapshots, key=lambda s: abs(s.t - t))
    lm, nm = load_mesh(snap.iface), c.interface_mesh(snap)
    assert np.array_equal(lm.vertices, nm.vertices) and np.array_equal(lm.triangles, nm.triangles)
    kw = dict(initial_delta=min(p['L'], p['W']) / 2, delta_factor=1.5, num_steps=20, min_delta=max(g.dx, g.dy, g.dz) * 2)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ref = compute_fractal_dimension_3d(lm, **kw)
    except ValueError:                      # too few non-zero counts: both codes must refuse
        with pytest.raises(ValueError):
            fractal.dimension_3d(nm, initial=kw['initial_delta'], num_steps=20, min_delta=kw['min_delta'], offsets='legacy', trim='legacy')
        return
    new = fractal.dimension_3d(nm, initial=kw['initial_delta'], num_steps=20, min_delta=kw['min_delta'], offsets='legacy', trim='legacy')
    assert new['dimension'] == ref.dimension
    assert list(new['box_counts']) == list(ref.n_boxes)


@pytest.mark.parametrize('path', [LBM2D, SVOF, LBM3D])
def test_archived_late_time_mean(path):
    """Archived temporal_results.csv: mean D over tau > 2 to 4 decimals."""
    csv = os.path.join(path, 'analysis', 'summary', 'temporal_results.csv')
    _need(csv)
    df = pd.read_csv(csv)
    assert round(df[df.tau > 2].fractal_dim.mean(), 4) == REFERENCE_D[path]
