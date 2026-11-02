"""Plots: the legacy figure set is written by `rtfractal analyze` and can be
redrawn from the folder alone by `rtfractal plot`.

Outputs go to $RTFRACTAL_PLOT_TESTS (default $TMPDIR/rtfractal_plot_tests),
never into the case folders.  Figures are drawn at low dpi to save time.
"""
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from rtfractal import cli, open_case, plotdata
from rtfractal.pipeline import phase1, phase2, phase3
from rtfractal.plots import plot_all
from rtfractal.plots.common import Meta, settings_note, settings_tag

LBM2D = '/media/rod/RT-Simulations-2/LatticeBoltzmann/cases/2D/M0.002/160x200'
SVOF = '/media/rod/RT-Simulations/Dalziel_1999/sola-vof/NDIS-6/160x200'
LBM3D = '/media/rod/RT-Simulations-2/LatticeBoltzmann/cases/3D/80x100x40-M0.002'
DPI = '40'

SUMMARY_2D = {'mixing_evolution.png', 'mixing_self_similar.png', 'fractal_dimension_evolution.png',
              'combined_evolution.png', 'D_and_dh_dtau.png', 'dalziel_integral_measures.png',
              'interface_snapshots.png'}
MF_2D = {'f_alpha_evolution_overlay.png', 'Dq_evolution_overlay.png', 'D012_vs_tau.png',
         'multifractality_measures.png', 'phase_portrait.png'}
MF_2D_PER_TIME = {'multifractal_dimensions.png', 'multifractal_spectrum.png', 'multifractal_r_squared.png'}
SPECTRA_2D = {'dalziel_slope_evolution.png', 'dalziel_spectra_snapshots.png', 'spectrum_evolution.png'}
VELOCITY_2D = {'velocity_statistics_evolution.png', 'velocity_field_snapshots.png'}
SUMMARY_3D = {'mixing_evolution_3d.png', 'selfsimilar_scaling_3d.png', 'fractal_evolution_3d.png',
              'dalziel_integrals_3d.png', 'surface_complexity_3d.png'}
MF_3D = {'f_alpha_evolution_3d.png', 'Dq_evolution_3d.png', 'D012_vs_tau_3d.png',
         'multifractality_measures_3d.png', 'phase_portrait_D0_h_3d.png'}


def _need(*paths):
    for p in paths:
        if not os.path.exists(p):
            pytest.skip(f'missing {p}')


def _out(name):
    d = os.path.join(os.environ.get('RTFRACTAL_PLOT_TESTS',
                                    os.path.join(os.environ.get('TMPDIR', '/tmp'), 'rtfractal_plot_tests')), name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


def _pngs(d, sub):
    p = os.path.join(d, sub)
    return {f for f in os.listdir(p) if f.endswith('.png')} if os.path.isdir(p) else set()


def _check_2d(d, mf_prefix, n_mf_dirs):
    assert SUMMARY_2D <= _pngs(d, 'summary')
    assert MF_2D <= _pngs(d, 'multifractal')
    mf = os.path.join(d, 'multifractal')
    subs = [s for s in os.listdir(mf) if s.startswith(mf_prefix) and os.path.isdir(os.path.join(mf, s))]
    assert len(subs) == n_mf_dirs
    for s in subs:
        assert MF_2D_PER_TIME <= _pngs(mf, s), s
    spec = _pngs(d, 'spectra')
    assert SPECTRA_2D <= spec
    assert len([f for f in spec if f.startswith('spectrum_t')]) >= 1
    assert VELOCITY_2D <= _pngs(d, 'velocity')
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith('.png'):
                assert os.path.getsize(os.path.join(root, f)) > 1000, f


# ─── unit ────────────────────────────────────────────────────────────────────

def test_settings_labels():
    m = Meta(analysis_dir='.', dim=2)
    assert 'not recorded' in settings_tag(m, 1) and 'not recorded' in settings_note(m, 1)
    m.phases = {'1': dict(offsets='covering', ladder='L8', trim='r2')}
    assert settings_tag(m, 1) == '[covering offsets, L/8, R² trim]'
    assert 'rtfractal default' in settings_note(m, 1)
    m.phases = {'1': dict(offsets='legacy', ladder='legacy', trim='legacy')}
    assert settings_tag(m, 1).startswith('[LEGACY') and 'L/2' in settings_tag(m, 1)
    m.dim = 3
    assert 'min(L,W)/2' in settings_tag(m, 1)
    m.phases = {'1': dict(offsets='legacy', ladder='L8', trim='r2')}
    assert 'LEGACY offsets' in settings_tag(m, 1) and 'mixed' in settings_note(m, 1)
    m.phases, m.assumed = {}, 'legacy'
    assert 'assumed' in settings_tag(m, 2) and 'ASSUMED' in settings_note(m, 2)


def test_picture_times_follow_legacy_rules():
    assert plotdata.interface_snapshot_targets([0, 1, 20]) == [2.0, 5.0, 8.0, 10.0, 14.0, 18.0]
    assert plotdata.interface_snapshot_targets([0, 9]) == [2.0, 5.0, 8.0]
    assert np.allclose(plotdata.interface_snapshot_targets([0, 1.0]), np.linspace(0, 1, 6))
    assert plotdata.velocity_snapshot_targets([2.0, 10.0, 16.0]) == [2.0, 8.0, 14.0]
    assert plotdata.velocity_snapshot_targets([0.5, 1.0]) == [0.5, 1.0]


def test_velocity_snapshot_strides_large_fields():
    nx, ny = 2100, 50
    x, y = np.meshgrid(np.arange(nx) * 0.1, np.arange(ny) * 0.2, indexing='ij')
    U, V = np.sin(y), np.cos(x)
    s = plotdata.velocity_snapshot({'x': x, 'y': y, 'U': U, 'V': V}, 1.0, 0.5, 1000)
    assert s['stride'] == 3 and s['vel_mag'].shape == (700, 17)
    vort = np.gradient(V, 0.1, axis=0) - np.gradient(U, 0.2, axis=1)       # legacy formula, full resolution
    assert np.allclose(s['vorticity'], vort[::3, ::3], atol=1e-6)
    assert s['vort_p98'] == pytest.approx(np.percentile(np.abs(vort), 98))


# ─── reference cases ─────────────────────────────────────────────────────────

@pytest.mark.regression
@pytest.mark.parametrize('path,prefix', [(LBM2D, 'time_'), (SVOF, 'parameter_')])
def test_analyze_2d_writes_legacy_figure_set_and_replots(path, prefix):
    _need(path)
    out = _out(f'analyze_{os.path.basename(os.path.dirname(path))}')
    cli.main(['analyze', path, '--output-dir', out, '--backend', 'cpu', '--mf-times', '2,10,16',
              '--dpi', DPI])
    _check_2d(out, prefix, 3)
    run = json.load(open(os.path.join(out, plotdata.RUN_JSON)))
    assert run['case']['path'] == path and run['case']['dim'] == 2
    for ph in ('1', '2'):
        assert {k: run['phases'][ph][k] for k in ('offsets', 'ladder', 'trim')} == \
            dict(offsets='covering', ladder='L8', trim='r2')
    assert run['phases']['3']['tke'] == 'uv'
    assert len(plotdata.list_npz(out, 'field')) == 6 and len(plotdata.list_npz(out, 'velocity')) == 3

    # the spectrum_t figure's fit is the phase-3 slope (same bins, same window)
    summ = pd.read_csv(os.path.join(out, 'spectra', 'spectrum_summary.csv'))
    for _, row in summ.iterrows():
        rs = pd.read_csv(os.path.join(out, 'plotdata', f"radial_spectra_t{int(round(row['time'] * 1000))}.csv"))
        k, p = rs['k'].values, rs['E_F'].values
        lo, hi = max(1, len(k) // 4), min(len(k) - 1, 3 * len(k) // 4)
        v = p[lo:hi] > 0
        assert np.polyfit(np.log10(k[lo:hi][v]), np.log10(p[lo:hi][v]), 1)[0] == pytest.approx(row['vof_slope_E'],
                                                                                               abs=1e-12)

    # redraw from the folder alone (no case): everything, including the field figures
    re = _out(f'replot_{os.path.basename(os.path.dirname(path))}')
    files = plot_all(None, out, out_dir=re, dpi=int(DPI), log=lambda *a: None)
    assert len(files) == len({os.path.relpath(f, re) for f in files})
    _check_2d(re, prefix, 3)


@pytest.mark.regression
def test_plot_cli_on_archived_legacy_folder():
    """A folder written by the legacy analyzer: no rtfractal_run.json and no
    plotdata/, so the field figures read the case (the analysis folder's parent)."""
    arch = os.path.join(SVOF, 'analysis')
    _need(os.path.join(arch, 'summary', 'temporal_results.csv'),
          os.path.join(arch, 'multifractal', 'multifractal_summary.csv'),
          os.path.join(arch, 'spectra', 'spectrum_summary.csv'))
    out = _out('replot_archived_svof')
    before = sorted(os.listdir(arch))
    cli.main(['plot', arch, '--output-dir', out, '--dpi', DPI, '--assume-settings', 'legacy'])
    assert sorted(os.listdir(arch)) == before                    # nothing written into the case
    n = len(pd.read_csv(os.path.join(arch, 'multifractal', 'multifractal_summary.csv'))['time'].unique())
    _check_2d(out, 'parameter_', n)


@pytest.mark.regression
def test_3d_figure_set_on_a_few_snapshots():
    _need(LBM3D)
    out = _out('lbm3d')
    case = open_case(LBM3D)
    keep = (0.0, 2.0, 4.0, 6.0, 10.0, 16.0)
    case.snapshots = [s for s in case.snapshots if any(abs(s.t - k) < 1e-9 for k in keep)]
    q = lambda *a, **k: None
    phase1.run_3d(case, out, backend='cpu', log=q)
    phase2.run_3d(case, out, times=(4.0, 10.0), backend='cpu', log=q)
    phase3.run_3d(case, out, times=(6.0, 16.0), log=q)
    files = plot_all(case, out, dpi=int(DPI), log=q)
    assert _pngs(out, 'summary') == SUMMARY_3D and _pngs(out, 'multifractal') == MF_3D
    assert len(files) == len(SUMMARY_3D) + len(MF_3D)

    # rtfractal plot on the folder: the case comes from rtfractal_run.json
    re = _out('lbm3d_replot')
    cli.main(['plot', out, '--output-dir', re, '--dpi', DPI, '--phases', '1,2'])
    assert _pngs(re, 'summary') == SUMMARY_3D and _pngs(re, 'multifractal') == MF_3D


def test_run_record_carries_source_provenance():
    from rtfractal.plotdata import source_info
    info = source_info()
    assert info['rtfractal_version']
    assert set(info) >= {'git_sha', 'git_dirty', 'source'}
    if info['git_sha'] is not None:
        assert len(info['git_sha']) == 40 and isinstance(info['git_dirty'], bool)
