"""Shared fixtures: a legacy analyzer frozen at the commit before it changed.

The bit-identity tests prove that `rtfractal --offsets legacy` reproduces
what produced the published numbers.  Their reference used to be whatever
sat in the sibling `FractalParameterAI` checkout, which worked only while
nobody edited it.  On 2026-11-21 commit `8ad729e` rewired
`rt_analyzer.compute_fractal_dimension` to call rtfractal with covering
offsets, so the tests began comparing rtfractal in legacy mode against
rtfractal in default mode and failed by construction: LBM 160x200 at
t = 10 gave 1.7622 against 1.5296, which is the offset correction and not
a regression.

A reproduction test needs an immutable reference, so the reference is now
pinned to `869103d`, the parent of that commit, checked out into a
throwaway directory for the session.  Pinning restores exact bit-identity
(1.7622201257971992 either way, verified 2026-11-21).

Two limits worth stating.  The pin freezes which counter gets called, not
what it computes: the arithmetic is `FractalAnalyzer.box_counting_unified`
in a different repository, which `8ad729e` never touched and which these
fixtures still take live.  And forcing the legacy branch at HEAD instead
of pinning does not work, because its constructor sets
`self.fractal_analyzer = None` in the same breath as announcing the
fallback, so every dimension comes back NaN.
"""
import contextlib
import io
import logging
import os
import subprocess
import sys
import tarfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPOS = os.path.dirname(os.path.dirname(HERE))

LEGACY_PIN = '869103d49ed77272b9f4abace99b151e5b4ce093'
# `analyze_rt.py` derives a repo root from its own path and adds core/ and
# integration/ to sys.path, so the pinned checkout has to keep that shape.
# These three are 0.8 MB of the repository's 87.
LEGACY_TREES = ('rt_analyzer', 'core', 'integration')
LEGACY_PIN_WHY = ('the parent of FractalParameterAI 8ad729e, which rewired phase 1 '
                  'to call rtfractal and so ended the reference\'s independence')
LEGACY_ENV = 'RTFRACTAL_LEGACY_RT_ANALYZER'


@pytest.fixture(scope='session')
def legacy_repo_root(tmp_path_factory):
    """-> a FractalParameterAI root as it was at LEGACY_PIN.

    Only the trees the analyzer and its driver import are checked out, in
    the layout they expect to find each other in.  Set
    RTFRACTAL_LEGACY_RT_ANALYZER to a root you prepared instead, for
    example a `git worktree` at some other commit.
    """
    override = os.environ.get(LEGACY_ENV)
    if override:
        if not os.path.isdir(os.path.join(override, 'rt_analyzer')):
            pytest.skip(f'{LEGACY_ENV} is {override!r}, which holds no rt_analyzer/')
        return override

    repo = os.path.join(REPOS, 'FractalParameterAI')
    if not os.path.isdir(os.path.join(repo, '.git')):
        pytest.skip(f'{repo} is not a git checkout, so the legacy analyzer cannot be pinned')
    if subprocess.run(['git', '-C', repo, 'rev-parse', '--verify', '--quiet',
                       f'{LEGACY_PIN}^{{commit}}'], capture_output=True).returncode:
        pytest.skip(f'{repo} has no commit {LEGACY_PIN[:7]} ({LEGACY_PIN_WHY})')

    archive = subprocess.run(['git', '-C', repo, 'archive', LEGACY_PIN, *LEGACY_TREES],
                             capture_output=True)
    if archive.returncode:
        pytest.skip(f'git archive {LEGACY_PIN[:7]} failed: '
                    f'{archive.stderr.decode(errors="replace").strip()[:200]}')
    out = tmp_path_factory.mktemp('legacy_fractalparameterai')
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(out, filter='data')
    if not (out / 'rt_analyzer' / 'rt_analyzer.py').is_file():
        pytest.skip(f'{LEGACY_PIN[:7]} has no rt_analyzer/rt_analyzer.py')
    return str(out)


@pytest.fixture(scope='session')
def legacy_rt_analyzer_dir(legacy_repo_root):
    """-> the pinned `rt_analyzer/` directory, the one that goes on sys.path."""
    return os.path.join(legacy_repo_root, 'rt_analyzer')


@pytest.fixture(scope='session')
def legacy_rtanalyzer(legacy_repo_root, legacy_rt_analyzer_dir):
    """-> the pinned RTAnalyzer class, with its sibling repos importable."""
    fa = os.path.join(REPOS, 'FractalAnalyzer')
    if not os.path.isdir(fa):
        pytest.skip(f'missing {fa}')
    sys.path[:0] = [legacy_rt_analyzer_dir,
                    os.path.join(legacy_repo_root, 'core'),
                    os.path.join(legacy_repo_root, 'integration'), fa]
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()):
        import rt_analyzer
    if getattr(rt_analyzer, '_HAS_RTFRACTAL', False):
        pytest.fail(
            f'the pin did not take: {rt_analyzer.__file__} delegates to rtfractal, so this '
            f'would compare rtfractal against itself. Something else put a newer '
            f'rt_analyzer on sys.path first.')
    return rt_analyzer.RTAnalyzer
