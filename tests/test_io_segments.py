"""The plain segment/polyline reader, and the partly-reconstructed 3D guard."""
import numpy as np
import pytest

from rtfractal import fractal, shapes
from rtfractal.io import segments as segio
from rtfractal.io.case import OpenFOAM3DCase, Snapshot


# ----------------------------------------------------------------- the reader
def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_four_columns_round_trip(tmp_path):
    segs = shapes.koch(4)
    p = tmp_path / 'k.txt'
    np.savetxt(p, segs, fmt='%.17g')
    assert np.allclose(segio.read_segments(str(p)), segs)


def test_delimiters_and_header(tmp_path):
    """Whitespace, comma, semicolon and tab, mixed, with a header and comments."""
    p = _write(tmp_path, 'mixed.csv',
               'x1,y1,x2,y2\n'
               '0,0,1,0\n'
               '1 0 1 1\n'
               '1;1;0;1\n'
               '# a comment\n'
               '\n'
               '% another\n'
               '0\t1\t0\t0\n')
    s = segio.read_segments(p)
    assert s.shape == (4, 4)
    assert np.allclose(s[0], [0, 0, 1, 0]) and np.allclose(s[3], [0, 1, 0, 0])


def test_extra_columns_are_ignored(tmp_path):
    p = _write(tmp_path, 'six.txt', '0 0 1 0 99 -1\n1 0 1 1 98 -2\n')
    assert segio.read_segments(p).shape == (2, 4)


def test_polyline_is_differenced_into_segments(tmp_path):
    segs = shapes.koch(4)
    pts = np.vstack([segs[:, :2], segs[-1, 2:]])
    p = tmp_path / 'poly.txt'
    np.savetxt(p, pts, fmt='%.17g')
    assert np.allclose(segio.read_segments(str(p), kind='polyline'), segs)


def test_polyline_and_segment_forms_give_the_same_dimension(tmp_path):
    """The whole point of the reader: two spellings of one curve, one D."""
    segs = shapes.koch(6)
    pts = np.vstack([segs[:, :2], segs[-1, 2:]])
    a, b = tmp_path / 'a.txt', tmp_path / 'b.txt'
    np.savetxt(a, segs, fmt='%.17g')
    np.savetxt(b, pts, fmt='%.17g')
    kw = dict(min_box=2 * 0.4 / 640, max_box=0.4 / 8, factor=1.5, backend='cpu')
    da = fractal.dimension_of_segments(segio.read_segments(str(a)), **kw)['dimension']
    db = fractal.dimension_of_segments(segio.read_segments(str(b), kind='polyline'), **kw)['dimension']
    assert da == pytest.approx(db, abs=1e-12)


def test_two_columns_in_auto_mode_refuses_and_says_which(tmp_path):
    p = _write(tmp_path, 'two.txt', '0 0\n1 0\n1 1\n')
    with pytest.raises(ValueError, match='ambiguous') as e:
        segio.read_segments(p)
    assert '--as polyline' in str(e.value) and '--as points' in str(e.value)


def test_read_segments_refuses_points_and_names_the_right_call(tmp_path):
    """A point set is not segments, so read_segments says where to go."""
    p = _write(tmp_path, 'two.txt', '0 0\n1 0\n1 1\n')
    with pytest.raises(ValueError, match='read_points'):
        segio.read_segments(p, kind='points')


def test_read_points(tmp_path):
    p = _write(tmp_path, 'cloud.csv', 'x,y\n0,0\n1,0\n0.5,0.5\n')
    pts = segio.read_points(p)
    assert pts.shape == (3, 2)
    assert np.allclose(pts[2], [0.5, 0.5])


def test_read_points_ignores_extra_columns(tmp_path):
    p = _write(tmp_path, 'xyz.txt', '0 0 7\n1 0 8\n0.5 0.5 9\n')
    assert segio.read_points(p).shape == (3, 2)


def test_read_points_needs_two_columns(tmp_path):
    p = _write(tmp_path, 'one.txt', '0\n1\n2\n')
    with pytest.raises(ValueError, match='needs 2 columns'):
        segio.read_points(p)


def test_describe_points_reports_the_spacing(tmp_path):
    g = np.linspace(0, 1, 11)
    pts = np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2)
    p = tmp_path / 'lattice.txt'
    np.savetxt(p, pts)
    d = segio.describe_points(segio.read_points(str(p)))
    assert d['n'] == 121
    assert d['extent'] == pytest.approx(1.0)
    assert d['spacing'] == pytest.approx(0.1)


def test_ragged_rows_name_the_line(tmp_path):
    p = _write(tmp_path, 'ragged.txt', '0 0 1 0\n1 0 1\n')
    with pytest.raises(ValueError, match=r'ragged\.txt:2'):
        segio.read_segments(p)


def test_non_numeric_after_the_first_row_names_the_line(tmp_path):
    p = _write(tmp_path, 'bad.txt', '0 0 1 0\nnonsense here now\n')
    with pytest.raises(ValueError, match=r'bad\.txt:2'):
        segio.read_segments(p)


def test_empty_file(tmp_path):
    p = _write(tmp_path, 'empty.txt', '# nothing\n\n')
    with pytest.raises(ValueError, match='no numeric data'):
        segio.read_segments(p)


def test_three_columns_is_an_error(tmp_path):
    p = _write(tmp_path, 'three.txt', '0 0 1\n1 0 1\n')
    with pytest.raises(ValueError, match='found 3'):
        segio.read_segments(p)


# ------------------------------------------------------------- the diagnostics
def test_describe_reports_extent_and_segment_lengths():
    d = segio.describe(shapes.koch(5))
    assert d['n'] == 4 ** 5
    assert d['extent'] == pytest.approx(0.4)
    assert d['len_min'] == pytest.approx(d['len_max'])      # Koch segments are equal


def test_suggest_min_box_warns_when_the_ladder_outruns_the_data():
    """Minkowski level 3 is the measured case: a ladder four steps below the
    construction scale reads -0.214 against exact, -0.086 at the feature size."""
    coarse = shapes.minkowski(3)
    mb, warn = segio.suggest_min_box(coarse)
    assert mb == pytest.approx(0.4 / 512)
    assert any('below the 10th-percentile segment length' in w for w in warn)


def test_suggest_min_box_is_quiet_on_a_well_resolved_curve():
    mb, warn = segio.suggest_min_box(shapes.koch(7))
    assert not any('10th-percentile' in w for w in warn)


def test_suggest_min_box_warns_about_units():
    """offset_fractions switches on ABSOLUTE box sizes, so D is not scale
    invariant; a file in LJ sigma and one in metres are not comparable."""
    _, warn = segio.suggest_min_box(shapes.koch(5, L=300.0))
    assert any('absolute box' in w for w in warn)
    _, warn = segio.suggest_min_box(shapes.koch(5, L=0.4))
    assert not any('absolute box' in w for w in warn)


def test_taking_the_min_box_advice_improves_the_dimension():
    """The warning has to be worth acting on, not just true."""
    segs = shapes.minkowski(3)
    d = segio.describe(segs)
    kw = dict(max_box=d['extent'] / 8, factor=1.5, backend='cpu')
    bad = fractal.dimension_of_segments(segs, min_box=d['extent'] / 512, **kw)['dimension']
    good = fractal.dimension_of_segments(segs, min_box=d['len_p10'], **kw)['dimension']
    exact = shapes.DIMENSION['minkowski']
    assert abs(good - exact) < abs(bad - exact) / 2


# --------------------------------------------------- the partly-reconstructed guard
def test_reading_fields_without_a_time_directory_says_to_run_reconstructPar():
    """A 3D OpenFOAM case writes interface.vtk live but needs reconstructPar for
    the volume fields, so a case can have every interface mesh and only some
    time directories.  Phase 1 catches a read failure and fills the mixing
    columns with NaN, so without this the analysis quietly produces NaN for
    exactly the snapshots being studied."""
    case = OpenFOAM3DCase.__new__(OpenFOAM3DCase)      # no case on disk needed
    case.path = '/somewhere/case'
    snap = Snapshot(t=11.6, time_idx=11600, path=None, iface='/somewhere/interface.vtk')
    with pytest.raises(FileNotFoundError, match='reconstructPar'):
        case.read_alpha_3d(snap)
    with pytest.raises(FileNotFoundError, match='reconstructPar'):
        case.read_U_3d(snap)
    with pytest.raises(FileNotFoundError, match='t = 11.6'):
        case.read_alpha_3d(snap)


def test_a_snapshot_with_a_time_directory_is_not_blocked():
    case = OpenFOAM3DCase.__new__(OpenFOAM3DCase)
    case.path = '/somewhere/case'
    snap = Snapshot(t=8.0, time_idx=8000, path='/somewhere/case/8', iface='x.vtk')
    case._require_fields(snap, 'alpha.water')          # must not raise
