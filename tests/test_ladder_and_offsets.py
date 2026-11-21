"""Ladder construction, the `single` offset mode, and the crossover diagnostics.

These exist because a measured +0.078 was attributed to the offset convention
when it belonged to the ladder.  The two are separable only if both are
expressible, which until now the ladder was not.
"""
import numpy as np
import pytest

from rtfractal import fractal, shapes
from rtfractal.boxcount import points2d, scaling, segments2d
from rtfractal.boxcount.ladder import box_ladder
from rtfractal.boxcount.offsets import offset_fractions, origin


# ---------------------------------------------------------------- box_ladder
def test_num_sizes_hits_both_ends_exactly():
    lad = box_ladder(0.8, 2.5, num_sizes=10)
    assert len(lad) == 10
    assert lad[0] == pytest.approx(2.5)
    assert lad[-1] == pytest.approx(0.8)
    ratios = lad[:-1] / lad[1:]
    assert np.allclose(ratios, ratios[0])


def test_factor_ladder_hits_neither_end():
    """Why num_sizes exists: a factor ladder cannot express the published window."""
    lad = box_ladder(0.8, 2.5, factor=1.5)
    assert len(lad) == 3
    assert lad[-1] == pytest.approx(2.5 / 1.5 ** 2)
    assert lad[-1] > 0.8


def test_explicit_sizes_are_taken_as_given_descending():
    lad = box_ladder(sizes=[0.5, 2.0, 1.0])
    assert list(lad) == [2.0, 1.0, 0.5]


def test_ladder_rejects_contradictions():
    with pytest.raises(ValueError, match='not both'):
        box_ladder(1.0, 2.0, num_sizes=5, sizes=[1.0, 2.0])
    with pytest.raises(ValueError, match='at least 2 sizes'):
        box_ladder(sizes=[1.0])
    with pytest.raises(ValueError, match='below min_box'):
        box_ladder(2.0, 1.0)
    with pytest.raises(ValueError, match='at least 2'):
        box_ladder(1.0, 2.0, num_sizes=1)


# ------------------------------------------------------------- single offsets
def test_single_mode_tries_one_origin_at_the_minimum_corner():
    assert offset_fractions(1.0, 'single').tolist() == [0.0]
    assert offset_fractions(0.001, 'single').tolist() == [0.0]
    assert origin(3.0, 0.5, 1.0, 'single') == 3.0


def test_default_mode_is_unchanged():
    assert offset_fractions(1.0).tolist() == offset_fractions(1.0, 'covering').tolist()
    assert len(offset_fractions(0.001)) == 4
    assert len(offset_fractions(0.01)) == 3
    assert len(offset_fractions(1.0)) == 2


def test_single_reproduces_a_hand_written_naive_count():
    rng = np.random.default_rng(7)
    pts = rng.random((3000, 2)) * 20.0
    s, c, _ = points2d.box_counting_2d_points(pts, 1.0, 4.0, num_sizes=5, offsets='single',
                                              allow_below_spacing=True)
    x, y = pts[:, 0], pts[:, 1]
    for size, count in zip(s, c):
        naive = len(set(zip(((x - x.min()) / size).astype(np.int64),
                            ((y - y.min()) / size).astype(np.int64))))
        assert int(count) == naive


def test_covering_never_exceeds_the_unshifted_grid_of_its_own_family():
    """`single` carries no ordering guarantee against `covering`, because
    dropping the margin puts it on a different grid entirely.  What is
    guaranteed is that covering, which minimises, cannot exceed any member of
    its own family, the f = 0 grid included."""
    rng = np.random.default_rng(11)
    pts = rng.random((3000, 2)) * 20.0
    sizes, c_cov, bbox = points2d.box_counting_2d_points(
        pts, 1.0, 5.0, num_sizes=6, offsets='covering', allow_below_spacing=True)
    lo_x, lo_y = bbox[0], bbox[1]
    x, y = pts[:, 0], pts[:, 1]
    for size, count in zip(sizes, c_cov):
        unshifted = len(set(zip(((x - lo_x) / size).astype(np.int64),
                                ((y - lo_y) / size).astype(np.int64))))
        assert int(count) <= unshifted


def _rough_band(seed=11, n=20000):
    """A rough curve with scatter: not axis-aligned, not space-filling."""
    rng = np.random.default_rng(seed)
    t = rng.random(n)
    return np.column_stack([t * 100.0, np.sin(t * 60.0) * 3.0 + rng.normal(0, 0.3, n)])


def test_single_and_covering_agree_in_d_on_a_rough_cloud():
    """Different grids, so different counts, but the same measurement."""
    pts = _rough_band()
    kw = dict(num_sizes=10, allow_below_spacing=True)
    d_single = fractal.dimension_of_points(pts, 0.8, 2.5, offsets='single', **kw)['dimension']
    d_cov = fractal.dimension_of_points(pts, 0.8, 2.5, offsets='covering', **kw)['dimension']
    assert d_single == pytest.approx(d_cov, abs=0.02)


def test_the_margin_quantises_the_largest_box_on_a_space_filling_cloud():
    """Why the top rung of a ladder is the least trustworthy one.

    A cloud filling a 20 x 20 square tiles exactly into 4 x 4 boxes of size 5.
    The 1% bounding-box margin pushes the span to 20.4, which needs 5 boxes per
    axis, and the count comes out half as large again.  It is a counting
    artefact at the top of the ladder and not a property of the set, which is
    the reason the default largest box is extent/8 and not extent/4.
    """
    rng = np.random.default_rng(11)
    pts = rng.random((3000, 2)) * 20.0
    _, c_single, _ = points2d.box_counting_2d_points(
        pts, 2.5, 5.0, num_sizes=2, offsets='single', allow_below_spacing=True)
    _, c_cov, _ = points2d.box_counting_2d_points(
        pts, 2.5, 5.0, num_sizes=2, offsets='covering', allow_below_spacing=True)
    assert c_single[0] == 16                      # the exact 4 x 4 tiling
    assert c_cov[0] > c_single[0]                 # the margin needs a fifth column


def test_koch_dimension_survives_the_single_mode():
    """Covering stays the reference; single must not drift far from it."""
    segs = shapes.koch(6)
    kw = dict(min_box=0.4 / 512, max_box=0.4 / 8, factor=1.5)
    cov = scaling.dimension_2d(*segments2d.box_counting_2d(segs, offsets='covering', **kw)[:2])
    sing = scaling.dimension_2d(*segments2d.box_counting_2d(segs, offsets='single', **kw)[:2])
    assert cov['dimension'] == pytest.approx(np.log(4) / np.log(3), abs=0.03)
    assert sing['dimension'] == pytest.approx(cov['dimension'], abs=0.05)


# ------------------------------------------------------------- local slope
def test_local_slope_of_an_exact_power_law_is_the_exponent():
    sizes = np.geomspace(10.0, 0.1, 25)
    counts = 3.0 * sizes ** -1.4
    mids, sl = scaling.local_slope(sizes, counts)
    assert len(mids) == len(sizes) - 1
    assert np.allclose(sl, 1.4)


def test_local_slope_tracks_a_bend():
    sizes = np.geomspace(10.0, 0.1, 21)
    counts = np.where(sizes > 1.0, sizes ** -1.0, sizes ** -2.0)
    _, sl = scaling.local_slope(sizes, counts)
    assert sl[0] == pytest.approx(1.0, abs=1e-9)
    assert sl[-1] == pytest.approx(2.0, abs=1e-9)


def test_local_slope_of_a_short_ladder_is_empty():
    mids, sl = scaling.local_slope([1.0], [1.0])
    assert len(mids) == 0 and len(sl) == 0


# ------------------------------------------------------------- window flag
def test_a_narrow_window_is_recorded_as_a_fallback():
    sizes = np.geomspace(2.5, 0.8, 10)          # 0.49 decades, the Cao window
    counts = 100.0 * sizes ** -1.36
    res = scaling.dimension_2d(sizes, counts)
    assert res['dimension'] == pytest.approx(1.36, abs=1e-6)
    assert res['span_decades'] < scaling.SPAN_MIN
    assert res['window'] == 'fallback'


def test_a_wide_clean_ladder_selects_a_window():
    sizes = np.geomspace(100.0, 0.1, 30)
    counts = 100.0 * sizes ** -1.5
    res = scaling.dimension_2d(sizes, counts)
    assert res['span_decades'] >= scaling.SPAN_MIN
    assert res['window'] == 'selected'


def test_3d_results_carry_the_same_two_keys():
    deltas = np.geomspace(10.0, 0.1, 25)
    counts = 100.0 * deltas ** -2.3
    res = scaling.dimension_3d(deltas, counts)
    assert res['dimension'] == pytest.approx(2.3, abs=1e-6)
    assert res['window'] == 'selected'
    assert res['span_decades'] > 1.0


# ------------------------------------------------------------- diagnostics
def test_point_result_reports_saturation_and_local_slope():
    rng = np.random.default_rng(3)
    pts = rng.random((2000, 2)) * 10.0
    res = fractal.dimension_of_points(pts, 0.5, 2.5, num_sizes=8, allow_below_spacing=True)
    lad = res['ladder']
    assert len(lad['sizes']) == 8
    assert np.all(np.diff(lad['saturation']) > 0)        # smaller boxes, more boxes
    assert np.all(lad['saturation'] <= 1.0)
    assert len(lad['local_slope']) == 7


def test_saturation_reaches_one_below_the_spacing():
    rng = np.random.default_rng(5)
    pts = rng.random((1500, 2)) * 10.0
    res = fractal.dimension_of_points(pts, 0.005, 1.0, num_sizes=6, allow_below_spacing=True)
    assert res['ladder']['saturation'].max() == pytest.approx(1.0, abs=0.01)


def test_the_ladder_choice_moves_d_more_than_the_offset_choice_on_a_dense_cloud():
    """The finding this module exists for, as a test on a reproducible cloud."""
    rng = np.random.default_rng(17)
    t = rng.random(20000)
    pts = np.column_stack([t * 100.0, np.sin(t * 60.0) * 3.0 + rng.normal(0, 0.3, len(t))])
    kw = dict(allow_below_spacing=True)
    ten = fractal.dimension_of_points(pts, 0.8, 2.5, num_sizes=10, **kw)['dimension']
    ten_s = fractal.dimension_of_points(pts, 0.8, 2.5, num_sizes=10, offsets='single', **kw)['dimension']
    three = fractal.dimension_of_points(pts, 0.8, 2.5, factor=1.5, **kw)['dimension']
    assert abs(ten - ten_s) < abs(ten - three)

