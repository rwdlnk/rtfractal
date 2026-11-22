"""LAMMPS dump reading and Cao's midpoint interface construction.

The angular filter is vectorised here and was a per-atom double loop in the
run-folder script it replaces, so it is tested against a literal
transcription of that loop as well as against the interfaces the script
already produced.
"""
import os

import numpy as np
import pytest

from rtfractal.io import lammps, segments as segio

CAO = '/media/rod/Scratch/Cao_MD/2D/A1_seed12345_T2.5'

DUMP_2D = """ITEM: TIMESTEP
500
ITEM: NUMBER OF ATOMS
4
ITEM: BOX BOUNDS ff ff pp
-2.0000000000000000e+00 1.0000000000000000e+01
-2.0000000000000000e+00 2.0000000000000000e+01
-5.0000000000000000e-01 5.0000000000000000e-01
ITEM: ATOMS id type x y
1 1 0.0 0.0
2 2 1.0 0.0
3 1 0.0 5.0
4 3 9.0 9.0
"""


def _write(tmp_path, text, name='dump.test.500'):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# ------------------------------------------------------------------ reading
def test_reads_a_single_frame(tmp_path):
    f = lammps.read_dump(_write(tmp_path, DUMP_2D))
    assert f.timestep == 500
    assert f.n_atoms == 4
    assert f.columns == ('id', 'type', 'x', 'y')
    assert f.box_lo.tolist() == [-2.0, -2.0, -0.5]
    assert f.box_hi.tolist() == [10.0, 20.0, 0.5]
    assert f.types().tolist() == [1, 2, 1, 3]
    assert f.positions(2).shape == (4, 2)


def test_box_bounds_line_count_follows_the_flags(tmp_path):
    """Two flags, two bound lines, and the ATOMS section still lands right."""
    two_d = (DUMP_2D.replace('ITEM: BOX BOUNDS ff ff pp', 'ITEM: BOX BOUNDS ff ff')
                    .replace('-5.0000000000000000e-01 5.0000000000000000e-01\n', ''))
    f = lammps.read_dump(_write(tmp_path, two_d))
    assert len(f.box_lo) == 2
    assert f.box_hi.tolist() == [10.0, 20.0]
    assert f.n_atoms == 4
    assert f.types().tolist() == [1, 2, 1, 3]


def test_triclinic_tilt_factors_are_not_axes():
    assert lammps._n_box_lines('BOX BOUNDS xy xz yz pp pp pp') == 3
    assert lammps._n_box_lines('BOX BOUNDS ff ff pp') == 3
    assert lammps._n_box_lines('BOX BOUNDS ff ff') == 2
    assert lammps._n_box_lines('BOX BOUNDS') == 3


def test_column_order_is_taken_from_the_atoms_line(tmp_path):
    shuffled = DUMP_2D.replace('ITEM: ATOMS id type x y', 'ITEM: ATOMS x y type id') \
                      .replace('1 1 0.0 0.0', '0.0 0.0 1 1') \
                      .replace('2 2 1.0 0.0', '1.0 0.0 2 2') \
                      .replace('3 1 0.0 5.0', '0.0 5.0 1 3') \
                      .replace('4 3 9.0 9.0', '9.0 9.0 3 4')
    f = lammps.read_dump(_write(tmp_path, shuffled))
    assert f.types().tolist() == [1, 2, 1, 3]
    assert f.positions(2)[1].tolist() == [1.0, 0.0]


def test_a_missing_column_is_named(tmp_path):
    f = lammps.read_dump(_write(tmp_path, DUMP_2D))
    with pytest.raises(ValueError, match="no column 'z'"):
        f.positions(3)


def test_several_frames_are_iterated_and_refused_singly(tmp_path):
    path = _write(tmp_path, DUMP_2D + DUMP_2D.replace('\n500\n', '\n1000\n'))
    assert [f.timestep for f in lammps.read_frames(path)] == [500, 1000]
    with pytest.raises(ValueError, match='more than one frame'):
        lammps.read_dump(path)


def test_a_short_frame_is_an_error_not_a_silent_truncation(tmp_path):
    short = DUMP_2D.replace('4 3 9.0 9.0\n', '')
    with pytest.raises(ValueError, match='declares 4 atoms and holds 3'):
        lammps.read_dump(_write(tmp_path, short))


def test_an_empty_file_is_an_error(tmp_path):
    with pytest.raises(ValueError, match='no dump frames'):
        lammps.read_dump(_write(tmp_path, ''))


def test_timestep_of_a_filename():
    assert lammps.timestep_of('/a/b/dump.mixing.500000') == 500000
    assert lammps.timestep_of('dump.mixing') is None


def test_select_keeps_order_and_drops_walls(tmp_path):
    f = lammps.read_dump(_write(tmp_path, DUMP_2D))
    assert f.select(exclude=3).types().tolist() == [1, 2, 1]
    assert f.select(types=[1]).positions(2).tolist() == [[0.0, 0.0], [0.0, 5.0]]


def test_a_dump_handed_to_the_point_reader_says_what_to_do(tmp_path):
    path = _write(tmp_path, DUMP_2D)
    with pytest.raises(ValueError, match='rtfractal interface'):
        segio.read_points(path)


# ------------------------------------------------------- midpoint interface
def test_two_atoms_give_one_midpoint():
    mid = lammps.midpoint_interface(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]))
    assert mid.shape == (1, 2)
    assert mid[0].tolist() == [0.5, 0.0]


def test_atoms_beyond_the_cutoff_give_nothing():
    a, b = np.array([[0.0, 0.0]]), np.array([[5.0, 0.0]])
    assert len(lammps.midpoint_interface(a, b, cutoff=1.8)) == 0
    assert len(lammps.midpoint_interface(a, b, cutoff=6.0)) == 1


def test_each_pair_contributes_once_after_dedup():
    """Both passes find the same pair, so without dedup it appears twice."""
    a, b = np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]])
    assert len(lammps.midpoint_interface(a, b, dedup=0.1)) == 1
    assert len(lammps.midpoint_interface(a, b, dedup=0.0)) == 2


def test_the_angular_filter_drops_a_hidden_neighbour():
    """Two b atoms in the same direction: only the nearer one counts."""
    a = np.array([[0.0, 0.0]])
    b = np.array([[1.0, 0.0], [1.6, 0.0]])
    assert len(lammps.midpoint_interface(a, b, angle=30.0, dedup=0.0)) == 3
    assert len(lammps.midpoint_interface(a, b, angle=0.0, dedup=0.0)) == 4


def test_an_empty_species_gives_an_empty_interface():
    assert len(lammps.midpoint_interface(np.zeros((0, 2)), np.ones((3, 2)))) == 0


def test_mismatched_dimensions_are_refused():
    with pytest.raises(ValueError, match='equal d'):
        lammps.midpoint_interface(np.zeros((2, 2)), np.zeros((2, 3)))


def test_extract_names_the_types_present_when_one_is_missing(tmp_path):
    only_one = DUMP_2D.replace('2 2 1.0 0.0', '2 1 1.0 0.0')
    f = lammps.read_dump(_write(tmp_path, only_one))
    with pytest.raises(ValueError, match=r'no interface between them'):
        lammps.extract_interface(f)


def test_a_wall_atom_is_not_a_species(tmp_path):
    """A type-3 wall pressed against the interface contributes nothing.

    `exclude` is belt and braces here: a type outside `types` is already no
    species.  It earns its keep only where the walls share a fluid type.
    """
    with_wall = DUMP_2D.replace('4 3 9.0 9.0', '4 3 0.5 0.0')
    f = lammps.read_dump(_write(tmp_path, with_wall))
    excluded = lammps.extract_interface(f, exclude=(3,))
    not_excluded = lammps.extract_interface(f, exclude=())
    assert np.array_equal(excluded, not_excluded)
    assert len(excluded) == 1


# ------------------------------- the vectorised filter against the original
def _angular_keep_reference(unit, valid, cos_min):
    """The run-folder script's filter, transcribed per reference atom.

    Its rule: over every pair of neighbours, if they subtend less than the
    threshold, drop the farther one; skip a neighbour already dropped.
    """
    n, k, _ = unit.shape
    keep = valid.copy()
    for row in range(n):
        for i in range(k):
            if not keep[row, i]:
                continue
            for j in range(i + 1, k):
                if not keep[row, j]:
                    continue
                if np.clip(np.dot(unit[row, i], unit[row, j]), -1.0, 1.0) > cos_min:
                    keep[row, j] = False          # distance-sorted: j is the farther
    return keep


@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
def test_the_vectorised_angular_filter_matches_the_original_loop(seed):
    rng = np.random.default_rng(seed)
    n, k = 200, 8
    vecs = rng.normal(size=(n, k, 2))
    unit = vecs / np.linalg.norm(vecs, axis=2)[:, :, None]
    valid = rng.random((n, k)) > 0.2
    valid[:, 0] = True
    cos_min = np.cos(np.radians(30.0))
    assert np.array_equal(lammps._angular_keep(unit, valid, cos_min),
                          _angular_keep_reference(unit, valid, cos_min))


# ----------------------------------------------- against the real extraction
@pytest.mark.regression
@pytest.mark.parametrize('step', [0, 250000, 500000])
def test_reproduces_the_existing_cao_extraction(step):
    """Byte-for-byte against what the run-folder script wrote."""
    dump = f'{CAO}/dump.mixing.{step}'
    ref_csv = f'{CAO}/interfaces/interface_t{step:08d}.csv'
    if not (os.path.exists(dump) and os.path.exists(ref_csv)):
        pytest.skip('the Cao MD run folder is not mounted')
    mine = lammps.extract_interface(lammps.read_dump(dump))
    ref = np.loadtxt(ref_csv, delimiter=',', skiprows=1)
    assert mine.shape == ref.shape
    assert np.allclose(mine, ref, atol=1e-6, rtol=0)     # the CSV is written %.6f


@pytest.mark.regression
def test_the_written_csv_is_byte_identical_to_the_existing_one(tmp_path):
    dump = f'{CAO}/dump.mixing.0'
    ref_csv = f'{CAO}/interfaces/interface_t00000000.csv'
    if not (os.path.exists(dump) and os.path.exists(ref_csv)):
        pytest.skip('the Cao MD run folder is not mounted')
    out = str(tmp_path / 'interface_t00000000.csv')
    lammps.write_interface(lammps.extract_interface(lammps.read_dump(dump)), out)
    with open(out) as a, open(ref_csv) as b:
        assert a.read() == b.read()
