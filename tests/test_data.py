"""Tests for the validation data model: LevelSetRun, ObservedReference, IO, adapter."""
import datetime as dt
import types

import numpy as np
import pytest

from simulations.validation import LevelSetRun, ObservedReference
from simulations.validation import data as D


def _simple_run():
    x = np.array([0.0, 10.0, 20.0])
    y = np.array([0.0, 10.0])
    # two time slices; psi shrinks so the burned region grows
    psi0 = np.array([[1.0, -1.0, 1.0], [1.0, 1.0, 1.0]])
    psi1 = psi0 - 2.0
    return LevelSetRun(psi=np.stack([psi0, psi1]), times=[0.0, 100.0], x=x, y=y,
                       t0="2018-11-08T14:30:00Z")


def test_cell_area_and_masks():
    run = _simple_run()
    assert run.cell_area == pytest.approx(100.0)  # 10 m * 10 m
    assert run.burned_mask(0).sum() == 1          # only the -1.0 cell
    # psi1 = psi0 - 2: psi0 max is 1, so every cell is now <= 0 -> all 6 burn.
    assert run.burned_mask(1).sum() == 6
    # threshold controls inclusion: raising it to 1.0 admits the +1.0 cells too.
    assert run.burned_mask(0, threshold=1.0).sum() == 6
    assert run.burned_mask(0, threshold=-2.0).sum() == 0


def test_uniform_spacing_guard():
    with pytest.raises(ValueError):
        D.cell_area(np.array([0.0, 1.0, 3.0]), np.array([0.0, 1.0]))


def test_field_at_interpolates_midpoint():
    run = _simple_run()
    mid = run.field_at(50.0)  # halfway between psi0 and psi1 = psi0 - 1
    np.testing.assert_allclose(mid, run.psi[0] - 1.0)
    # clamped outside the window
    np.testing.assert_allclose(run.field_at(-10.0), run.psi[0])
    np.testing.assert_allclose(run.field_at(999.0), run.psi[1])


def test_datetime_helpers():
    run = _simple_run()
    assert run.datetime_at(0.0) == dt.datetime(2018, 11, 8, 14, 30, tzinfo=dt.timezone.utc)
    assert run.elapsed_for("2018-11-08T15:30:00Z") == pytest.approx(3600.0)


def test_construction_validates_shapes():
    with pytest.raises(ValueError):
        LevelSetRun(psi=np.zeros((2, 2)), times=[0, 1], x=[0, 1], y=[0, 1])  # psi not 3-D
    with pytest.raises(ValueError):
        LevelSetRun(psi=np.zeros((2, 2, 2)), times=[0], x=[0, 1], y=[0, 1])  # times wrong


def test_npz_round_trip(tmp_path):
    run = _simple_run()
    path = run.save_npz(tmp_path / "run.npz")
    back = LevelSetRun.load_npz(path)
    np.testing.assert_allclose(back.psi, run.psi)
    np.testing.assert_allclose(back.times, run.times)
    assert back.t0 == run.t0


def test_from_simulate_result_reconstructs_grid():
    # variables named "<state>[i,j]" with 1-based i (x), j (y), as run_camp_fire emits.
    names, rows = [], []
    nx, ny = 3, 2
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            names.append(f"psi[{i},{j}]")
            rows.append([float(i * 10 + j), float(i * 10 + j) - 5.0])  # 2 time steps
    result = types.SimpleNamespace(t=np.array([0.0, 1.0]), y=np.array(rows), vars=names)
    run = LevelSetRun.from_simulate_result(result, dx=10.0, t0="2018-11-08T14:30Z")
    assert run.psi.shape == (2, ny, nx)
    # psi[t=0, y=j-1, x=i-1] == i*10 + j
    assert run.psi[0, 0, 0] == pytest.approx(11.0)  # i=1,j=1
    assert run.psi[0, 1, 2] == pytest.approx(32.0)  # i=3,j=2
    np.testing.assert_allclose(run.x, [0.0, 10.0, 20.0])
    np.testing.assert_allclose(run.y, [0.0, 10.0])


def test_from_simulate_result_requires_grid_vars():
    result = types.SimpleNamespace(t=np.array([0.0]), y=np.array([[1.0]]), vars=["scalar"])
    with pytest.raises(ValueError):
        LevelSetRun.from_simulate_result(result, dx=1.0)


def test_observed_reference_mask_and_validation():
    frac = np.array([[0.0, 0.6], [0.9, 0.3]])
    ref = ObservedReference(burned_fraction_final=frac, x=[0.0, 1.0], y=[0.0, 1.0])
    assert ref.burned_mask_final(0.5).tolist() == [[False, True], [True, False]]
    with pytest.raises(ValueError):
        ObservedReference(burned_fraction_final=frac, x=[0.0, 1.0, 2.0], y=[0.0, 1.0])
    with pytest.raises(ValueError):  # series without times
        ObservedReference(burned_fraction_final=frac, x=[0, 1], y=[0, 1],
                          burned_fraction_series=np.zeros((1, 2, 2)))


def test_regrid_identity_and_subset():
    frac = np.arange(12, dtype=float).reshape(3, 4)
    ref = ObservedReference(burned_fraction_final=frac, x=[0.0, 1.0, 2.0, 3.0],
                            y=[0.0, 1.0, 2.0])
    # identity: same grid returns the same object
    assert ref.regrid_to(ref.x, ref.y) is ref
    # nearest-neighbour onto a coarser grid picks the nearest source cells
    sub = ref.regrid_to(np.array([0.1, 2.9]), np.array([0.0, 2.1]))
    assert sub.burned_fraction_final.shape == (2, 2)
    # nearest x to 0.1 -> col 0; to 2.9 -> col 3; nearest y to 0.0->row0, 2.1->row2
    assert sub.burned_fraction_final[0, 0] == frac[0, 0]
    assert sub.burned_fraction_final[0, 1] == frac[0, 3]
    assert sub.burned_fraction_final[1, 1] == frac[2, 3]


def test_nearest_index_midpoint_rounding():
    src = np.array([0.0, 10.0, 20.0])
    idx = D._nearest_index(src, np.array([-5.0, 4.0, 6.0, 100.0]))
    # clamp low, <5 -> 0, >5 -> 1, clamp high
    assert idx.tolist() == [0, 0, 1, 2]
