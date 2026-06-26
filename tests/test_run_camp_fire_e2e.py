"""Tests for the pure-kinematic parts of the E3 end-to-end driver.

The driver's assembly / loader / chain stages need the EarthSciSerialization
toolkit + EarthSciModels checkout (exercised by running the script). The
fire-progression and physical-sanity logic is pure numpy, so it imports and
tests without the toolkit — these guard the kinematic front model.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pytest

from simulations.run_camp_fire_e2e import (
    _length_to_breadth,
    fire_progression,
    physical_sanity,
)

T0 = _dt.datetime(2018, 11, 8, 14, 30, tzinfo=_dt.timezone.utc)


def _uniform_behavior(ny, nx, *, R=1.0, U=3.0, heading=(1.0, 0.0)):
    return {
        "R": np.full((ny, nx), float(R)),
        "U": np.full((ny, nx), float(U)),
        "hx": np.full((ny, nx), float(heading[0])),
        "hy": np.full((ny, nx), float(heading[1])),
    }


# -- length-to-breadth ------------------------------------------------------
def test_length_to_breadth_no_wind_is_circular():
    assert _length_to_breadth(0.0) == pytest.approx(1.0, abs=1e-9)


def test_length_to_breadth_monotone_and_clamped():
    vals = [_length_to_breadth(u) for u in (0, 2, 5, 10, 20, 40)]
    assert all(b <= a + 1e-9 for a, b in zip(vals[1:], vals[1:]))  # finite
    assert all(1.0 <= v <= 8.0 for v in vals)
    assert vals[1] > vals[0]  # rises with wind
    assert _length_to_breadth(100.0) == pytest.approx(8.0)  # clamp


# -- fire progression -------------------------------------------------------
def test_progression_shape_and_monotone_growth():
    ny, nx = 11, 13
    x = np.arange(nx) * 2000.0
    y = np.arange(ny) * 2000.0
    run = fire_progression(_uniform_behavior(ny, nx), x, y, (x[nx // 2], y[ny // 2]),
                           n_times=6, duration_h=4.0, t0=T0, verbose=False)
    assert run.psi.shape == (6, ny, nx)
    areas = [int(run.burned_mask(k).sum()) for k in range(run.n_times)]
    assert areas == sorted(areas)  # monotonically non-decreasing
    assert areas[0] >= 1  # the ignition cell is burning at t0
    assert run.t0 == T0


def test_progression_is_anisotropic_downwind():
    # Heading due east (+x); the front must reach farther east than west.
    ny, nx = 9, 21
    x = np.arange(nx) * 1000.0
    y = np.arange(ny) * 1000.0
    cx = nx // 2
    run = fire_progression(_uniform_behavior(ny, nx, R=2.0, U=6.0, heading=(1.0, 0.0)),
                           x, y, (x[cx], y[ny // 2]), n_times=4, duration_h=2.0,
                           t0=T0, verbose=False)
    burned = run.burned_mask(run.n_times - 1)
    row = burned[ny // 2]
    east_reach = (np.where(row)[0].max() - cx) if row.any() else 0
    west_reach = (cx - np.where(row)[0].min()) if row.any() else 0
    assert east_reach > west_reach  # elliptical, wind-driven


def test_progression_seeds_nearest_cell_when_between():
    # Ignition far outside the tiny grid: still seeds the nearest cell.
    ny, nx = 5, 5
    x = np.arange(nx) * 2000.0
    y = np.arange(ny) * 2000.0
    run = fire_progression(_uniform_behavior(ny, nx), x, y, (1.0e7, 1.0e7),
                           n_times=3, duration_h=2.0, t0=T0, verbose=False)
    assert run.burned_mask(0).sum() >= 1


# -- physical sanity --------------------------------------------------------
def test_physical_sanity_passes_for_a_spreading_fire():
    ny, nx = 21, 21
    x = np.arange(nx) * 2000.0
    y = np.arange(ny) * 2000.0
    ign = (x[nx // 2], y[ny // 2])
    run = fire_progression(_uniform_behavior(ny, nx, R=2.0, U=6.0), x, y, ign,
                           n_times=9, duration_h=16.0, t0=T0, verbose=False)
    checks = physical_sanity(run, _uniform_behavior(ny, nx, R=2.0, U=6.0), ign,
                             verbose=False)
    assert checks["fire spreads from ignition"]
    assert checks["burned area monotonically non-decreasing"]
    assert checks["extreme heading spread rate (> 30 m/min)"]
