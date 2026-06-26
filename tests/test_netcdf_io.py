"""Reading an observed reference from NetCDF — the E1 -> E2 file handoff.

Skipped when netCDF4 is unavailable; the rest of the harness never needs it.
"""
import datetime as dt

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from simulations.validation import ObservedReference, validate, synthetic_run


def _write_reference(path, frac, x, y, times=None):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("x", len(x))
        ds.createDimension("y", len(y))
        ds.createVariable("x", "f8", ("x",))[:] = x
        ds.createVariable("y", "f8", ("y",))[:] = y
        if times is None:
            ds.createVariable("burned_fraction", "f8", ("y", "x"))[:] = frac
        else:
            ds.createDimension("time", len(times))
            tv = ds.createVariable("time", "f8", ("time",))
            tv.units = "hours since 2018-11-08 14:30:00"
            tv.calendar = "standard"
            tv[:] = times
            ds.createVariable("burned_fraction", "f8", ("time", "y", "x"))[:] = frac


def test_from_netcdf_static(tmp_path):
    path = tmp_path / "static.nc"
    frac = np.array([[0.0, 1.0], [0.5, 0.2]])
    _write_reference(path, frac, x=[0.0, 2000.0], y=[0.0, 2000.0])
    ref = ObservedReference.from_netcdf(path)
    np.testing.assert_allclose(ref.burned_fraction_final, frac)
    assert ref.perimeter_times is None
    assert ref.cell_area == pytest.approx(2000.0 * 2000.0)


def test_from_netcdf_timeseries_and_validate(tmp_path):
    path = tmp_path / "series.nc"
    nx = ny = 5
    x = np.arange(nx) * 2000.0
    y = np.arange(ny) * 2000.0
    # two daily footprints, growing
    series = np.zeros((2, ny, nx))
    series[0, 2, 2] = 1.0
    series[1, 1:4, 1:4] = 1.0
    _write_reference(path, series, x=x, y=y, times=[0.0, 24.0])
    ref = ObservedReference.from_netcdf(path)
    assert ref.burned_fraction_series.shape == (2, ny, nx)
    assert ref.perimeter_times[0] == dt.datetime(2018, 11, 8, 14, 30, tzinfo=dt.timezone.utc)
    assert ref.perimeter_times[1] == dt.datetime(2018, 11, 9, 14, 30, tzinfo=dt.timezone.utc)
    # the final footprint is the last slice
    np.testing.assert_allclose(ref.burned_fraction_final, series[1])

    # and it drives validate() without error (grids differ -> internal regrid)
    run = synthetic_run(n=9, dx=2000.0)
    report = validate(run, ref)
    assert report.metadata["reference_source"].endswith("series.nc")
    assert "iou" in report.burned_area
