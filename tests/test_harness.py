"""End-to-end tests for validate(): synthetic run vs reference, verdict logic."""
import numpy as np
import pytest

from simulations.validation import (
    ObservedReference,
    demo_report,
    synthetic_reference,
    synthetic_run,
    validate,
)


def _verdict(report, name):
    for v in report.verdicts:
        if v.metric == name:
            return v
    raise AssertionError(f"no verdict named {name}")


def test_perfect_agreement_when_reference_is_the_run():
    run = synthetic_run(n=41)
    # Build a reference whose final + series masks ARE the run's masks exactly.
    series = np.stack([run.burned_mask(k).astype(float) for k in range(run.n_times)])
    ref = ObservedReference(
        burned_fraction_final=run.burned_mask(run.n_times - 1).astype(float),
        x=run.x,
        y=run.y,
        perimeter_times=[run.datetime_at(t) for t in run.times],
        burned_fraction_series=series,
        ignition_xy=(0.0, 0.0),
        ignition_time=run.t0,
    )
    report = validate(run, ref)
    assert report.burned_area["iou"] == pytest.approx(1.0)
    assert report.burned_area["dice"] == pytest.approx(1.0)
    assert report.burned_area["area_ratio"] == pytest.approx(1.0)
    assert report.ignition["distance_km"] == pytest.approx(0.0)
    assert report.perimeter_timing["mean_iou"] == pytest.approx(1.0)
    assert report.overall_status() == "ok"


def test_area_ratio_tracks_radius_scale():
    run = synthetic_run(n=61)
    ref = synthetic_reference(run, radius_scale=1.25, with_series=False)
    report = validate(run, ref)
    # observed disk is 1.25x the radius -> ~1.25^2 the area -> ratio ~ 1/1.5625
    assert report.burned_area["area_ratio"] == pytest.approx(1 / 1.25**2, rel=0.05)
    # simulated disk sits inside the larger observed disk -> precision ~ 1
    assert report.burned_area["precision"] == pytest.approx(1.0, abs=0.02)


def test_large_offset_flags_dice_warn():
    run = synthetic_run(n=61, r0=3000.0, rate=0.0)  # static small disk
    # shove the observed footprint far away -> poor overlap
    ref = synthetic_reference(run, center_offset=(40000.0, 0.0), with_series=False)
    report = validate(run, ref)
    assert report.burned_area["dice"] < 0.3
    assert _verdict(report, "final_footprint_dice").status == "warn"
    assert report.overall_status() == "warn"


def test_ignition_distance_and_time_verdicts():
    run = synthetic_run(n=41)
    ref = synthetic_reference(
        run, with_series=False, ignition_offset=(9000.0, 0.0), ignition_time_offset_h=10.0
    )
    report = validate(run, ref)
    assert report.ignition["distance_km"] == pytest.approx(9.0, rel=0.01)
    assert _verdict(report, "ignition_distance_km").status == "warn"  # > 5 km default
    assert report.ignition["time_offset_hours"] == pytest.approx(10.0)
    assert _verdict(report, "ignition_time_offset_hours").status == "warn"  # > 6 h default


def test_perimeter_timing_skipped_without_series():
    run = synthetic_run(n=41)
    ref = synthetic_reference(run, with_series=False)
    report = validate(run, ref)
    assert report.perimeter_timing["status"] == "skipped"
    assert _verdict(report, "area_vs_time_rmse_km2").status == "n/a"


def test_ignition_skipped_without_observed_point():
    run = synthetic_run(n=41)
    ref = ObservedReference(
        burned_fraction_final=run.burned_mask(run.n_times - 1).astype(float),
        x=run.x,
        y=run.y,
    )
    report = validate(run, ref)
    assert report.ignition["status"] == "skipped"
    assert _verdict(report, "ignition_distance_km").status == "n/a"


def test_regrid_path_when_reference_on_coarser_grid():
    run = synthetic_run(n=61)
    fine_ref = synthetic_reference(run, with_series=False)
    # Re-sample the reference onto a coarser grid, then let validate() regrid back.
    coarse_x = run.x[::2]
    coarse_y = run.y[::2]
    coarse = fine_ref.regrid_to(coarse_x, coarse_y)
    assert coarse.x.size < run.x.size
    report = validate(run, coarse)  # regrids coarse -> run grid internally
    assert report.burned_area["iou"] > 0.9  # near-perfect despite the grid hop


def test_spread_rate_is_positive_for_growing_fire():
    run = synthetic_run(n=61, rate=0.5)
    ref = synthetic_reference(run, radius_scale=1.0)
    report = validate(run, ref)
    pt = report.perimeter_timing
    assert pt["spread_rate_sim_m_per_s"] > 0
    assert pt["spread_rate_obs_m_per_s"] > 0
    # synthetic front advances at 0.5 m/s; equivalent-radius rate recovers it closely
    assert pt["spread_rate_sim_m_per_s"] == pytest.approx(0.5, abs=0.05)


def test_demo_report_runs_and_is_ok():
    report = demo_report()
    assert report.overall_status() == "ok"
    assert report.metadata["grid"]["nx"] == 61
