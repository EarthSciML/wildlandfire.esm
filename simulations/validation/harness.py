"""Top-level validation: compare a :class:`LevelSetRun` to an
:class:`ObservedReference` and assemble a :class:`ValidationReport`.

``validate()`` is the one entry point a caller needs. It computes the three
metric families the campaign asks for — burned-area agreement, perimeter-over-
time / spread-rate, and ignition match — aligning the observed reference onto the
run grid first. The ``synthetic_*`` builders construct a known fire so the whole
pipeline is exercisable (and testable) without the live E1 data or E3 run.
"""
from __future__ import annotations

import datetime as _dt
import math
from typing import Optional

import numpy as np

from .data import LevelSetRun, ObservedReference, mesh
from .metrics import (
    area_metrics,
    burned_area,
    centroid,
    distance,
    equivalent_radius,
    mean_spread_rate,
    overlap_metrics,
    series_error,
)
from .report import (
    NA,
    OK,
    WARN,
    MetricVerdict,
    ValidationConfig,
    ValidationReport,
)

_M2_PER_KM2 = 1.0e6
# Tolerance (s) for treating an observed perimeter time as inside the run window.
_WINDOW_EPS_S = 1.0


def validate(
    run: LevelSetRun,
    reference: ObservedReference,
    config: Optional[ValidationConfig] = None,
    *,
    title: str = "Camp Fire validation",
) -> ValidationReport:
    """Validate a simulated level-set run against an observed reference.

    The observed reference is nearest-neighbour resampled onto the run grid, then
    three metric families are computed. Each populates a report section and
    contributes one or more advisory verdicts. Metric families with no input
    (no perimeter series, no observed ignition) are reported as ``skipped`` and
    their verdicts as ``n/a`` rather than dropped, so the report shape is stable.
    """
    config = config or ValidationConfig()
    ref = reference.regrid_to(run.x, run.y)
    ca = run.cell_area
    xg, yg = mesh(run.x, run.y)
    verdicts: list[MetricVerdict] = []

    burned = _burned_area_section(run, ref, config, ca, verdicts)
    timing = _perimeter_timing_section(run, ref, config, ca, verdicts)
    ignition = _ignition_section(run, ref, config, xg, yg, verdicts)

    metadata = {
        "reference_source": ref.source,
        "grid": {
            "nx": int(run.x.size),
            "ny": int(run.y.size),
            "cell_size_m": float(math.sqrt(ca)),
            "cell_area_km2": ca / _M2_PER_KM2,
            "extent_m": [
                float(run.x.min()),
                float(run.x.max()),
                float(run.y.min()),
                float(run.y.max()),
            ],
        },
        "run_window": {
            "t0": run.t0.astimezone(_dt.timezone.utc).isoformat() if run.t0 else None,
            "duration_hours": float((run.times[-1] - run.times[0]) / 3600.0),
            "n_times": int(run.n_times),
        },
    }

    return ValidationReport(
        title=title,
        metadata=metadata,
        burned_area=burned,
        perimeter_timing=timing,
        ignition=ignition,
        verdicts=verdicts,
        config=config.to_dict(),
    )


def _burned_area_section(run, ref, config, ca, verdicts) -> dict:
    sim_mask = run.burned_mask(run.n_times - 1, config.levelset_threshold)
    obs_mask = ref.burned_mask_final(config.fraction_threshold)
    overlap = overlap_metrics(sim_mask, obs_mask, ca)
    am = area_metrics(burned_area(sim_mask, ca), burned_area(obs_mask, ca))

    section = {
        "area_sim_km2": am["area_sim_m2"] / _M2_PER_KM2,
        "area_obs_km2": am["area_obs_m2"] / _M2_PER_KM2,
        "area_ratio": am["area_ratio"],
        "area_bias_km2": am["area_bias_m2"] / _M2_PER_KM2,
        "area_abs_pct_error": am["area_abs_pct_error"],
        "iou": overlap["iou"],
        "dice": overlap["dice"],
        "precision": overlap["precision"],
        "recall": overlap["recall"],
        "intersection_km2": overlap["intersection_area_m2"] / _M2_PER_KM2,
        "symmetric_difference_km2": overlap["symmetric_difference_area_m2"] / _M2_PER_KM2,
    }

    lo, hi = config.area_ratio_ok
    ratio = am["area_ratio"]
    ratio_ok = math.isfinite(ratio) and lo <= ratio <= hi
    verdicts.append(
        MetricVerdict("burned_area_ratio", OK if ratio_ok else WARN, ratio, f"{lo}–{hi}")
    )
    verdicts.append(
        MetricVerdict(
            "final_footprint_dice",
            OK if overlap["dice"] >= config.dice_ok else WARN,
            overlap["dice"],
            f">= {config.dice_ok}",
        )
    )
    return section


def _perimeter_timing_section(run, ref, config, ca, verdicts) -> dict:
    if ref.burned_fraction_series is None or ref.perimeter_times is None:
        verdicts.append(
            MetricVerdict("area_vs_time_rmse_km2", NA, None, "—", "no observed perimeter series")
        )
        return {"status": "skipped", "reason": "no observed perimeter time series"}
    if run.t0 is None:
        verdicts.append(
            MetricVerdict("area_vs_time_rmse_km2", NA, None, "—", "run has no t0 for time alignment")
        )
        return {"status": "skipped", "reason": "run has no t0 to align timestamps"}

    t_lo, t_hi = float(run.times[0]), float(run.times[-1])
    per_time = []
    sim_areas_km2, obs_areas_km2, ious, elapsed_list = [], [], [], []
    for when, frac in zip(ref.perimeter_times, ref.burned_fraction_series):
        elapsed = (when - run.t0).total_seconds()
        if elapsed < t_lo - _WINDOW_EPS_S or elapsed > t_hi + _WINDOW_EPS_S:
            continue
        sim_mask = run.mask_at(elapsed, config.levelset_threshold)
        obs_mask = frac >= config.fraction_threshold
        ov = overlap_metrics(sim_mask, obs_mask, ca)
        sa = burned_area(sim_mask, ca) / _M2_PER_KM2
        oa = burned_area(obs_mask, ca) / _M2_PER_KM2
        per_time.append(
            {
                "time": when.astimezone(_dt.timezone.utc).isoformat(),
                "elapsed_hours": elapsed / 3600.0,
                "area_sim_km2": sa,
                "area_obs_km2": oa,
                "iou": ov["iou"],
            }
        )
        sim_areas_km2.append(sa)
        obs_areas_km2.append(oa)
        ious.append(ov["iou"])
        elapsed_list.append(elapsed)

    n_available = len(ref.perimeter_times)
    if not per_time:
        verdicts.append(
            MetricVerdict("area_vs_time_rmse_km2", NA, None, "—", "no perimeter time in run window")
        )
        return {
            "status": "skipped",
            "reason": "no observed perimeter time falls within the run window",
            "n_available": n_available,
        }

    se = series_error(np.array(sim_areas_km2), np.array(obs_areas_km2))
    spread_sim = spread_obs = float("nan")
    if len(elapsed_list) >= 2:
        dt = elapsed_list[-1] - elapsed_list[0]
        spread_sim = mean_spread_rate(
            sim_areas_km2[0] * _M2_PER_KM2, sim_areas_km2[-1] * _M2_PER_KM2, dt
        )
        spread_obs = mean_spread_rate(
            obs_areas_km2[0] * _M2_PER_KM2, obs_areas_km2[-1] * _M2_PER_KM2, dt
        )

    mean_iou = float(np.mean(ious))
    verdicts.append(
        MetricVerdict(
            "mean_perimeter_iou",
            OK if mean_iou >= config.dice_ok else WARN,
            mean_iou,
            f">= {config.dice_ok}",
        )
    )
    return {
        "status": "computed",
        "n_available": n_available,
        "n_matched": len(per_time),
        "area_rmse_km2": se["rmse"],
        "area_bias_km2": se["bias"],
        "mean_iou": mean_iou,
        "spread_rate_sim_m_per_s": spread_sim,
        "spread_rate_obs_m_per_s": spread_obs,
        "per_time": per_time,
    }


def _ignition_section(run, ref, config, xg, yg, verdicts) -> dict:
    if ref.ignition_xy is None:
        verdicts.append(
            MetricVerdict("ignition_distance_km", NA, None, "—", "no observed ignition point")
        )
        return {"status": "skipped", "reason": "no observed ignition point"}

    sim_mask0 = run.burned_mask(0, config.levelset_threshold)
    sim_xy = centroid(sim_mask0, xg, yg)
    if sim_xy is None:
        verdicts.append(
            MetricVerdict("ignition_distance_km", NA, None, "—", "no simulated burn at t0")
        )
        return {"status": "skipped", "reason": "simulated run has no burned cells at the first step"}

    dist_m = distance(sim_xy, ref.ignition_xy)
    verdicts.append(
        MetricVerdict(
            "ignition_distance_km",
            OK if dist_m <= config.ignition_distance_ok_m else WARN,
            dist_m / 1000.0,
            f"<= {config.ignition_distance_ok_m / 1000.0} km",
        )
    )

    offset_hours = None
    if run.t0 is not None and ref.ignition_time is not None:
        offset_hours = (run.t0 - ref.ignition_time).total_seconds() / 3600.0
        verdicts.append(
            MetricVerdict(
                "ignition_time_offset_hours",
                OK if abs(offset_hours) <= config.ignition_time_ok_hours else WARN,
                offset_hours,
                f"<= {config.ignition_time_ok_hours} h (abs)",
            )
        )

    return {
        "status": "computed",
        "distance_km": dist_m / 1000.0,
        "time_offset_hours": offset_hours,
        "sim_xy": [sim_xy[0], sim_xy[1]],
        "obs_xy": [ref.ignition_xy[0], ref.ignition_xy[1]],
    }


# --------------------------------------------------------------------------
# Synthetic builders — a known expanding-disk fire for demos and tests.
# --------------------------------------------------------------------------
def _disk_grid(n: int, dx: float, origin: float) -> np.ndarray:
    return origin + dx * np.arange(n)


def synthetic_run(
    *,
    n: int = 61,
    dx: float = 2000.0,
    center: tuple[float, float] = (0.0, 0.0),
    r0: float = 3000.0,
    rate: float = 0.5,
    duration_h: float = 16.0,
    n_times: int = 9,
    t0: Optional[_dt.datetime] = None,
) -> LevelSetRun:
    """An expanding circular fire: ``psi = |x - c| - (r0 + rate·t)``.

    Burned region is an exact disk of radius ``r0 + rate·t`` so areas
    (``π r²``) and overlaps are analytically known — the backbone of the tests.
    Defaults mimic the Camp Fire domain (≈120 km box at 2 km, 16 h window).
    """
    if t0 is None:
        t0 = _dt.datetime(2018, 11, 8, 14, 30, tzinfo=_dt.timezone.utc)
    half = (n - 1) * dx / 2.0
    x = _disk_grid(n, dx, center[0] - half)
    y = _disk_grid(n, dx, center[1] - half)
    xg, yg = mesh(x, y)
    dist = np.hypot(xg - center[0], yg - center[1])
    times = np.linspace(0.0, duration_h * 3600.0, n_times)
    psi = np.stack([dist - (r0 + rate * t) for t in times], axis=0)
    return LevelSetRun(psi=psi, times=times, x=x, y=y, t0=t0)


def synthetic_reference(
    run: LevelSetRun,
    *,
    radius_scale: float = 1.0,
    center_offset: tuple[float, float] = (0.0, 0.0),
    with_series: bool = True,
    ignition_offset: tuple[float, float] = (0.0, 0.0),
    ignition_time_offset_h: float = 0.0,
) -> ObservedReference:
    """An observed disk fire derived from ``run`` for end-to-end demos/tests.

    ``radius_scale`` / ``center_offset`` perturb the observed footprint relative
    to the simulated one so the metrics exercise real (non-degenerate) overlaps;
    the defaults reproduce the run exactly (perfect agreement).
    """
    x, y = run.x, run.y
    xg, yg = mesh(x, y)
    # Recover the run's disk geometry from its first/last fields.
    c = (
        float(xg[run.burned_mask(0)].mean()) if run.burned_mask(0).any() else 0.0,
        float(yg[run.burned_mask(0)].mean()) if run.burned_mask(0).any() else 0.0,
    )
    oc = (c[0] + center_offset[0], c[1] + center_offset[1])

    def disk_fraction(radius: float) -> np.ndarray:
        return (np.hypot(xg - oc[0], yg - oc[1]) <= radius).astype(float)

    # Final observed radius from the run's last burned area, scaled.
    final_area = burned_area(run.burned_mask(run.n_times - 1), run.cell_area)
    final_radius = equivalent_radius(final_area) * radius_scale
    final = disk_fraction(final_radius)

    perimeter_times = None
    series = None
    if with_series and run.t0 is not None:
        perimeter_times = [run.datetime_at(t) for t in run.times]
        series = np.stack(
            [
                disk_fraction(
                    equivalent_radius(burned_area(run.burned_mask(k), run.cell_area)) * radius_scale
                )
                for k in range(run.n_times)
            ],
            axis=0,
        )

    ignition_time = None
    if run.t0 is not None:
        ignition_time = run.t0 - _dt.timedelta(hours=ignition_time_offset_h)
    return ObservedReference(
        burned_fraction_final=final,
        x=x,
        y=y,
        perimeter_times=perimeter_times,
        burned_fraction_series=series,
        ignition_xy=(c[0] + ignition_offset[0], c[1] + ignition_offset[1]),
        ignition_time=ignition_time,
        source="synthetic",
    )


def demo_report(config: Optional[ValidationConfig] = None) -> ValidationReport:
    """A complete report on a synthetic, mildly-imperfect Camp-Fire-like run."""
    run = synthetic_run()
    ref = synthetic_reference(
        run,
        radius_scale=1.12,
        center_offset=(1500.0, -1000.0),
        ignition_offset=(800.0, 600.0),
        ignition_time_offset_h=1.5,
    )
    return validate(run, ref, config, title="Camp Fire validation (synthetic demo)")
