"""Metric primitives for comparing a simulated fire footprint to observations.

Pure functions over boolean masks and coordinate arrays — no knowledge of the
:mod:`~simulations.validation.data` containers, so each is independently
testable against hand-computed answers. Areas are in m^2 (callers convert to
km^2 for reporting); distances in metres.

The three metric families the harness reports are assembled from these:

* **burned-area agreement** — :func:`overlap_metrics` + :func:`area_metrics`;
* **perimeter-over-time / spread-rate** — per-time :func:`overlap_metrics`
  plus :func:`equivalent_radius` differenced over the window;
* **ignition match** — :func:`centroid` distance + a timestamp offset.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def burned_area(mask: np.ndarray, cell_area: float) -> float:
    """Total burned area (m^2) of a boolean mask on a uniform grid."""
    return float(np.count_nonzero(mask)) * float(cell_area)


def fractional_area(fraction: np.ndarray, cell_area: float) -> float:
    """Burned area (m^2) from a fraction-of-cell grid, weighting partial cells."""
    return float(np.sum(np.clip(fraction, 0.0, 1.0))) * float(cell_area)


def equivalent_radius(area: float) -> float:
    """Radius (m) of the circle with the given area — a scale for a spread rate."""
    return math.sqrt(max(area, 0.0) / math.pi)


def centroid(mask: np.ndarray, xg: np.ndarray, yg: np.ndarray) -> Optional[tuple[float, float]]:
    """Area-centroid ``(x, y)`` of a boolean mask, or ``None`` if empty."""
    if not np.any(mask):
        return None
    return float(xg[mask].mean()), float(yg[mask].mean())


def overlap_metrics(
    mask_sim: np.ndarray,
    mask_obs: np.ndarray,
    cell_area: float,
) -> dict:
    """Confusion-matrix overlap of two boolean footprints.

    Returns counts and areas for true/false positives and false negatives, plus
    the standard set-overlap scores. Treating the *simulated* footprint as the
    prediction and the *observed* as truth:

    * ``iou`` (Jaccard) — intersection over union; the headline overlap score.
    * ``dice`` (Sørensen) — ``2·TP / (2·TP + FP + FN)``.
    * ``precision`` — fraction of the simulated burn that really burned.
    * ``recall`` — fraction of the real burn the simulation captured.

    Two empty footprints score a perfect ``iou``/``dice`` of 1.0 (nothing to
    disagree about); precision/recall are ``nan`` when their denominator is 0.
    """
    if mask_sim.shape != mask_obs.shape:
        raise ValueError(f"mask shape mismatch: {mask_sim.shape} vs {mask_obs.shape}")
    sim = mask_sim.astype(bool)
    obs = mask_obs.astype(bool)
    tp = int(np.count_nonzero(sim & obs))
    fp = int(np.count_nonzero(sim & ~obs))
    fn = int(np.count_nonzero(~sim & obs))
    tn = int(np.count_nonzero(~sim & ~obs))

    union = tp + fp + fn
    iou = 1.0 if union == 0 else tp / union
    dice_den = 2 * tp + fp + fn
    dice = 1.0 if dice_den == 0 else (2 * tp) / dice_den
    precision = float("nan") if (tp + fp) == 0 else tp / (tp + fp)
    recall = float("nan") if (tp + fn) == 0 else tp / (tp + fn)

    ca = float(cell_area)
    return {
        "true_positive_cells": tp,
        "false_positive_cells": fp,
        "false_negative_cells": fn,
        "true_negative_cells": tn,
        "intersection_area_m2": tp * ca,
        "union_area_m2": union * ca,
        "symmetric_difference_area_m2": (fp + fn) * ca,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }


def area_metrics(area_sim: float, area_obs: float) -> dict:
    """Total-area agreement: ratio, signed bias (m^2), absolute percent error."""
    bias = float(area_sim) - float(area_obs)
    ratio = float("inf") if area_obs == 0 else area_sim / area_obs
    abs_pct = float("nan") if area_obs == 0 else abs(bias) / area_obs * 100.0
    return {
        "area_sim_m2": float(area_sim),
        "area_obs_m2": float(area_obs),
        "area_ratio": ratio,
        "area_bias_m2": bias,
        "area_abs_pct_error": abs_pct,
    }


def mean_spread_rate(area_start: float, area_end: float, dt_seconds: float) -> float:
    """Mean radial spread rate (m/s): change in equivalent radius over ``dt``."""
    if dt_seconds <= 0:
        return float("nan")
    return (equivalent_radius(area_end) - equivalent_radius(area_start)) / dt_seconds


def distance(p: tuple[float, float], q: tuple[float, float]) -> float:
    """Euclidean distance (m) between two projected points."""
    return math.hypot(p[0] - q[0], p[1] - q[1])


def series_error(sim_values: np.ndarray, obs_values: np.ndarray) -> dict:
    """RMSE and signed mean bias between two aligned 1-D series (e.g. area-vs-time)."""
    sim = np.asarray(sim_values, dtype=float)
    obs = np.asarray(obs_values, dtype=float)
    if sim.shape != obs.shape:
        raise ValueError(f"series shape mismatch: {sim.shape} vs {obs.shape}")
    if sim.size == 0:
        return {"rmse": float("nan"), "bias": float("nan"), "n": 0}
    diff = sim - obs
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "bias": float(np.mean(diff)),
        "n": int(sim.size),
    }
