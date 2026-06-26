"""Unit tests for the metric primitives — each against a hand-computed answer."""
import math

import numpy as np
import pytest

from simulations.validation import metrics as M


def test_burned_and_fractional_area():
    mask = np.array([[True, False], [True, True]])
    assert M.burned_area(mask, 4.0) == pytest.approx(12.0)  # 3 cells * 4 m^2
    frac = np.array([[0.5, 0.0], [1.0, 0.25]])
    assert M.fractional_area(frac, 4.0) == pytest.approx((0.5 + 1.0 + 0.25) * 4.0)
    # fractions are clipped into [0, 1] before weighting
    assert M.fractional_area(np.array([[2.0, -1.0]]), 1.0) == pytest.approx(1.0)


def test_equivalent_radius_inverts_circle_area():
    r = 123.4
    assert M.equivalent_radius(math.pi * r * r) == pytest.approx(r)
    assert M.equivalent_radius(-5.0) == 0.0  # negative area clamped


def test_centroid_known_and_empty():
    xg, yg = np.meshgrid([0.0, 10.0, 20.0], [0.0, 10.0], indexing="xy")
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 0] = mask[0, 2] = True  # (0,0) and (20,0)
    assert M.centroid(mask, xg, yg) == (10.0, 0.0)
    assert M.centroid(np.zeros((2, 3), bool), xg, yg) is None


def test_overlap_identical_and_disjoint():
    a = np.array([[True, True], [False, False]])
    same = M.overlap_metrics(a, a, 1.0)
    assert same["iou"] == 1.0 and same["dice"] == 1.0
    assert same["precision"] == 1.0 and same["recall"] == 1.0
    assert same["false_positive_cells"] == 0 and same["false_negative_cells"] == 0

    b = ~a
    disj = M.overlap_metrics(a, b, 1.0)
    assert disj["iou"] == 0.0 and disj["dice"] == 0.0
    assert disj["precision"] == 0.0 and disj["recall"] == 0.0


def test_overlap_partial_hand_counts():
    # sim burns the top row (2 cells); obs burns the left column (2 cells);
    # they share exactly the top-left cell.
    sim = np.array([[True, True], [False, False]])
    obs = np.array([[True, False], [True, False]])
    m = M.overlap_metrics(sim, obs, 10.0)
    assert m["true_positive_cells"] == 1
    assert m["false_positive_cells"] == 1
    assert m["false_negative_cells"] == 1
    assert m["iou"] == pytest.approx(1 / 3)
    assert m["dice"] == pytest.approx(2 / 4)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["intersection_area_m2"] == pytest.approx(10.0)
    assert m["symmetric_difference_area_m2"] == pytest.approx(20.0)


def test_overlap_subset_precision_one():
    obs = np.ones((3, 3), dtype=bool)
    sim = np.zeros((3, 3), dtype=bool)
    sim[0, :] = True  # 3 of 9 truly-burned cells
    m = M.overlap_metrics(sim, obs, 1.0)
    assert m["precision"] == 1.0
    assert m["recall"] == pytest.approx(3 / 9)
    assert m["iou"] == pytest.approx(3 / 9)


def test_overlap_both_empty_is_perfect():
    z = np.zeros((2, 2), dtype=bool)
    m = M.overlap_metrics(z, z, 1.0)
    assert m["iou"] == 1.0 and m["dice"] == 1.0
    assert math.isnan(m["precision"]) and math.isnan(m["recall"])


def test_overlap_shape_mismatch_raises():
    with pytest.raises(ValueError):
        M.overlap_metrics(np.zeros((2, 2), bool), np.zeros((2, 3), bool), 1.0)


def test_area_metrics_and_zero_observed():
    m = M.area_metrics(150.0, 100.0)
    assert m["area_ratio"] == pytest.approx(1.5)
    assert m["area_bias_m2"] == pytest.approx(50.0)
    assert m["area_abs_pct_error"] == pytest.approx(50.0)

    z = M.area_metrics(5.0, 0.0)
    assert math.isinf(z["area_ratio"])
    assert math.isnan(z["area_abs_pct_error"])


def test_mean_spread_rate():
    a0 = math.pi * 100.0**2
    a1 = math.pi * 200.0**2
    assert M.mean_spread_rate(a0, a1, 100.0) == pytest.approx((200.0 - 100.0) / 100.0)
    assert math.isnan(M.mean_spread_rate(a0, a1, 0.0))


def test_distance_3_4_5():
    assert M.distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


def test_series_error():
    s = M.series_error(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert s["bias"] == pytest.approx((-2.0) / 3)
    assert s["rmse"] == pytest.approx(math.sqrt(4.0 / 3))
    assert s["n"] == 3
    empty = M.series_error(np.array([]), np.array([]))
    assert empty["n"] == 0 and math.isnan(empty["rmse"])
