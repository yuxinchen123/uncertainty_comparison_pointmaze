"""Tests for vector_distance metrics (GT vs Pred)."""
import numpy as np
import pytest

from rnd_exploration.metrics.vector_distance import (
    min_c_l1_diff,
    min_c_l2_diff,
    min_c_l1_inv,
    min_c_l2_inv,
    normalized_l2,
    normalized_angle_rad,
)


def test_min_c_l2_diff_closed_form():
    # min_c ||GT - c*Pred||_2 => c* = (GT·Pred)/(Pred·Pred)
    # When Pred = k*GT, c* = k, distance = 0
    gt = np.array([1.0, 2.0, 3.0])
    pred = 2.0 * gt
    d = min_c_l2_diff(gt, pred)
    assert d == pytest.approx(0.0, abs=1e-10)
    # Orthogonal: GT·Pred=0 => c*=0 => distance = ||GT||_2
    pred_orth = np.array([1.0, -0.5, 0.0])
    np.testing.assert_allclose(np.dot(gt, pred_orth), 0.0, atol=1e-10)
    d_orth = min_c_l2_diff(gt, pred_orth)
    assert d_orth == pytest.approx(np.linalg.norm(gt, 2), abs=1e-10)
    # General: c* = dot(gt,pred)/dot(pred,pred), then ||gt - c*pred||_2
    gt = np.array([1.0, 0.0, 1.0])
    pred = np.array([1.0, 1.0, 0.0])
    c_opt = np.dot(gt, pred) / np.dot(pred, pred)
    expected = np.linalg.norm(gt - c_opt * pred, 2)
    assert min_c_l2_diff(gt, pred) == pytest.approx(expected, abs=1e-10)


def test_min_c_l1_diff_exact_scale():
    # When Pred = c0*GT, optimal c = c0 gives distance 0
    gt = np.array([1.0, 2.0, 3.0])
    c0 = 2.5
    pred = c0 * gt
    d = min_c_l1_diff(gt, pred)
    assert d == pytest.approx(0.0, abs=1e-4)


def test_min_c_l1_diff_vs_direct():
    # Compare to evaluating at known optimal (for L1, optimal c is weighted median)
    gt = np.array([1.0, 2.0, 4.0])
    pred = np.array([1.0, 1.0, 1.0])
    # min_c |1-c| + |2-c| + |4-c| => c* = median([1,2,4]) = 2
    c_median = 2.0
    expected = np.sum(np.abs(gt - c_median * pred))
    assert min_c_l1_diff(gt, pred) == pytest.approx(expected, abs=1e-4)


def test_min_c_l2_inv_closed_form():
    # min_c ||c*1 - GT^{-1}*Pred||_2 => c* = mean(Pred/GT) on safe indices
    gt = np.array([1.0, 2.0, 4.0])
    pred = np.array([2.0, 4.0, 8.0])  # pred = 2*gt => pred/gt = 2 everywhere
    z = pred / gt
    c_opt = np.mean(z)
    assert c_opt == pytest.approx(2.0, abs=1e-10)
    expected = np.linalg.norm(c_opt - z, 2)
    assert min_c_l2_inv(gt, pred) == pytest.approx(expected, abs=1e-10)
    assert min_c_l2_inv(gt, pred) == pytest.approx(0.0, abs=1e-10)


def test_min_c_l1_inv_median():
    # min_c ||c*1 - z||_1 => c* = median(z)
    gt = np.array([1.0, 1.0, 1.0])
    pred = np.array([1.0, 2.0, 3.0])
    z = pred / gt
    c_opt = np.median(z)
    expected = np.sum(np.abs(c_opt - z))
    assert min_c_l1_inv(gt, pred) == pytest.approx(expected, abs=1e-10)


def test_normalized_l2_scale_invariant():
    # Normalize by max abs: [1,2,3] -> [1/3, 2/3, 1], so same shape => distance 0 for proportional
    gt = np.array([1.0, 2.0, 3.0])
    pred = 10.0 * gt  # same shape after norm
    d = normalized_l2(gt, pred)
    assert d == pytest.approx(0.0, abs=1e-10)


def test_normalized_l2_known():
    gt = np.array([2.0, 4.0])   # max abs 4 => [0.5, 1]
    pred = np.array([0.0, 4.0]) # max abs 4 => [0, 1]
    norm_gt = gt / 4.0
    norm_pred = pred / 4.0
    expected = np.linalg.norm(norm_gt - norm_pred, 2)
    assert normalized_l2(gt, pred) == pytest.approx(expected, abs=1e-10)
    assert expected == pytest.approx(0.5, abs=1e-10)  # ||[0.5,0]||_2 = 0.5


def test_normalized_angle_rad_parallel():
    # Same direction => angle 0
    gt = np.array([1.0, 0.0, 0.0])
    pred = np.array([10.0, 0.0, 0.0])
    assert normalized_angle_rad(gt, pred) == pytest.approx(0.0, abs=2e-5)


def test_normalized_angle_rad_orthogonal():
    # Orthogonal => pi/2
    gt = np.array([1.0, 0.0, 0.0])
    pred = np.array([0.0, 1.0, 0.0])
    assert normalized_angle_rad(gt, pred) == pytest.approx(np.pi / 2, abs=1e-10)


def test_normalized_angle_rad_opposite():
    # Opposite => pi
    gt = np.array([1.0, 0.0, 0.0])
    pred = np.array([-1.0, 0.0, 0.0])
    assert normalized_angle_rad(gt, pred) == pytest.approx(np.pi, abs=3e-5)


def test_min_c_l1_inv_robust_to_zeros():
    # Regularized inverse: no division by zero; returns finite value when gt has zeros
    gt = np.array([0.0, 1.0, 1.0])
    pred = np.array([0.0, 1.0, 1.0])
    d = min_c_l1_inv(gt, pred, eps=1e-8)
    assert np.isfinite(d)
    # When all |gt| > eps, matches median formula
    gt_safe = np.array([1.0, 2.0, 4.0])
    pred_safe = np.array([1.0, 2.0, 2.0])
    z = pred_safe / gt_safe
    expected = np.sum(np.abs(np.median(z) - z))
    assert min_c_l1_inv(gt_safe, pred_safe, eps=1e-8) == pytest.approx(expected, abs=1e-10)


def test_min_c_l2_diff_pred_zero():
    gt = np.array([1.0, 2.0])
    pred = np.array([0.0, 0.0])
    assert min_c_l2_diff(gt, pred) == pytest.approx(np.linalg.norm(gt, 2), abs=1e-10)


def test_min_c_l1_diff_pred_zero():
    gt = np.array([1.0, -2.0])
    pred = np.array([0.0, 0.0])
    assert min_c_l1_diff(gt, pred) == pytest.approx(3.0, abs=1e-10)
