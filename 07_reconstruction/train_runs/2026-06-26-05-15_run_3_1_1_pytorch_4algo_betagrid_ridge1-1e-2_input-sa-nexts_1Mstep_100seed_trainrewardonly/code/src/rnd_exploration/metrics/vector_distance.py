"""
Distance metrics between two vectors GT (ground truth) and Pred (prediction).
All functions take (gt, pred) as 1D arrays and return a scalar distance (float).
Assumes scipy is available for min_c_l1_diff.
"""
import numpy as np
from scipy.optimize import minimize_scalar


def min_c_l1_diff(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    min_c || GT - c * Pred ||_1.
    Finds scalar c that minimizes L1 distance; returns the minimum L1 value.
    """
    if np.all(pred == 0):
        return np.sum(np.abs(gt))

    def objective(c: float) -> float:
        return float(np.sum(np.abs(gt - c * pred)))

    res = minimize_scalar(objective, method="bounded", bounds=(-1e10, 1e10))
    return float(res.fun)


def min_c_l2_diff(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    min_c || GT - c * Pred ||_2.
    Closed-form c* = (GT·Pred) / (Pred·Pred); returns the minimum L2 value.
    """
    pred_sq = np.dot(pred, pred)
    if pred_sq == 0:
        return float(np.linalg.norm(gt, 2))
    c_opt = np.dot(gt, pred) / pred_sq
    return float(np.linalg.norm(gt - c_opt * pred, 2))


def _regularized_inv(gt: np.ndarray, eps: float) -> np.ndarray:
    """Element-wise 1/GT with |GT| clipped to >= eps to avoid division by zero."""
    gt_safe = np.where(np.abs(gt) >= eps, gt, np.copysign(eps, gt))
    # copysign(eps, 0) is eps, so no zero divisor
    return gt_safe


def min_c_l1_inv(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    min_c || c * 1 - GT^{-1} * Pred ||_1.
    GT^{-1} is element-wise inverse; uses regularized inverse (|gt| clipped to >= eps) so zeros are safe.
    Optimal c is median(GT^{-1} * Pred).
    """
    gt_inv = 1.0 / _regularized_inv(gt, eps)
    z = gt_inv * pred
    c_opt = np.median(z)
    return float(np.sum(np.abs(c_opt - z)))


def min_c_l2_inv(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    min_c || c * 1 - GT^{-1} * Pred ||_2.
    GT^{-1} is element-wise inverse; uses regularized inverse (|gt| clipped to >= eps) so zeros are safe.
    Optimal c = mean(GT^{-1} * Pred).
    """
    gt_inv = 1.0 / _regularized_inv(gt, eps)
    z = gt_inv * pred
    c_opt = np.mean(z)
    return float(np.linalg.norm(c_opt - z, 2))


def normalized_l2(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-10) -> float:
    """
    || Normalized GT - Normalized Pred ||_2.
    Each vector is normalized by dividing by its largest absolute value (so max absolute value becomes 1).
    """
    m_gt = np.max(np.abs(gt))
    m_pred = np.max(np.abs(pred))
    norm_gt = gt / (m_gt + eps)
    norm_pred = pred / (m_pred + eps)
    return float(np.linalg.norm(norm_gt - norm_pred, 2))


def normalized_angle_rad(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-10) -> float:
    """
    Angle (radians) between unit-normalized GT and unit-normalized Pred.
    Unit vector = vector / ||vector||_2. Returns theta in [0, pi].
    """
    n_gt = np.linalg.norm(gt, 2) + eps
    n_pred = np.linalg.norm(pred, 2) + eps
    u_gt = gt / n_gt
    u_pred = pred / n_pred
    cos_theta = np.clip(np.dot(u_gt, u_pred), -1.0, 1.0)
    return float(np.arccos(cos_theta))
