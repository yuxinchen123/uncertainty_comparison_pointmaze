"""
Intrinsic reward vector for each observation in full_observation_list (discretized valid cells),
using a given IntrinsicRewardModel. Distance-to-GT for algorithms that do not need action.
"""
from typing import Any, Dict, Optional

import numpy as np

from .full_observation_list import full_observation_list
from . import vector_distance
from env_wrapper.point_maze_utils import observation_to_grid, velocity_to_grid


ALGORITHM_NAMES = [
    "no_exploration",                  # no intrinsic reward (beta=0)
    "gt_position",                      # position only
    "gt_position_velocity",             # position + velocity
    "rnd_next_state",                  # next_state
    "rnd_next_state_position_only",    # next_state
    "rnd_state",                       # state
    "rnd_state_action",                # state + action
    "rnd_state_action_next_state",      # state + action + next_state
    "rnd_linear_next_state",           # next_state
    "rnd_elliptical",                  # state + action
]

VALID_DISTANCE_ALGORITHMS = [
    "min_c_l1_diff",
    "min_c_l2_diff",
    "min_c_l1_inv",
    "min_c_l2_inv",
    "normalized_l2",
    "normalized_angle_rad",
]

# Algorithms whose intrinsic vector can be computed without actions (for distance-to-GT).
# GT is the visit count vector from PositionVelocityVisitCountWrapper (gt_position_velocity).
ALGORITHMS_NO_ACTION = [
    "no_exploration",
    "gt_position",
    "gt_position_velocity",
    "rnd_next_state",
    "rnd_next_state_position_only",
    "rnd_state",
    "rnd_linear_next_state",
]


def compute_intrinsic_vector_distance(
    algorithm: str,
    maze_map: np.ndarray,
    intrinsic_reward_model: Any,
    position_velocity_visit_count_wrapper: Any,
) -> Optional[Dict[str, float]]:
    """
    Compute distance between the algorithm's intrinsic vector and GT (position_velocity visit counts).
    Only runs when algorithm is in ALGORITHMS_NO_ACTION.
    For no_exploration, pred is a zero vector.
    Returns a dict of metric name -> value for all VALID_DISTANCE_ALGORITHMS, or None if skipped.
    Caller should log the dict (e.g. print_or_wandb_log or wandb.log with step).
    """
    if algorithm not in ALGORITHMS_NO_ACTION:
        return None

    # Same data manipulation as full_intrinsic_vector: obs_list -> np.stack(obs_list).astype(np.float32).
    maze_map = np.asarray(maze_map)
    grid_rows, grid_cols = maze_map.shape
    visit_counts = position_velocity_visit_count_wrapper.visit_counts
    obs_list = full_observation_list(maze_map)
    obs_arr = np.stack(obs_list).astype(np.float32)
    gt_list = []
    for obs in obs_arr:
        row, col = observation_to_grid(obs, grid_rows, grid_cols)
        vx_bin, vy_bin = velocity_to_grid(obs, n_bins=10)
        gt_list.append(visit_counts[row, col, vx_bin, vy_bin])
    gt = np.array(gt_list, dtype=np.float64)

    # pred is the intrinsic vector for the algorithm
    if algorithm == "no_exploration":
        pred = np.zeros_like(gt, dtype=np.float64)
    else:
        pred = full_intrinsic_vector(maze_map, intrinsic_reward_model)
    out = {}
    for name in VALID_DISTANCE_ALGORITHMS:
        fn = getattr(vector_distance, name)
        out[f"distance_to_gt/{name}"] = float(fn(gt, pred))
    return out


def full_intrinsic_vector(
    maze_map: np.ndarray,
    intrinsic_reward_model: Any,
) -> np.ndarray:
    """
    Intrinsic value for each observation in full_observation_list(maze_map),
    using the provided IntrinsicRewardModel. Works for algorithms that do not need action:
    state-only, next_state-only, or both (observations and next_observations are the same grid).
    Not for state+action or state+action+next_state (e.g. rnd_state_action, rnd_elliptical).
    Distance to GT is only computed for such algorithms; GT is the visit count vector from gt_position_velocity.

    Parameters
    ----------
    maze_map : np.ndarray
        Maze map, shape (grid_rows, grid_cols). Only valid (non-wall) cells are used.
    intrinsic_reward_model : IntrinsicRewardModel
        Model with compute(samples). samples gets "observations" and "next_observations"
        both set to the full observation list (no "actions").

    Returns
    -------
    np.ndarray
        1D array of intrinsic values, same length as full_observation_list(maze_map).
    """
    obs_list = full_observation_list(maze_map)
    obs_arr = np.stack(obs_list).astype(np.float32)
    samples = {"observations": obs_arr, "next_observations": obs_arr}
    result = intrinsic_reward_model.compute(samples)
    if hasattr(result, "cpu"):
        result = result.cpu().numpy()
    result = np.asarray(result).reshape(-1)
    return result.astype(np.float64)
