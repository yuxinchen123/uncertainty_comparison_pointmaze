"""
Intrinsic reward vector for each observation in full_observation_list (discretized valid cells),
using a given IntrinsicRewardModel. Distance-to-GT for algorithms that do not need action.
GT = intrinsic (bonus) vector of gt_position_velocity, so distance is 0 when algorithm is gt_position_velocity.
"""
from typing import Any, Dict, Optional

import numpy as np

from .full_observation_list import full_observation_list
from . import vector_distance
from utilities.format import to_numpy
from intrinsic.intrinsic_method.visit_count import VisitCount


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
    Compute distance between the algorithm's intrinsic vector and GT.
    GT = intrinsic (bonus) vector of gt_position_velocity (VisitCount on position_velocity wrapper).
    Only runs when algorithm is in ALGORITHMS_NO_ACTION. For no_exploration, pred is a zero vector.
    Returns a dict of metric name -> value for all VALID_DISTANCE_ALGORITHMS, or None if skipped.
    """
    if algorithm not in ALGORITHMS_NO_ACTION:
        return None

    maze_map = np.asarray(maze_map)
    # Previous count-based GT (raw visit counts per (row, col, vx_bin, vy_bin)):
    # from env_wrapper.point_maze_utils import observation_to_grid, velocity_to_grid
    # visit_counts = position_velocity_visit_count_wrapper.visit_counts
    # obs_list = full_observation_list(maze_map)
    # obs_arr = np.stack(obs_list).astype(np.float32)
    # gt_list = []
    # for obs in obs_arr:
    #     row, col = observation_to_grid(obs, maze_map.shape[0], maze_map.shape[1])
    #     vx_bin, vy_bin = velocity_to_grid(obs, n_bins=10)
    #     gt_list.append(visit_counts[row, col, vx_bin, vy_bin])
    # gt = np.array(gt_list, dtype=np.float64)
    gt_model = VisitCount(position_velocity_visit_count_wrapper)
    gt = full_intrinsic_vector(maze_map, gt_model)

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
    Distance to GT is only computed for such algorithms; GT is the intrinsic (bonus) vector of gt_position_velocity.

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
    result = to_numpy(result).reshape(-1)
    return result.astype(np.float64)
