"""
Intrinsic reward vector for each observation in full_observation_list (discretized valid cells),
using a given IntrinsicRewardModel.
"""
import os
import sys
from typing import Any

import numpy as np

from .full_observation_list import full_observation_list


ALGORITHM_NAMES = [
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


def full_intrinsic_vector(
    maze_map: np.ndarray,
    intrinsic_reward_model: Any,
) -> np.ndarray:
    """
    Intrinsic value for each observation in full_observation_list(maze_map, include_velocity_bins=True),
    using the provided IntrinsicRewardModel. Works for algorithms that do not need action:
    state-only, next_state-only, or both (observations and next_observations are the same grid).
    Not for state+action or state+action+next_state (e.g. rnd_state_action, rnd_elliptical).

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
        1D array of intrinsic values, same length as full_observation_list(maze_map, include_velocity_bins=True).
    """
    obs_list = full_observation_list(maze_map, include_velocity_bins=True)
    obs_arr = np.stack(obs_list).astype(np.float32)
    samples = {"observations": obs_arr, "next_observations": obs_arr}
    result = intrinsic_reward_model.compute(samples)
    if hasattr(result, "cpu"):
        result = result.cpu().numpy()
    result = np.asarray(result).reshape(-1)
    return result.astype(np.float64)
