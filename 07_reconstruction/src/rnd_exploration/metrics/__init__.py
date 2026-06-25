"""Distance-to-GT and discretization utilities for PointMaze."""
from .full_observation_list import full_observation_list
from .algorithm_vector import full_intrinsic_vector
from rnd_exploration.methods import ALGORITHM_NAMES  # single source of truth (the registry)
from .vector_distance import (
    min_c_l1_diff,
    min_c_l2_diff,
    min_c_l1_inv,
    min_c_l2_inv,
    normalized_l2,
    normalized_angle_rad,
)

__all__ = [
    "full_observation_list",
    "ALGORITHM_NAMES",
    "full_intrinsic_vector",
    "min_c_l1_diff",
    "min_c_l2_diff",
    "min_c_l1_inv",
    "min_c_l2_inv",
    "normalized_l2",
    "normalized_angle_rad",
]
