"""Tests for the distance-to-ground-truth metric modules:
``rnd_exploration.metrics.full_observation_list`` (discretized valid-cell observation grid) and
``rnd_exploration.metrics.algorithm_vector`` (intrinsic vector + distance-to-GT for action-free
algorithms). Kept fast: tiny mazes, a stub visit-count wrapper, no environment construction and no
SAC training.
"""
import math

import numpy as np
import pytest
import torch

from rnd_exploration.metrics.full_observation_list import (
    full_observation_list,
    VELOCITY_N_BINS,
    LEFT,
    RIGHT,
    BOTTOM,
    TOP,
)
from rnd_exploration.metrics.algorithm_vector import (
    compute_intrinsic_vector_distance,
    full_intrinsic_vector,
    VALID_DISTANCE_ALGORITHMS,
)
from rnd_exploration.envs.point_maze_utils import observation_to_grid, velocity_to_grid
from rnd_exploration.methods.visit_count import VisitCount


# ----------------------------------------------------------------------------------------------- #
# Test doubles
# ----------------------------------------------------------------------------------------------- #
class StubPositionVelocityWrapper:
    """Minimal stand-in for PositionVelocityVisitCountWrapper: only the attributes VisitCount and
    full_intrinsic_vector touch (maze_map, grid shape, observation_to_count). No gym env needed."""

    VELOCITY_N_BINS = VELOCITY_N_BINS

    def __init__(self, maze_map, visit_counts=None):
        # store maze + grid shape; build a per-(row,col,vx_bin,vy_bin) count array if none given
        self.maze_map = np.asarray(maze_map)
        self.grid_rows, self.grid_cols = self.maze_map.shape
        n = self.VELOCITY_N_BINS
        if visit_counts is None:
            self.visit_counts = np.zeros((self.grid_rows, self.grid_cols, n, n), dtype=int)
        else:
            self.visit_counts = np.asarray(visit_counts)

    def observation_to_count(self, obs) -> int:
        """Map a flat [x, y, vx, vy] observation to its (row, col, vx_bin, vy_bin) count, mirroring
        the real wrapper so the round-trip with full_observation_list is exact."""
        row, col = observation_to_grid(obs, self.grid_rows, self.grid_cols)
        vx_bin, vy_bin = velocity_to_grid(obs, n_bins=self.VELOCITY_N_BINS)
        return int(self.visit_counts[row, col, vx_bin, vy_bin])


class ColumnVectorModel:
    """Stub IntrinsicRewardModel whose compute() returns a (N, 1) column vector, to confirm
    full_intrinsic_vector flattens to 1D."""

    def compute(self, samples):
        # one constant value per observation, shaped (N, 1) on purpose
        n = np.asarray(samples["next_observations"]).shape[0]
        return np.full((n, 1), 0.7, dtype=np.float64)


class TorchModel:
    """Stub IntrinsicRewardModel returning a torch tensor, to confirm the to_numpy() path."""

    def compute(self, samples):
        # return a 1D torch tensor of length N
        n = np.asarray(samples["next_observations"]).shape[0]
        return torch.arange(n, dtype=torch.float32)


# ----------------------------------------------------------------------------------------------- #
# Fixtures / helpers
# ----------------------------------------------------------------------------------------------- #
def _small_int_maze():
    """A 4x4 maze (1 = wall border, 0 = open) with exactly four open interior cells."""
    return np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        dtype=int,
    )


def _n_valid_int(maze_map):
    """Count open (== 0) cells in an integer maze."""
    return int(np.sum(np.asarray(maze_map) == 0))


def _varied_counts(wrapper):
    """Fill a wrapper's visit_counts with a deterministic, non-uniform pattern so the resulting GT
    bonus vector has real spread (some counts >= 2 give bonuses < 1)."""
    # before: all zeros; after: 0..(size-1) modulo 50 reshaped to the count grid
    size = wrapper.visit_counts.size
    wrapper.visit_counts = (np.arange(size) % 50).reshape(wrapper.visit_counts.shape)
    return wrapper


# ----------------------------------------------------------------------------------------------- #
# full_observation_list
# ----------------------------------------------------------------------------------------------- #
def test_full_observation_list_length_and_shape():
    """Golden: list length == (open cells) x VELOCITY_N_BINS^2 with 4D float32 obs; edge: an
    all-wall maze yields an empty list."""
    # golden: small maze -> n_valid cells each expanded over the full velocity grid
    maze = _small_int_maze()
    obs_list = full_observation_list(maze)
    n_valid = _n_valid_int(maze)
    assert n_valid == 4
    assert len(obs_list) == n_valid * VELOCITY_N_BINS * VELOCITY_N_BINS
    assert all(o.shape == (4,) and o.dtype == np.float32 for o in obs_list)

    # edge: every cell is a wall -> no valid cells -> empty list
    all_walls = np.ones((3, 5), dtype=int)
    assert full_observation_list(all_walls) == []


def test_full_observation_list_velocity_bins_and_bounds():
    """Golden: each cell expands to exactly VELOCITY_N_BINS distinct vx and vy values within
    [-5,5] and x,y inside the maze bounds; edge: a single open cell gives VELOCITY_N_BINS^2 obs."""
    # golden: positions stay inside the configured maze extent, velocities cover all bins
    maze = _small_int_maze()
    obs_arr = np.stack(full_observation_list(maze))
    xs, ys, vxs, vys = obs_arr[:, 0], obs_arr[:, 1], obs_arr[:, 2], obs_arr[:, 3]
    assert xs.min() >= LEFT and xs.max() <= RIGHT
    assert ys.min() >= BOTTOM and ys.max() <= TOP
    # the first cell's block of VELOCITY_N_BINS^2 rows must show exactly VELOCITY_N_BINS unique
    # vx values and VELOCITY_N_BINS unique vy values, all within the clamp range [-5, 5]
    first_block = obs_arr[: VELOCITY_N_BINS * VELOCITY_N_BINS]
    assert len(np.unique(first_block[:, 2])) == VELOCITY_N_BINS
    assert len(np.unique(first_block[:, 3])) == VELOCITY_N_BINS
    assert vxs.min() >= -5.0 and vxs.max() <= 5.0
    assert vys.min() >= -5.0 and vys.max() <= 5.0

    # edge: a maze with a single open cell -> exactly VELOCITY_N_BINS^2 observations
    one_cell = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
    assert len(full_observation_list(one_cell)) == VELOCITY_N_BINS * VELOCITY_N_BINS


def test_full_observation_list_string_symbols():
    """Golden: string cell symbols g/r/c/'0' all count as open; edge: '1' (wall string) is
    excluded, so the count matches only the open symbols."""
    # golden + edge in one map: four open symbols among a string wall border
    maze = np.array(
        [
            ["1", "1", "1", "1"],
            ["1", "0", "c", "1"],
            ["1", "r", "g", "1"],
            ["1", "1", "1", "1"],
        ]
    )
    obs_list = full_observation_list(maze)
    # only ("0", "c", "r", "g") are valid -> 4 cells; "1" strings excluded
    assert len(obs_list) == 4 * VELOCITY_N_BINS * VELOCITY_N_BINS


def test_full_observation_list_round_trips_to_grid():
    """Golden: each generated obs maps back to a valid open cell via observation_to_grid; edge:
    the velocity of each obs maps back to a bin in [0, VELOCITY_N_BINS-1]."""
    # golden: every observation must round-trip to an open cell of the maze
    maze = _small_int_maze()
    rows, cols = maze.shape
    for obs in full_observation_list(maze):
        r, c = observation_to_grid(obs, rows, cols)
        assert maze[r, c] == 0
        # edge: velocity discretization stays inside the valid bin range
        vx_bin, vy_bin = velocity_to_grid(obs, n_bins=VELOCITY_N_BINS)
        assert 0 <= vx_bin < VELOCITY_N_BINS
        assert 0 <= vy_bin < VELOCITY_N_BINS


# ----------------------------------------------------------------------------------------------- #
# full_intrinsic_vector
# ----------------------------------------------------------------------------------------------- #
def test_full_intrinsic_vector_is_1d():
    """Golden: VisitCount over a stub wrapper yields a 1D float64 vector matching the obs-list
    length; edge: a model returning a (N,1) column vector is flattened to 1D."""
    # golden: VisitCount bonus vector is 1D, right length, all bonuses in (0, 1]
    maze = _small_int_maze()
    wrapper = _varied_counts(StubPositionVelocityWrapper(maze))
    vec = full_intrinsic_vector(maze, VisitCount(wrapper))
    expected_len = len(full_observation_list(maze))
    assert vec.ndim == 1
    assert vec.shape == (expected_len,)
    assert vec.dtype == np.float64
    assert np.all(vec > 0.0) and np.all(vec <= 1.0)

    # edge: a (N, 1) column-vector model output must be reshaped to a 1D vector of length N
    col_vec = full_intrinsic_vector(maze, ColumnVectorModel())
    assert col_vec.ndim == 1
    assert col_vec.shape == (expected_len,)
    assert np.allclose(col_vec, 0.7)


def test_full_intrinsic_vector_accepts_torch_output():
    """Golden: a model returning a torch tensor is converted to a 1D numpy vector; edge: values
    survive the tensor->numpy round-trip unchanged."""
    # golden + edge: torch tensor output becomes a matching 1D numpy float64 vector
    maze = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
    expected_len = len(full_observation_list(maze))
    vec = full_intrinsic_vector(maze, TorchModel())
    assert vec.ndim == 1
    assert vec.shape == (expected_len,)
    np.testing.assert_allclose(vec, np.arange(expected_len, dtype=np.float64))


# ----------------------------------------------------------------------------------------------- #
# compute_intrinsic_vector_distance
# ----------------------------------------------------------------------------------------------- #
def test_distance_is_none_for_action_using_algorithm():
    """Golden: an action-using algorithm (rnd_state_action) returns None; edge: other action
    algorithms (rnd_state_action_next_state, rnd_elliptical) also return None."""
    # action-using algorithms are skipped entirely -> None, without touching model/wrapper
    maze = _small_int_maze()
    assert (
        compute_intrinsic_vector_distance("rnd_state_action", maze, None, None) is None
    )
    # edge: every other action-using algorithm is skipped the same way
    for algo in ("rnd_state_action_next_state", "rnd_elliptical"):
        assert compute_intrinsic_vector_distance(algo, maze, None, None) is None


def test_distance_to_self_is_zero_for_gt_position_velocity():
    """Golden: gt_position_velocity scored against an identical VisitCount model gives ~0 on every
    metric; edge: the returned dict has exactly the distance_to_gt/<metric> keys."""
    # golden: GT (built internally) and pred (passed in) are the same VisitCount model -> distance 0
    maze = _small_int_maze()
    wrapper = _varied_counts(StubPositionVelocityWrapper(maze))
    model = VisitCount(wrapper)
    out = compute_intrinsic_vector_distance("gt_position_velocity", maze, model, wrapper)
    assert out is not None
    # edge: the dict keys are exactly the namespaced metric names
    assert set(out.keys()) == {f"distance_to_gt/{m}" for m in VALID_DISTANCE_ALGORITHMS}
    for name, value in out.items():
        assert value == pytest.approx(0.0, abs=1e-3), f"{name} should be ~0, got {value}"


def test_distance_for_no_exploration_against_nonzero_gt():
    """Golden: no_exploration compares a zero pred against a non-trivial GT, so the metrics are
    finite and positive; edge: the angle metric equals pi/2 (zero pred is orthogonal to any GT)."""
    # golden: a varied GT vs the all-zero no_exploration prediction yields positive distances
    maze = _small_int_maze()
    wrapper = _varied_counts(StubPositionVelocityWrapper(maze))
    out = compute_intrinsic_vector_distance("no_exploration", maze, None, wrapper)
    assert out is not None
    assert out["distance_to_gt/min_c_l2_diff"] > 0.0
    assert out["distance_to_gt/min_c_l1_diff"] > 0.0
    assert all(math.isfinite(v) for v in out.values())
    # edge: with pred == 0, the cosine is 0 so the normalized angle is exactly pi/2
    assert out["distance_to_gt/normalized_angle_rad"] == pytest.approx(math.pi / 2, abs=1e-6)
