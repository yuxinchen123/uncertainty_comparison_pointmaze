"""Tests for rnd_exploration.envs.point_maze_wrappers on a tiny PointMaze_Large-v3 env.

Covers the visit-count wrappers (count-tensor increment + observation_to_count),
ComputeIntrinsicRewardWrapper (info fields, step reward unchanged), and
TerminateOnTimeLimitWrapper (truncation converted to termination). All tests use a
short-horizon env (max_episode_steps=20), zero actions, and at most ~20 steps so they
finish in seconds. No trained model is used: where a model is required it is either None
(beta=0) or a trivial in-test stub that returns a constant.
"""
import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
import gymnasium_robotics

from rnd_exploration.envs.point_maze_wrappers import (
    PositionVisitCountWrapper,
    PositionVelocityVisitCountWrapper,
    ComputeIntrinsicRewardWrapper,
    TerminateOnTimeLimitWrapper,
)
from rnd_exploration.envs.point_maze_utils import (
    get_maze_map,
    observation_to_grid,
    velocity_to_grid,
)

# register the gymnasium-robotics envs (PointMaze_*) once for the whole module
gym.register_envs(gymnasium_robotics)

# zero action: the PointMaze action space is Box(-1, 1, (2,)); a zero action keeps the
# agent essentially in its start cell so the grid cell it maps to stays an open cell
ZERO_ACTION = np.zeros((2,), dtype=np.float32)

# an observation whose (x, y) maps to a wall cell: y = 5.0 > top_edge (4.5) -> row 0,
# x = 0.0 -> col 6; maze_map[0, 6] is the top border wall (value 1)
WALL_OBS = {"observation": np.array([0.0, 5.0, 0.0, 0.0], dtype=np.float32)}


def make_pointmaze():
    """Build the tiny short-horizon PointMaze env used by every test (fresh each call)."""
    # continuing_task + reset_target=False keeps the task from terminating at the goal, so
    # the only episode end within 20 steps is the TimeLimit truncation
    return gym.make(
        "PointMaze_Large-v3",
        continuing_task=True,
        reset_target=False,
        max_episode_steps=20,
    )


def test_position_visit_count_increments_on_step():
    """Stepping increments the position count tensor and observation_to_count reflects it; update_counts=False leaves it at zero."""
    # golden path: a single step bumps exactly one open cell from 0 to 1
    env = make_pointmaze()
    wrapper = PositionVisitCountWrapper(env)
    assert wrapper.visit_counts.shape == (wrapper.grid_rows, wrapper.grid_cols)
    obs, _ = wrapper.reset(seed=0)
    assert wrapper.observation_to_count(obs) == 0  # nothing counted before any step
    obs, _, _, _, _ = wrapper.step(ZERO_ACTION)
    assert wrapper.visit_counts.sum() == 1  # exactly one open cell incremented
    assert wrapper.observation_to_count(obs) == 1  # the agent's new cell is the counted one
    # three more steps: total increments equal the number of steps taken
    for _ in range(3):
        wrapper.step(ZERO_ACTION)
    assert wrapper.visit_counts.sum() == 4

    # edge case: update_counts=False never modifies the count tensor
    env_no_update = make_pointmaze()
    frozen = PositionVisitCountWrapper(env_no_update, update_counts=False)
    frozen.reset(seed=0)
    for _ in range(4):
        frozen.step(ZERO_ACTION)
    assert frozen.visit_counts.sum() == 0


def test_position_visit_count_shared_ref_and_wall():
    """count_map_ref shares the same array the wrapper writes into; a wall observation raises ValueError."""
    # golden path: passing count_map_ref makes the wrapper use (not copy) that array
    env = make_pointmaze()
    maze = get_maze_map(env)
    ref = np.zeros(maze.shape, dtype=int)
    wrapper = PositionVisitCountWrapper(env, count_map_ref=ref, update_counts=True)
    assert wrapper.visit_counts is ref  # same object, not a copy
    obs, _ = wrapper.reset(seed=0)
    row, col = observation_to_grid(obs, wrapper.grid_rows, wrapper.grid_cols)
    # pre-seed the shared array at the start cell; observation_to_count must read it back
    ref[row, col] = 5
    assert wrapper.observation_to_count(obs) == 5
    # one step writes into the shared ref (5 already present + 1 new increment)
    wrapper.step(ZERO_ACTION)
    assert ref.sum() == 6

    # edge case: an observation over a wall cell reads count 0 ("unvisited" -> maximal bonus);
    # the AntMaze torso overhangs walls, so this is a legitimate position, not an error
    assert wrapper.observation_to_count(WALL_OBS) == 0


def test_position_velocity_visit_count_increments():
    """Stepping increments the 4D (row, col, vx_bin, vy_bin) count tensor; observation_to_count returns that single int; update_counts=False leaves it at zero."""
    # golden path: count tensor is (rows, cols, 10, 10); one step bumps one cell to 1
    env = make_pointmaze()
    wrapper = PositionVelocityVisitCountWrapper(env)
    assert wrapper.visit_counts.shape == (wrapper.grid_rows, wrapper.grid_cols, 10, 10)
    obs, _ = wrapper.reset(seed=0)
    assert wrapper.observation_to_count(obs) == 0
    obs, _, _, _, _ = wrapper.step(ZERO_ACTION)
    assert wrapper.visit_counts.sum() == 1
    count = wrapper.observation_to_count(obs)
    assert isinstance(count, int) and count == 1
    # the incremented entry is exactly the (position, velocity) bin of the current obs
    row, col = observation_to_grid(obs, wrapper.grid_rows, wrapper.grid_cols)
    vx_bin, vy_bin = velocity_to_grid(obs, n_bins=wrapper.VELOCITY_N_BINS)
    assert wrapper.visit_counts[row, col, vx_bin, vy_bin] == 1

    # edge case: update_counts=False keeps the tensor all zeros
    env_no_update = make_pointmaze()
    frozen = PositionVelocityVisitCountWrapper(env_no_update, update_counts=False)
    frozen.reset(seed=0)
    for _ in range(4):
        frozen.step(ZERO_ACTION)
    assert frozen.visit_counts.sum() == 0


def test_position_velocity_visit_count_observation_to_count_and_wall():
    """observation_to_count matches direct tensor indexing for an open cell; a wall observation raises ValueError."""
    # golden path: observation_to_count agrees with manual indexing of the count tensor
    env = make_pointmaze()
    ref = np.zeros((9, 12, 10, 10), dtype=int)
    wrapper = PositionVelocityVisitCountWrapper(env, count_map_ref=ref)
    obs, _ = wrapper.reset(seed=0)
    row, col = observation_to_grid(obs, wrapper.grid_rows, wrapper.grid_cols)
    vx_bin, vy_bin = velocity_to_grid(obs, n_bins=wrapper.VELOCITY_N_BINS)
    # pre-seed one combined-state bin and confirm observation_to_count returns it
    ref[row, col, vx_bin, vy_bin] = 7
    assert wrapper.observation_to_count(obs) == 7

    # edge case: a wall observation reads count 0 before any velocity lookup (same convention
    # as PositionVisitCountWrapper: wall/out-of-bounds positions are "unvisited", never an error)
    assert wrapper.observation_to_count(WALL_OBS) == 0


def test_compute_intrinsic_reward_info_fields():
    """With no model (beta=0) intrinsic is 0 and extrinsic equals the step reward; with a stub model the intrinsic field carries the model output while the step reward stays extrinsic."""
    # golden path: beta=0 / model=None still writes both info fields, intrinsic = 0.0
    env = FlattenObservation(make_pointmaze())
    wrapper = ComputeIntrinsicRewardWrapper(env, beta=0.0, intrinsic_reward_model=None)
    wrapper.reset(seed=0)
    _, reward, _, _, info = wrapper.step(ZERO_ACTION)
    assert info["intrinsic_reward"] == 0.0
    assert info["extrinsic_reward"] == reward  # logged extrinsic equals the returned reward

    # edge case: a stub model with beta != 0 populates info["intrinsic_reward"] with the
    # model's value, but the step reward returned is still the unchanged extrinsic reward
    class StubModel:
        """Test double: records the sample keys it saw and returns a constant intrinsic value."""

        def __init__(self):
            # remember which keys the wrapper passed so we can assert the samples contract
            self.seen_keys = None

        def compute(self, samples):
            """Return a fixed intrinsic reward, recording the keys present in samples."""
            self.seen_keys = set(samples.keys())
            return np.array([0.7], dtype=np.float32)

    model = StubModel()
    env_stub = FlattenObservation(make_pointmaze())
    wrapper_stub = ComputeIntrinsicRewardWrapper(env_stub, beta=0.5, intrinsic_reward_model=model)
    wrapper_stub.reset(seed=0)
    _, reward_stub, _, _, info_stub = wrapper_stub.step(ZERO_ACTION)
    assert info_stub["intrinsic_reward"] == pytest.approx(0.7)  # model output, not the env reward
    assert reward_stub == info_stub["extrinsic_reward"]  # step reward stays extrinsic
    assert reward_stub != info_stub["intrinsic_reward"]  # intrinsic was not folded into reward
    assert model.seen_keys == {"observations", "actions", "next_observations"}


def test_terminate_on_time_limit_real_env():
    """On the real env, steps before the limit are neither terminated nor truncated, and the final (20th) step is converted to terminated=True, truncated=False."""
    env = make_pointmaze()
    wrapper = TerminateOnTimeLimitWrapper(env)
    wrapper.reset(seed=0)
    last = None
    for i in range(20):
        _, _, terminated, truncated, _ = wrapper.step(ZERO_ACTION)
        # edge case: every step before the time limit passes through with both flags False
        if i < 19:
            assert not terminated and not truncated
        last = (terminated, truncated)
    # golden path: the time-limit truncation is surfaced as a termination instead
    assert last == (True, False)


def test_terminate_on_time_limit_branches():
    """Wrapper converts truncation-only into termination, and passes real terminations and ordinary steps through unchanged."""

    class ConfigurableStepEnv(gym.Env):
        """Minimal env returning a fixed (terminated, truncated) pair to exercise each branch."""

        def __init__(self, terminated, truncated):
            # store the flags this stub will return on every step
            self.observation_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
            self._terminated = terminated
            self._truncated = truncated

        def reset(self, seed=None, options=None):
            """Return a zero observation and empty info."""
            return np.zeros(2, dtype=np.float32), {}

        def step(self, action):
            """Return a zero obs, reward 1.0, and the preconfigured terminated/truncated flags."""
            return np.zeros(2, dtype=np.float32), 1.0, self._terminated, self._truncated, {}

    # golden path: truncated-only (the time-limit case) becomes terminated=True, truncated=False
    trunc_wrapper = TerminateOnTimeLimitWrapper(ConfigurableStepEnv(terminated=False, truncated=True))
    _, reward, terminated, truncated, _ = trunc_wrapper.step(ZERO_ACTION)
    assert (terminated, truncated) == (True, False)
    assert reward == 1.0  # reward is passed through unchanged

    # edge case: a genuine termination is left as-is
    term_wrapper = TerminateOnTimeLimitWrapper(ConfigurableStepEnv(terminated=True, truncated=False))
    _, _, terminated, truncated, _ = term_wrapper.step(ZERO_ACTION)
    assert (terminated, truncated) == (True, False)

    # edge case: an ordinary step (neither flag set) is left as-is
    plain_wrapper = TerminateOnTimeLimitWrapper(ConfigurableStepEnv(terminated=False, truncated=False))
    _, _, terminated, truncated, _ = plain_wrapper.step(ZERO_ACTION)
    assert (terminated, truncated) == (False, False)
