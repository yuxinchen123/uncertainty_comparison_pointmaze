"""
Gymnasium wrappers for PointMaze: fixed goal/start, goal removal, visit counting.

Used by the training script and tests. Self-contained within 07_reconstruction.
"""
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .point_maze_utils import get_cell_size, get_maze_map, observation_to_grid, velocity_to_grid


class FixedGoalWrapper(gym.Wrapper):
    """Wraps PointMaze to use a fixed goal on every reset, selected deterministically by seed."""

    def __init__(self, env, goal_cell):
        super().__init__(env)
        self.goal_cell = goal_cell  # (row, col) 0-based

    def reset(self, seed=None, options=None, **kwargs):
        opts = dict(options) if options else {}
        opts["goal_cell"] = [int(self.goal_cell[0]), int(self.goal_cell[1])]
        return self.env.reset(seed=seed, options=opts, **kwargs)


class FixedStartWrapper(gym.Wrapper):
    """Wraps PointMaze to use a fixed start cell on every reset (each seed corresponds to one fixed start)."""

    def __init__(self, env, start_cell):
        super().__init__(env)
        self.start_cell = start_cell  # (row, col) 0-based

    def reset(self, seed=None, options=None, **kwargs):
        opts = dict(options) if options else {}
        opts["reset_cell"] = [int(self.start_cell[0]), int(self.start_cell[1])]
        return self.env.reset(seed=seed, options=opts, **kwargs)


class RemoveGoalWrapper(gym.Wrapper):
    """Removes desired_goal (and achieved_goal) from observation. Use when goal is fixed."""

    def __init__(self, env, remove_keys=None):
        super().__init__(env)
        self.remove_keys = remove_keys or ["desired_goal", "achieved_goal"]
        if isinstance(env.observation_space, spaces.Dict):
            new_spaces = {
                k: v for k, v in env.observation_space.spaces.items()
                if k not in self.remove_keys
            }
            self.observation_space = spaces.Dict(new_spaces)
        else:
            self.observation_space = env.observation_space

    def _filter_obs(self, obs):
        if isinstance(obs, dict):
            return {k: v for k, v in obs.items() if k not in self.remove_keys}
        return obs

    def reset(self, seed=None, options=None, **kwargs):
        obs, info = self.env.reset(seed=seed, options=options, **kwargs)
        return self._filter_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._filter_obs(obs), reward, terminated, truncated, info


class PositionVisitCountWrapper(gym.Wrapper):
    """
    Wraps a maze env to track visit counts per position grid cell only.
    Reads maze map and cell size from env. Maps (x,y) to (row,col), increments visit_counts on open cells.
    subdivision=n splits every maze cell into n x n sub-cells (e.g. AntMaze 4 m cells with
    subdivision=4 give a 1 m x 1 m grid); subdivision=1 (default) is the plain per-cell grid.
    Does not compute or add intrinsic reward; use ComputeIntrinsicRewardWrapper for that.
    For eval: pass count_map_ref=train_env.visit_counts and update_counts=False to reuse train counts.
    """

    def __init__(self, env, count_map_ref=None, update_counts=True, subdivision: int = 1):
        super().__init__(env)
        self.update_counts = update_counts
        base_map = get_maze_map(env)
        if base_map is None:
            raise ValueError("Could not extract maze_map from environment")
        self.subdivision = int(subdivision)
        # expand the wall map to the sub-grid so every consumer (wall check, open-cell count) sees one
        # uniform grid. before: 9x12 map of 4 m cells, subdivision 4; after: 36x48 map where each wall
        # entry became a 4x4 block of walls and each open entry a 4x4 block of open sub-cells.
        self.maze_map = np.kron(base_map, np.ones((self.subdivision, self.subdivision), dtype=base_map.dtype)) \
            if self.subdivision > 1 else base_map
        self.grid_rows, self.grid_cols = self.maze_map.shape
        # sub-cell edge length in meters (PointMaze cell 1 m / AntMaze cell 4 m, divided by subdivision)
        self.cell_size = get_cell_size(env) / self.subdivision
        if count_map_ref is not None:
            self.visit_counts = count_map_ref
        else:
            self.visit_counts = np.zeros((self.grid_rows, self.grid_cols), dtype=int)

    def observation_to_count(self, obs) -> int:
        """Map observation to grid cell and return visit count for that cell. Raises ValueError if out-of-bounds or wall."""
        row, col = observation_to_grid(obs, self.grid_rows, self.grid_cols, self.cell_size)
        if not (0 <= row < self.grid_rows and 0 <= col < self.grid_cols):
            raise ValueError(f"observation maps to grid (row={row}, col={col}) which is out of bounds for grid shape ({self.grid_rows}, {self.grid_cols})")
        if self.maze_map[row, col] != 0:
            raise ValueError(f"observation maps to wall cell (row={row}, col={col})")
        return int(self.visit_counts[row, col])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        row, col = observation_to_grid(obs, self.grid_rows, self.grid_cols, self.cell_size)
        if self.update_counts and 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            if self.maze_map[row, col] == 0:
                self.visit_counts[row, col] += 1
        return obs, reward, terminated, truncated, info

    def get_visit_counts(self):
        return self.visit_counts.copy()


class PositionVelocityVisitCountWrapper(gym.Wrapper):
    """
    Tracks visit count per (position, velocity) cell: one count per (row, col, vx_bin, vy_bin).
    Velocity is discretized via velocity_to_grid (vx, vy in [-5, 5] m/s -> 10 bins each).
    observation_to_count(obs) returns a single int (visit count for that combined state).
    """

    VELOCITY_N_BINS = 10

    def __init__(self, env, count_map_ref=None, update_counts=True):
        super().__init__(env)
        self.update_counts = update_counts
        self.maze_map = get_maze_map(env)
        self.grid_rows, self.grid_cols = self.maze_map.shape
        self.cell_size = get_cell_size(env)
        n = self.VELOCITY_N_BINS
        if count_map_ref is not None:
            self.visit_counts = np.asarray(count_map_ref)
        else:
            self.visit_counts = np.zeros((self.grid_rows, self.grid_cols, n, n), dtype=int)

    def observation_to_count(self, obs) -> int:
        """Return visit count for the (position, velocity) cell. Raises ValueError if position is out-of-bounds or wall."""
        row, col = observation_to_grid(obs, self.grid_rows, self.grid_cols, self.cell_size)
        if not (0 <= row < self.grid_rows and 0 <= col < self.grid_cols):
            raise ValueError(f"observation maps to grid (row={row}, col={col}) which is out of bounds for grid shape ({self.grid_rows}, {self.grid_cols})")
        if self.maze_map[row, col] != 0:
            raise ValueError(f"observation maps to wall cell (row={row}, col={col})")
        vx_bin, vy_bin = velocity_to_grid(obs, n_bins=self.VELOCITY_N_BINS)
        return int(self.visit_counts[row, col, vx_bin, vy_bin])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.update_counts:
            row, col = observation_to_grid(obs, self.grid_rows, self.grid_cols, self.cell_size)
            if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols and self.maze_map[row, col] == 0:
                vx_bin, vy_bin = velocity_to_grid(obs, n_bins=self.VELOCITY_N_BINS)
                self.visit_counts[row, col, vx_bin, vy_bin] += 1
        return obs, reward, terminated, truncated, info

    def get_visit_counts(self):
        """Return visit count array (grid_rows, grid_cols, 10, 10) for eval reuse."""
        return self.visit_counts.copy()


class AttachAchievedGoalWrapper(gym.ObservationWrapper):
    """Prepend achieved_goal (the global x, y) to the 'observation' key of a Dict observation.
    AntMaze strips x, y out of 'observation' (ant_maze_v5 _get_obs: observation = ant_obs[2:]), so
    without this the network state has no maze position. Apply BEFORE RemoveGoalWrapper; the
    resulting flat state then starts with (x, y), matching the PointMaze layout."""

    def __init__(self, env):
        super().__init__(env)
        obs_spaces = dict(env.observation_space.spaces)
        inner = obs_spaces["observation"]
        achieved = obs_spaces["achieved_goal"]
        obs_spaces["observation"] = spaces.Box(
            low=np.concatenate([achieved.low, inner.low]),
            high=np.concatenate([achieved.high, inner.high]),
            dtype=inner.dtype,
        )
        self.observation_space = spaces.Dict(obs_spaces)

    def observation(self, obs):
        """Rebuild the dict with (x, y) prepended to 'observation'; goal keys pass through unchanged."""
        out = dict(obs)
        out["observation"] = np.concatenate([obs["achieved_goal"], obs["observation"]])
        return out


class RewardShiftWrapper(gym.Wrapper):
    """Add a constant to every step's extrinsic reward. With shift=-1 the sparse maze reward
    (1 within the goal radius, else 0) becomes the ExPLORe convention: -1 per step, 0 on the
    goal-reaching step."""

    def __init__(self, env, shift: float):
        super().__init__(env)
        self.shift = float(shift)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward + self.shift, terminated, truncated, info


class TerminateOnTimeLimitWrapper(gym.Wrapper):
    """Convert time-limit truncation to termination: when inner env returns truncated=True
    (e.g. max_episode_steps reached), return terminated=True, truncated=False instead.
    So the episode ends as a full termination, not a timeout."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if truncated and not terminated:
            return obs, reward, True, False, info
        return obs, reward, terminated, truncated, info


class ComputeIntrinsicRewardWrapper(gym.Wrapper):
    """
    Wraps an env to compute intrinsic reward via intrinsic_reward_model.compute() in step() and fill step info.
    Does not change the step reward (stays extrinsic); sets info['intrinsic_reward'] and
    info['extrinsic_reward'] for logging.

    intrinsic_reward_model must implement compute(samples) returning a 1D tensor or array.
    samples always has "next_observations"; for models that use (s,a) it also has "observations" and "actions".
    """

    def __init__(self, env, beta: float = 0.0, intrinsic_reward_model: Optional[Any] = None):
        super().__init__(env)
        self.beta = beta
        self.intrinsic_reward_model = intrinsic_reward_model
        self._last_obs = None

    def reset(self, seed=None, options=None, **kwargs):
        obs, info = self.env.reset(seed=seed, options=options, **kwargs)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        intrinsic = 0.0
        if self.beta != 0.0 and self.intrinsic_reward_model is not None:
            next_arr = np.atleast_2d(np.asarray(obs, dtype=np.float32))
            last_arr = np.atleast_2d(self._last_obs)
            act_arr = np.atleast_2d(np.asarray(action, dtype=np.float32))
            samples = {
                "observations": last_arr,
                "actions": act_arr,
                "next_observations": next_arr,
            }
            r = self.intrinsic_reward_model.compute(samples)
            intrinsic = float(r.item()) if hasattr(r, "item") else float(np.asarray(r).ravel()[0])
        self._last_obs = np.asarray(obs, dtype=np.float32)
        info["intrinsic_reward"] = intrinsic
        info["extrinsic_reward"] = reward
        return obs, reward, terminated, truncated, info
