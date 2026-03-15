"""
Gymnasium wrappers for PointMaze: fixed goal/start, goal removal, visit counting.

Used by the training script and tests. Self-contained within 07_reconstruction.
"""
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .point_maze_utils import get_maze_map, observation_to_grid


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


class VisitCountWrapper(gym.Wrapper):
    """
    Wraps PointMaze to track visit counts per grid cell only.
    Reads maze map from env. Maps (x,y) to (row,col), increments visit_counts on open cells.
    Does not compute or add intrinsic reward; use ComputeIntrinsicRewardWrapper for that.
    For eval: pass count_map_ref=train_env.visit_counts and update_counts=False to reuse train counts.
    """

    def __init__(self, env, count_map_ref=None, update_counts=True):
        super().__init__(env)
        self.update_counts = update_counts
        self.maze_map = get_maze_map(env)
        if self.maze_map is None:
            raise ValueError("Could not extract maze_map from environment")
        self.grid_rows, self.grid_cols = self.maze_map.shape
        if count_map_ref is not None:
            self.visit_counts = count_map_ref
        else:
            self.visit_counts = np.zeros((self.grid_rows, self.grid_cols), dtype=int)

    def _state_to_grid(self, obs):
        """Map observation to 0-based (row, col)."""
        return observation_to_grid(obs, self.grid_rows, self.grid_cols)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        row, col = self._state_to_grid(obs)
        if self.update_counts and 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            if self.maze_map[row, col] == 0:
                self.visit_counts[row, col] += 1
        return obs, reward, terminated, truncated, info

    def get_visit_counts(self):
        return self.visit_counts.copy()


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
    This class for logging.
    Wraps an env to compute intrinsic reward via intrinsic_reward_model.compute() in step() and fill step info.
    Does not change the step reward (stays extrinsic); sets info['intrinsic_reward'] and
    info['extrinsic_reward'] for logging.

    intrinsic_reward_model must implement compute(samples) returning a 1D tensor or array;
    samples is a dict with "next_observations" of shape (batch_size, obs_dim).
    """

    def __init__(self, env, beta: float = 0.0, intrinsic_reward_model: Optional[Any] = None):
        super().__init__(env)
        self.beta = beta
        self.intrinsic_reward_model = intrinsic_reward_model

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        intrinsic = 0.0
        if self.beta != 0.0 and self.intrinsic_reward_model is not None:
            obs_arr = np.atleast_2d(np.asarray(obs, dtype=np.float32))
            samples = {"next_observations": obs_arr}
            r = self.intrinsic_reward_model.compute(samples)
            intrinsic = float(r.item()) if hasattr(r, "item") else float(np.asarray(r).ravel()[0])
        info["intrinsic_reward"] = intrinsic
        info["extrinsic_reward"] = reward
        return obs, reward, terminated, truncated, info
