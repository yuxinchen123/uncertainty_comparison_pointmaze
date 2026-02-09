"""
Gymnasium wrappers for PointMaze: fixed goal/start, goal removal, visit counting.

Used by the training script and tests. Self-contained within 07_reconstruction.
"""
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
    Wraps PointMaze to track visit counts per grid cell.
    Reads maze map from env. Maps (x,y) to (row,col), increments visit_counts on open cells.
    Optional: add intrinsic reward when beta > 0. Decay: intrinsic_decay_rate -2 = 1/sqrt(n), -1 = 1/n.
    For eval: pass count_map_ref=train_env.visit_counts and update_counts=False to reuse train counts
    without modifying them.
    """

    def __init__(self, env, beta=0.0, intrinsic_decay_rate=-0.5, count_map_ref=None, update_counts=True):
        super().__init__(env)
        self.beta = beta
        self.intrinsic_decay_rate = intrinsic_decay_rate
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

    def _get_intrinsic_reward(self, row, col):
        """Intrinsic reward = 1/n^exponent before this visit; exponent = -decay_rate. Capped at 1.0 for unvisited."""
        if not (0 <= row < self.grid_rows and 0 <= col < self.grid_cols):
            return 0.0
        if self.maze_map[row, col] != 0:
            return 0.0
        count = self.visit_counts[row, col]
        if count <= 0:
            return 1.0
        return min(1.0, pow(float(count), self.intrinsic_decay_rate))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        row, col = self._state_to_grid(obs)
        intrinsic = 0.0
        if self.beta != 0.0:
            intrinsic = self._get_intrinsic_reward(row, col)
        if self.update_counts and 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            if self.maze_map[row, col] == 0:
                self.visit_counts[row, col] += 1
        total_reward = reward + self.beta * intrinsic
        info["intrinsic_reward"] = intrinsic
        info["extrinsic_reward"] = reward
        return obs, total_reward, terminated, truncated, info

    def get_visit_counts(self):
        return self.visit_counts.copy()

    def compute_intrinsic_reward(self, obs):
        """Compute intrinsic reward for obs using current visit counts. For replay buffer sample-time recomputation."""
        row, col = self._state_to_grid(obs)
        return self._get_intrinsic_reward(row, col)
