"""
Visit-count-based intrinsic reward model.
Implements IntrinsicRewardModel; update() is a no-op (counts are updated by VisitCountWrapper).
Decay: intrinsic_decay_rate -0.5 => 1/sqrt(n), -1 => 1/n.
"""
import numpy as np
from typing import Any, Dict

from .base import IntrinsicRewardModel


class VisitCount(IntrinsicRewardModel):
    """
    Visit-count intrinsic reward. Expects a visit-count wrapper with:
    visit_counts, maze_map, _state_to_grid(obs).
    compute(samples) returns bonus array; update(samples) is no-op.
    """

    def __init__(self, visit_count_wrapper, intrinsic_decay_rate: float):
        self.visit_count_wrapper = visit_count_wrapper
        self.intrinsic_decay_rate = intrinsic_decay_rate

    def _bonus(self, row: int, col: int) -> float:
        visit_counts = self.visit_count_wrapper.visit_counts
        maze_map = self.visit_count_wrapper.maze_map
        grid_rows, grid_cols = visit_counts.shape
        if not (0 <= row < grid_rows and 0 <= col < grid_cols) or maze_map[row, col] != 0:
            return 0.0
        count = visit_counts[row, col]
        return 1.0 if count <= 0 else min(1.0, pow(float(count), self.intrinsic_decay_rate))

    def compute(self, samples: Dict[str, Any]) -> np.ndarray:
        """samples["next_observations"] shape (N, obs_dim) -> (N,) bonus array."""
        next_obs = np.asarray(samples["next_observations"])
        if next_obs.ndim == 1:
            next_obs = next_obs.reshape(1, -1)
        batch_size = next_obs.shape[0]
        out = np.zeros(batch_size)
        for i in range(batch_size):
            row, col = self.visit_count_wrapper._state_to_grid(next_obs[i])
            out[i] = self._bonus(row, col)
        return out

    def update(self, samples: Dict[str, Any]) -> None:
        """No-op; visit counts are updated by the env wrapper."""
        pass
