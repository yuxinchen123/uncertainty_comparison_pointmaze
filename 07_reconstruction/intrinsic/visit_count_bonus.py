"""
Visit-count-based intrinsic reward (exploration bonus).

For use with VectorIntrinsicReplayBuffer (Box obs only); use an adapter (e.g. _CallableAsRND)
if you want visit-count as the intrinsic. Decay: intrinsic_decay_rate -0.5 => 1/sqrt(n), -1 => 1/n.
Expects a visit-count wrapper with: visit_counts, maze_map, _state_to_grid(obs).
"""
import numpy as np


def make_visit_count_intrinsic_reward_fn(visit_count_wrapper, intrinsic_decay_rate: float):
    """
    Return a callable that uses current visit_counts from the wrapper on each call.

    f(samples) -> np.ndarray: samples["next_observations"] shape (N, obs_dim) -> (N,).
    Bonus = 1/n^exponent for visit count n (exponent = -decay_rate), capped 1.0; 0.0 for wall/out-of-bounds.
    """

    def _bonus(row: int, col: int) -> float:
        visit_counts = visit_count_wrapper.visit_counts
        maze_map = visit_count_wrapper.maze_map
        grid_rows, grid_cols = visit_counts.shape
        if not (0 <= row < grid_rows and 0 <= col < grid_cols) or maze_map[row, col] != 0:
            return 0.0
        count = visit_counts[row, col]
        return 1.0 if count <= 0 else min(1.0, pow(float(count), intrinsic_decay_rate))

    def intrinsic_reward_fn(samples: dict) -> np.ndarray:
        next_obs = np.asarray(samples["next_observations"])
        if next_obs.ndim == 1:
            next_obs = next_obs.reshape(1, -1)
        batch_size = next_obs.shape[0]
        out = np.zeros(batch_size)
        for i in range(batch_size):
            row, col = visit_count_wrapper._state_to_grid(next_obs[i])
            out[i] = _bonus(row, col)
        return out

    return intrinsic_reward_fn
