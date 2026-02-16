"""
Visit-count-based intrinsic reward (exploration bonus).

Used for sample-time recomputation in IntrinsicReplayBuffer and for step info in
ComputeIntrinsicRewardWrapper. Decay: intrinsic_decay_rate -0.5 => 1/sqrt(n), -1 => 1/n.

Expects a visit-count wrapper with: visit_counts, maze_map, _state_to_grid(obs).
"""
import numpy as np
from typing import Any, Optional


def compute_visit_count_bonus(
    row: int,
    col: int,
    visit_counts: np.ndarray,
    maze_map: np.ndarray,
    intrinsic_decay_rate: float,
) -> float:
    """
    Intrinsic reward = 1/n^exponent for current visit count; exponent = -decay_rate.
    Capped at 1.0 for unvisited/open cells. Returns 0.0 for out-of-bounds or wall.
    """
    grid_rows, grid_cols = visit_counts.shape
    if not (0 <= row < grid_rows and 0 <= col < grid_cols):
        return 0.0
    if maze_map[row, col] != 0:
        return 0.0
    count = visit_counts[row, col]
    if count <= 0:
        return 1.0
    return min(1.0, pow(float(count), intrinsic_decay_rate))


def make_visit_count_intrinsic_reward_fn(visit_count_wrapper, intrinsic_decay_rate: float):
    """
    Return a callable suitable for IntrinsicReplayBuffer and ComputeIntrinsicRewardWrapper.

    Reads visit_counts, maze_map, and _state_to_grid from visit_count_wrapper on each call,
    so the bonus always uses current counts.

    - f(obs) -> float: single observation (dict or array) -> intrinsic reward.
    - f(batch_dict) -> np.ndarray: batch_dict with "next_observations" -> 1d array (next_state only).
    """

    def _obs_to_reward(obs: Any) -> float:
        row, col = visit_count_wrapper._state_to_grid(obs)
        return compute_visit_count_bonus(
            row,
            col,
            visit_count_wrapper.visit_counts,
            visit_count_wrapper.maze_map,
            intrinsic_decay_rate,
        )

    def intrinsic_reward_fn(x: Any) -> Optional[np.ndarray]:
        obs_batch = x.get("next_observations") if isinstance(x, dict) else None
        if isinstance(obs_batch, dict) and obs_batch:
            first = next(iter(obs_batch.values()))
            n = first.shape[0] if hasattr(first, "shape") else len(first)
            out = np.zeros(n, dtype=np.float32)
            for i in range(n):
                single = {}
                for k, v in obs_batch.items():
                    vi = v[i] if hasattr(v, "__getitem__") else v
                    single[k] = vi.cpu().numpy() if hasattr(vi, "cpu") else vi
                out[i] = _obs_to_reward(single)
            return out
        return _obs_to_reward(x)

    return intrinsic_reward_fn
