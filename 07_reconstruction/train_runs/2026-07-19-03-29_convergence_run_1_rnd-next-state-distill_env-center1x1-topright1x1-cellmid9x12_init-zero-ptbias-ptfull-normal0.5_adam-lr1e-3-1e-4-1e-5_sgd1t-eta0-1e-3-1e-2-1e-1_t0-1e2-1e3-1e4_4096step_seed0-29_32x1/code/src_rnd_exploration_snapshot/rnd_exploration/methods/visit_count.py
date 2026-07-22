"""
Visit-count-based intrinsic reward model.
Implements IntrinsicRewardModel; update() is a no-op (counts are updated by the visit-count wrapper).
Decay: intrinsic_decay_rate (default -0.5). -0.5 => min(1, 1/sqrt(n)), -1 => min(1, 1/n).
Positive decay makes count**decay >= 1 for n>=1, so min(1, ...) stays 1.0 (no decrease with visits).
Works with PositionVisitCountWrapper or PositionVelocityVisitCountWrapper; both expose observation_to_count(obs) -> int.
"""
import numpy as np
from typing import Any, Dict

from rnd_exploration.common.format import to_numpy
from .base import IntrinsicRewardModel


class VisitCount(IntrinsicRewardModel):
    """
    Visit-count intrinsic reward. Expects a wrapper with observation_to_count(obs) returning an int.
    compute(samples) returns bonus array; update(samples) is no-op.
    """

    def __init__(self, visit_count_wrapper, intrinsic_decay_rate: float = -0.5):
        self.visit_count_wrapper = visit_count_wrapper
        self.intrinsic_decay_rate = intrinsic_decay_rate

    def _count_to_bonus(self, count: int) -> float:
        """count <= 0 -> 1.0 (treat as max exploration bonus). Else min(1.0, count**decay).

        Use negative ``intrinsic_decay_rate`` so bonus decreases as count increases (e.g. -0.5 ~ 1/sqrt(n)).
        """
        return 1.0 if count <= 0 else min(1.0, pow(float(count), self.intrinsic_decay_rate))

    def compute(self, samples: Dict[str, Any]) -> np.ndarray:
        """samples["next_observations"] shape (N, obs_dim) -> (N,) bonus array. Handles torch tensors on GPU."""
        next_obs = to_numpy(samples["next_observations"])
        if next_obs.ndim == 1:
            next_obs = next_obs.reshape(1, -1)
        batch_size = next_obs.shape[0]
        out = np.zeros(batch_size)
        for i in range(batch_size):
            count = self.visit_count_wrapper.observation_to_count(next_obs[i])
            out[i] = self._count_to_bonus(count)
        return out

    def update(self, samples: Dict[str, Any]) -> None:
        """No-op; visit counts are updated by the env wrapper."""
        pass
