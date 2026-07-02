"""
Abstract intrinsic reward model for VectorIntrinsicReplayBuffer.
Implementations must provide compute(samples) and update(samples).
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import numpy as np
import torch


class IntrinsicRewardModel(ABC):
    """
    Abstract base for intrinsic reward models used by VectorIntrinsicReplayBuffer.
    samples is a dict with "observations" and "next_observations" (batch_size, obs_dim).
    """

    @abstractmethod
    def compute(self, samples: Dict[str, Any]) -> Union[torch.Tensor, np.ndarray]:
        """
        Return intrinsic rewards for the batch.
        samples["next_observations"] shape (N, obs_dim) -> returns shape (N,) or (N, 1).
        """
        pass

    @abstractmethod
    def update(self, samples: Dict[str, Any]) -> None:
        """
        Optional training step using a replay minibatch (called by the buffer on sample()).
        No-op for non-learned methods (e.g. visit count).
        """
        pass

    def observe(self, samples: Dict[str, Any]) -> None:
        """
        Optional update from freshly collected transition(s), called by the buffer on add().
        Default no-op. The elliptical family overrides this to support an optional add-time
        covariance update (update_timing="add", counting environment visitation rather than replay
        draws); every other model ignores it.
        """
        pass
