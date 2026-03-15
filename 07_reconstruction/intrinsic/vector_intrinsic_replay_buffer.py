"""
VectorIntrinsicReplayBuffer: ReplayBuffer (Box obs) that recomputes intrinsic rewards on sample.
On sample(), uses the sampled batch to call intrinsic_reward_model.compute(samples) and
intrinsic_reward_model.update(samples), then returns a batch with rewards = extrinsic + beta * intrinsic.
"""
import numpy as np
import torch
from typing import Any, Callable, Optional, Union

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from utilities.format import to_numpy_flat, to_tensor


class VectorIntrinsicReplayBuffer(ReplayBuffer):
    """ReplayBuffer (Box observation) that recomputes intrinsic rewards on sample()."""

    def __init__(
        self,
        buffer_size: int,
        observation_space,
        action_space,
        device: Union[str, torch.device] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        intrinsic_reward_fn: Optional[Callable] = None,
        beta: float = 1.0,
        intrinsic_reward_model: Optional[Any] = None,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.intrinsic_reward_fn = intrinsic_reward_fn
        self.beta = beta
        self.intrinsic_reward_model = intrinsic_reward_model

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list,
    ) -> None:
        # The env reward passed in is already the extrinsic reward (no intrinsic added),
        # so we can store it directly.
        super().add(obs, next_obs, action, reward, done, infos)

    def sample(
        self,
        batch_size: int,
        env: Optional = None,
    ) -> ReplayBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        batch = super()._get_samples(batch_inds, env)

        if self.intrinsic_reward_model is None or self.beta <= 0.0:
            return batch

        # Build minimal samples dict directly from the sampled batch
        obs = to_tensor(batch.observations, self.device)
        next_obs = to_tensor(batch.next_observations, self.device)
        actions = to_tensor(batch.actions, self.device)
        samples = {
            "observations": obs,
            "next_observations": next_obs,
            "actions": actions,
        }
        intrinsic_rewards = self.intrinsic_reward_model.compute(samples)
        self.intrinsic_reward_model.update(samples)
        intrinsic = to_numpy_flat(intrinsic_rewards)
        extrinsic = to_numpy_flat(batch.rewards)
        # SB3 expects rewards shape (batch_size, 1); (batch_size,) can cause critic MSE shape mismatch
        total = (extrinsic + self.beta * intrinsic).reshape(-1, 1)
        total = to_tensor(total, self.device)

        return ReplayBufferSamples(
            observations=batch.observations,
            actions=batch.actions,
            next_observations=batch.next_observations,
            dones=batch.dones,
            rewards=total,
        )
