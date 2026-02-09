"""
DictReplayBuffer for SAC that recomputes intrinsic rewards on sample
using current visit counts instead of rewards stored at collection time.
"""
import numpy as np
import torch
from typing import Callable, Optional, Union

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples


class IntrinsicReplayBuffer(DictReplayBuffer):
    """
    DictReplayBuffer that recomputes intrinsic rewards when sampling.

    Stores extrinsic rewards from info; on sample(), recomputes intrinsic via
    intrinsic_reward_fn (using current visit counts) and returns total = extrinsic + beta * intrinsic.
    """

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
        self.extrinsic_rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)

    def add(
        self,
        obs: dict,
        next_obs: dict,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list,
    ) -> None:
        extrinsic_reward = reward.copy()
        for env_idx, info in enumerate(infos):
            if isinstance(info, dict) and "extrinsic_reward" in info:
                extrinsic_reward[env_idx] = info["extrinsic_reward"]
        self.extrinsic_rewards[self.pos] = extrinsic_reward
        super().add(obs, next_obs, action, reward, done, infos)

    def sample(
        self,
        batch_size: int,
        env: Optional = None,
    ) -> DictReplayBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        batch = super()._get_samples(batch_inds, env)

        if self.intrinsic_reward_fn is None or self.beta == 0.0:
            return batch

        next_obs_dict = {}
        for key, obs_tensor in batch.next_observations.items():
            if hasattr(obs_tensor, "cpu"):
                next_obs_dict[key] = obs_tensor.cpu().numpy()
            else:
                next_obs_dict[key] = obs_tensor

        batch_size_actual = len(next_obs_dict[list(next_obs_dict.keys())[0]])
        intrinsic_rewards = []
        for i in range(batch_size_actual):
            obs_dict = {k: next_obs_dict[k][i] for k in next_obs_dict}
            try:
                r = self.intrinsic_reward_fn(obs_dict)
                intrinsic_rewards.append(float(r) if not isinstance(r, (list, np.ndarray)) else float(r[0]))
            except Exception:
                intrinsic_rewards.append(0.0)

        intrinsic_rewards = np.array(intrinsic_rewards, dtype=np.float32)
        extrinsic_rewards = self.extrinsic_rewards[batch_inds, 0]
        total_rewards = extrinsic_rewards + self.beta * intrinsic_rewards
        # SB3 expects rewards shape (batch_size, 1); (batch_size,) can cause critic MSE shape mismatch
        total_rewards = total_rewards.reshape(-1, 1)

        if hasattr(batch.rewards, "device"):
            total_rewards = torch.from_numpy(total_rewards).float().to(batch.rewards.device)

        return DictReplayBufferSamples(
            observations=batch.observations,
            actions=batch.actions,
            next_observations=batch.next_observations,
            dones=batch.dones,
            rewards=total_rewards,
        )
