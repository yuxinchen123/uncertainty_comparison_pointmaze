"""
VectorIntrinsicReplayBuffer: ReplayBuffer (Box obs) that recomputes intrinsic rewards on sample.
On sample(), uses the sampled batch directly to build RND inputs, calls
rnd_module.compute(samples) and rnd_module.update(samples), and returns a batch with
rewards = extrinsic + beta * intrinsic. All MyRND data comes from sample() only.
"""
import numpy as np
import torch
from typing import Any, Callable, Optional, Union

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


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
        rnd_module: Optional[Any] = None,
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
        self.rnd_module = rnd_module

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

        if self.rnd_module is None or self.beta <= 0.0:
            return batch

        # Build minimal samples dict directly from the sampled batch
        rnd_device = getattr(self.rnd_module, "device", self.device)
        if not hasattr(rnd_device, "type"):
            rnd_device = torch.device(rnd_device)
        obs = batch.observations.to(rnd_device).float() if hasattr(batch.observations, "to") else torch.as_tensor(
            batch.observations, dtype=torch.float32, device=rnd_device
        )
        next_obs = (
            batch.next_observations.to(rnd_device).float()
            if hasattr(batch.next_observations, "to")
            else torch.as_tensor(batch.next_observations, dtype=torch.float32, device=rnd_device)
        )
        samples = {
            "observations": obs,
            "next_observations": next_obs,
        }
        intrinsic_rewards = self.rnd_module.compute(samples)
        self.rnd_module.update(samples)
        if hasattr(intrinsic_rewards, "cpu"):
            intrinsic = intrinsic_rewards.cpu().numpy().ravel()
        else:
            intrinsic = np.asarray(intrinsic_rewards, dtype=np.float32).ravel()

        # Use the stored rewards as extrinsic
        if hasattr(batch.rewards, "cpu"):
            extrinsic = batch.rewards.cpu().numpy().reshape(-1)
        else:
            extrinsic = np.asarray(batch.rewards, dtype=np.float32).reshape(-1)

        total = extrinsic + self.beta * intrinsic
        total = total.reshape(-1, 1)
        if hasattr(batch.rewards, "device"):
            total = torch.from_numpy(total).float().to(batch.rewards.device)

        return ReplayBufferSamples(
            observations=batch.observations,
            actions=batch.actions,
            next_observations=batch.next_observations,
            dones=batch.dones,
            rewards=total,
        )
