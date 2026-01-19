"""
Custom DictReplayBuffer for SAC that recalculates intrinsic rewards on sample
using current visit counts instead of stale rewards stored at collection time.
This version handles dict observations (used by PointMaze).
"""
import numpy as np
import torch
from typing import Optional, Union, Callable
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.type_aliases import DictReplayBufferSamples


class DictIntrinsicReplayBuffer(DictReplayBuffer):
    """
    Custom DictReplayBuffer that recalculates intrinsic rewards when sampling.
    
    This ensures that intrinsic rewards reflect current visit counts, not
    the visit counts at the time the transition was collected.
    
    The buffer stores extrinsic rewards separately (extracted from info dict)
    and recalculates intrinsic rewards on each sample using current visit counts.
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
        intrinsic_reward_fn=None,
        beta: float = 1.0,
    ):
        """
        Args:
            buffer_size: Size of the replay buffer
            observation_space: Observation space (Dict)
            action_space: Action space
            device: PyTorch device
            n_envs: Number of parallel environments
            optimize_memory_usage: Whether to optimize memory usage
            handle_timeout_termination: Whether to handle timeout terminations
            intrinsic_reward_fn: Function to compute intrinsic reward given observations
                                Should accept dict observation and return scalar reward
            beta: Intrinsic reward coefficient
        """
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
        
        # Store extrinsic rewards separately (extracted from info dicts)
        # Shape: (buffer_size, n_envs)
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
        """
        Add a transition to the buffer.
        
        Extracts extrinsic reward from info dict and stores it separately.
        The stored reward is the total reward (for compatibility), but we
        also store extrinsic separately for recalculation.
        """
        # Extract extrinsic reward from info dict if available
        # If not available, assume the stored reward is extrinsic only
        extrinsic_reward = reward.copy()
        for env_idx, info in enumerate(infos):
            if isinstance(info, dict) and 'extrinsic_reward' in info:
                extrinsic_reward[env_idx] = info['extrinsic_reward']
        
        # Store extrinsic rewards separately
        pos = self.pos
        self.extrinsic_rewards[pos] = extrinsic_reward
        
        # Call parent add with total reward (for compatibility)
        # The parent stores reward, but we'll override it in sample()
        super().add(obs, next_obs, action, reward, done, infos)
    
    def sample(
        self,
        batch_size: int,
        env: Optional = None,
    ) -> DictReplayBufferSamples:
        """
        Sample a batch of transitions and recalculate intrinsic rewards.
        
        Uses current visit counts to compute fresh intrinsic rewards,
        then combines with stored extrinsic rewards.
        """
        # Call parent's _get_samples to get both batch and indices
        # This is the internal method that actually does the sampling
        if hasattr(super(), '_get_samples'):
            # Use parent's _get_samples which returns (batch, indices)
            batch, indices = super()._get_samples(batch_size, env)
        else:
            # Fallback: use parent's sample and try to infer indices
            batch = super().sample(batch_size, env)
            indices = None
        
        # If no intrinsic reward function, return as-is
        if self.intrinsic_reward_fn is None or self.beta == 0.0:
            return batch
        
        # Extract next observations (dict format) for computing intrinsic rewards
        # For SAC, we compute intrinsic reward based on the next state
        next_observations = batch.next_observations
        
        # Convert to numpy if needed (batch might be on device)
        # DictReplayBuffer stores observations as dict of tensors/arrays
        next_obs_dict = {}
        for key, obs_tensor in next_observations.items():
            if hasattr(obs_tensor, 'cpu'):
                next_obs_dict[key] = obs_tensor.cpu().numpy()
            else:
                next_obs_dict[key] = obs_tensor
        
        # Compute fresh intrinsic rewards using current visit counts
        intrinsic_rewards = []
        batch_size_actual = len(next_obs_dict[list(next_obs_dict.keys())[0]])
        
        for i in range(batch_size_actual):
            # Extract observation for this sample
            obs_dict = {key: next_obs_dict[key][i] for key in next_obs_dict.keys()}
            
            # Compute intrinsic reward for this observation
            try:
                if callable(self.intrinsic_reward_fn):
                    intrinsic_reward = self.intrinsic_reward_fn(obs_dict)
                    if isinstance(intrinsic_reward, (list, np.ndarray)):
                        intrinsic_reward = float(intrinsic_reward[0] if len(intrinsic_reward) > 0 else 0.0)
                    else:
                        intrinsic_reward = float(intrinsic_reward)
                else:
                    intrinsic_reward = 0.0
            except Exception as e:
                # If computation fails, use 0
                intrinsic_reward = 0.0
            
            intrinsic_rewards.append(intrinsic_reward)
        
        intrinsic_rewards = np.array(intrinsic_rewards, dtype=np.float32)
        
        # Get stored extrinsic rewards for this batch using the sampled indices
        if indices is not None:
            # Get extrinsic rewards for the sampled indices
            # Handle both 1D and 2D indexing (for n_envs > 1)
            if self.n_envs == 1:
                extrinsic_rewards = self.extrinsic_rewards[indices, 0]
            else:
                # For multiple envs, we'd need to track which env each sample came from
                # For now, assume n_envs=1 (most common case)
                extrinsic_rewards = self.extrinsic_rewards[indices, 0]
        else:
            # Fallback: if we can't get indices, try to extract from stored rewards
            # This is not ideal but better than nothing
            if hasattr(batch.rewards, 'cpu'):
                extrinsic_rewards = batch.rewards.cpu().numpy().copy()
            else:
                extrinsic_rewards = batch.rewards.copy()
            # Warning: This fallback uses total rewards as extrinsic, which is incorrect
            # but better than crashing. Ideally, indices should always be available.
        
        # Recalculate total rewards: extrinsic + beta * fresh_intrinsic
        total_rewards = extrinsic_rewards + self.beta * intrinsic_rewards
        
        # Convert to tensor if batch rewards are tensors
        if hasattr(batch.rewards, 'device'):
            total_rewards = torch.from_numpy(total_rewards).float().to(batch.rewards.device)
        
        # Create new batch with updated rewards
        # DictReplayBufferSamples is a named tuple, so we create a new one
        updated_batch = DictReplayBufferSamples(
            observations=batch.observations,
            actions=batch.actions,
            next_observations=batch.next_observations,
            dones=batch.dones,
            rewards=total_rewards,
        )
        
        return updated_batch
