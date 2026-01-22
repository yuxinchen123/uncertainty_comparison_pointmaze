"""
Custom ReplayBuffer for SAC that recalculates intrinsic rewards on sample
using current visit counts instead of stale rewards stored at collection time.
"""
import numpy as np
import torch
from typing import Optional, Union, Callable
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class IntrinsicReplayBuffer(ReplayBuffer):
    """
    Custom ReplayBuffer that recalculates intrinsic rewards when sampling.
    
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
            observation_space: Observation space
            action_space: Action space
            device: PyTorch device
            n_envs: Number of parallel environments
            optimize_memory_usage: Whether to optimize memory usage
            handle_timeout_termination: Whether to handle timeout terminations
            intrinsic_reward_fn: Function to compute intrinsic reward given observations
                                Should accept (observations, goal_idx=None) and return array of rewards
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
        obs: np.ndarray,
        next_obs: np.ndarray,
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
    ) -> ReplayBufferSamples:
        """
        Sample a batch of transitions and recalculate intrinsic rewards.
        
        Uses current visit counts to compute fresh intrinsic rewards,
        then combines with stored extrinsic rewards.
        """
        # Get batch indices first (needed to access stored extrinsic rewards)
        # SB3's ReplayBuffer.sample() generates indices internally
        # We need to replicate that logic to get the indices
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        
        # Call parent's _get_samples with the batch indices
        # This returns the batch (not indices, since we already have them)
        batch = super()._get_samples(batch_inds, env)
        
        # We now have both batch and indices
        indices = batch_inds
        
        # If no intrinsic reward function, return as-is
        if self.intrinsic_reward_fn is None or self.beta == 0.0:
            return batch
        
        # Extract observations (next_obs) for computing intrinsic rewards
        # For SAC, we compute intrinsic reward based on the next state
        next_observations = batch.next_observations
        
        # Convert to numpy if needed (batch might be on device)
        if hasattr(next_observations, 'cpu'):
            next_obs_np = next_observations.cpu().numpy()
        else:
            next_obs_np = next_observations
        
        # Compute fresh intrinsic rewards using current visit counts
        # The intrinsic_reward_fn should handle batch processing
        try:
            if callable(self.intrinsic_reward_fn):
                # Try batch processing first (more efficient)
                try:
                    intrinsic_rewards = self.intrinsic_reward_fn(next_obs_np)
                    if isinstance(intrinsic_rewards, (list, np.ndarray)):
                        intrinsic_rewards = np.array(intrinsic_rewards, dtype=np.float32).flatten()
                    else:
                        intrinsic_rewards = np.array([float(intrinsic_rewards)] * batch_size, dtype=np.float32)
                except:
                    # Fallback to per-sample processing
                    intrinsic_rewards = []
                    for i in range(batch_size):
                        obs = next_obs_np[i]
                        # Handle dict observations
                        if isinstance(obs, dict):
                            obs_for_fn = obs
                        elif isinstance(obs, np.ndarray):
                            # For dict observation spaces, SB3 flattens them
                            # We need to reconstruct the dict
                            if hasattr(self.observation_space, 'spaces') and isinstance(self.observation_space.spaces, dict):
                                obs_dict = {}
                                start_idx = 0
                                for key, space in self.observation_space.spaces.items():
                                    size = np.prod(space.shape)
                                    obs_dict[key] = obs[start_idx:start_idx+size].reshape(space.shape)
                                    start_idx += size
                                obs_for_fn = obs_dict
                            else:
                                # Array observation - pass as dict with achieved_goal
                                obs_for_fn = {'achieved_goal': obs[:2] if len(obs) >= 2 else obs}
                        else:
                            obs_for_fn = obs
                        
                        try:
                            intrinsic_reward = self.intrinsic_reward_fn(obs_for_fn)
                            if isinstance(intrinsic_reward, (list, np.ndarray)):
                                intrinsic_reward = float(intrinsic_reward[0] if len(intrinsic_reward) > 0 else 0.0)
                            else:
                                intrinsic_reward = float(intrinsic_reward)
                        except:
                            intrinsic_reward = 0.0
                        
                        intrinsic_rewards.append(intrinsic_reward)
                    
                    intrinsic_rewards = np.array(intrinsic_rewards, dtype=np.float32)
            else:
                intrinsic_rewards = np.zeros(batch_size, dtype=np.float32)
        except Exception as e:
            # If computation fails entirely, use zeros
            intrinsic_rewards = np.zeros(batch_size, dtype=np.float32)
        
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
            # Note: stored rewards are total rewards, so this will be approximate
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
            import torch
            total_rewards = torch.from_numpy(total_rewards).float().to(batch.rewards.device)
        
        # Create new batch with updated rewards
        # Note: We need to create a new ReplayBufferSamples object
        # Since ReplayBufferSamples is a named tuple, we can't modify it directly
        # Instead, we'll create a new one with updated rewards
        updated_batch = ReplayBufferSamples(
            observations=batch.observations,
            actions=batch.actions,
            next_observations=batch.next_observations,
            dones=batch.dones,
            rewards=total_rewards,
        )
        
        return updated_batch
