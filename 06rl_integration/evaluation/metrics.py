"""
Evaluation metrics for RL performance (episode returns, success rate, etc.).
"""
import numpy as np
from collections import deque


class RLPerformanceMetrics:
    """
    Tracks RL performance metrics during training.
    Focuses on episode returns, success rate, episode length, etc.
    """
    
    def __init__(self, window_size=100):
        """
        Args:
            window_size: Window size for moving averages
        """
        self.window_size = window_size
        
        # Episode metrics
        self.episode_returns = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.episode_intrinsic_rewards = deque(maxlen=window_size)
        self.episode_extrinsic_rewards = deque(maxlen=window_size)
        self.episode_success = deque(maxlen=window_size)  # True/False for each episode
        
        # Cumulative statistics
        self.total_episodes = 0
        self.total_steps = 0
    
    def record_episode(self, episode_return, episode_length, intrinsic_reward=0.0, 
                      extrinsic_reward=0.0, success=False):
        """
        Record metrics for a completed episode.
        
        Args:
            episode_return: Total episode return (extrinsic + intrinsic)
            episode_length: Episode length in steps
            intrinsic_reward: Total intrinsic reward in episode
            extrinsic_reward: Total extrinsic reward in episode
            success: Whether episode was successful (task completion)
        """
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)
        self.episode_intrinsic_rewards.append(intrinsic_reward)
        self.episode_extrinsic_rewards.append(extrinsic_reward)
        self.episode_success.append(success)
        
        self.total_episodes += 1
        self.total_steps += episode_length
    
    def get_statistics(self):
        """Get current statistics"""
        if len(self.episode_returns) == 0:
            return {
                'mean_episode_return': 0.0,
                'std_episode_return': 0.0,
                'mean_episode_length': 0.0,
                'mean_intrinsic_reward': 0.0,
                'mean_extrinsic_reward': 0.0,
                'success_rate': 0.0,
                'total_episodes': 0,
                'total_steps': 0,
            }
        
        return {
            'mean_episode_return': float(np.mean(self.episode_returns)),
            'std_episode_return': float(np.std(self.episode_returns)),
            'min_episode_return': float(np.min(self.episode_returns)),
            'max_episode_return': float(np.max(self.episode_returns)),
            'mean_episode_length': float(np.mean(self.episode_lengths)),
            'mean_intrinsic_reward': float(np.mean(self.episode_intrinsic_rewards)),
            'mean_extrinsic_reward': float(np.mean(self.episode_extrinsic_rewards)),
            'success_rate': float(np.mean(self.episode_success)),
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
        }
    
    def reset(self):
        """Reset all metrics"""
        self.episode_returns.clear()
        self.episode_lengths.clear()
        self.episode_intrinsic_rewards.clear()
        self.episode_extrinsic_rewards.clear()
        self.episode_success.clear()
        self.total_episodes = 0
        self.total_steps = 0

