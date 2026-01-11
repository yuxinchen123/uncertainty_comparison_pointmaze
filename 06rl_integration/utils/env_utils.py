"""
Environment utilities for PointMaze.
"""
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import sys
import os

# Add parent directories to path to import utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../01sweep_uncertainty/utilities'))
from environment import get_maze_map


def make_pointmaze_env(env_name='PointMaze_Large-v3', seed=None, continuing_task=False):
    """
    Create PointMaze environment.
    
    Args:
        env_name: Name of environment
        seed: Random seed
        continuing_task: If True, episode continues after reaching goal (new goal generated).
                         If False, episode terminates when goal is reached.
        
    Returns:
        env: Gymnasium environment
    """
    gym.register_envs(gymnasium_robotics)
    env = gym.make(env_name, continuing_task=continuing_task)
    
    if seed is not None:
        env.reset(seed=seed)
    
    return env


def get_env_info(env):
    """
    Get information about environment.
    
    Args:
        env: Gymnasium environment
        
    Returns:
        info: Dictionary with environment information
    """
    obs_space = env.observation_space
    action_space = env.action_space
    
    return {
        'observation_space': obs_space,
        'action_space': action_space,
        'observation_shape': obs_space.shape if hasattr(obs_space, 'shape') else None,
        'action_shape': action_space.shape if hasattr(action_space, 'shape') else None,
    }

