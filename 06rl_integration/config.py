"""
Configuration system for RL training with intrinsic rewards.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class RLConfig:
    """Main configuration for RL training"""
    
    # Algorithm
    algorithm: str = 'sac'  # 'sac' or 'ppo'
    
    # Environment
    env_name: str = 'PointMaze_Large-v3'
    grid_rows: int = 9
    grid_cols: int = 12
    
    # Uncertainty method
    uncertainty_method: str = 'rnd_linear_ls'  # Method name
    uncertainty_config: Dict[str, Any] = field(default_factory=lambda: {
        'feature_dim': 128,
        'theta_seed': 42,
        'regularization': 1e-2,
        'update_frequency': 1,  # Update every step
        'num_epochs_per_update': 1,
    })
    
    # Intrinsic reward
    beta: float = 1.0  # Intrinsic reward coefficient (constant)
    use_gt_baseline: bool = False  # If True, use GT as intrinsic reward instead
    
    # Training
    total_timesteps: int = 100000
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1000000
    
    # SAC-specific (if algorithm == 'sac')
    sac_config: Dict[str, Any] = field(default_factory=lambda: {
        'tau': 0.005,  # Soft update coefficient
        'gamma': 0.99,  # Discount factor
        'train_freq': 1,  # Train every N steps
        'gradient_steps': 1,  # Gradient steps per update
        'learning_starts': 100,  # Steps before learning starts (SB3 default)
    })
    
    # PPO-specific (if algorithm == 'ppo')
    ppo_config: Dict[str, Any] = field(default_factory=lambda: {
        'n_steps': 2048,  # Steps per rollout
        'batch_size': 64,  # Mini-batch size
        'n_epochs': 10,  # Number of epochs per update
        'gamma': 0.99,  # Discount factor
        'gae_lambda': 0.95,  # GAE lambda
        'clip_range': 0.2,  # PPO clip range
        'ent_coef': 0.0,  # Entropy coefficient (SB3 default)
    })
    
    # Evaluation
    eval_freq: int = 5000  # Evaluate every N steps
    n_eval_episodes: int = 10  # Number of episodes for evaluation
    
    # Logging
    log_dir: str = './logs'
    tensorboard_log: Optional[str] = None
    verbose: int = 1
    wandb_switch: bool = True  # Enable WandB logging
    
    # Device
    device: str = 'cpu'  # 'cpu' or 'cuda'
    
    # Seed
    seed: int = 42


def get_default_config():
    """Get default configuration"""
    return RLConfig()


def update_config_from_dict(config: RLConfig, config_dict: Dict[str, Any]):
    """
    Update configuration from dictionary.
    
    Args:
        config: RLConfig object to update
        config_dict: Dictionary with configuration values
    """
    for key, value in config_dict.items():
        if hasattr(config, key):
            if isinstance(getattr(config, key), dict) and isinstance(value, dict):
                # Merge dictionaries
                getattr(config, key).update(value)
            else:
                setattr(config, key, value)
        else:
            print(f"Warning: Unknown config key: {key}")

