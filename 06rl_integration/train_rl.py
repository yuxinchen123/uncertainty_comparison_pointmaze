"""
Main training script for RL with intrinsic rewards using Stable Baselines 3.
"""
import os
import sys
import argparse
import numpy as np
import torch
import wandb
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Try to import WandbCallback (available in sb3-contrib or newer versions)
try:
    from stable_baselines3.common.callbacks import WandbCallback
    WANDB_CALLBACK_AVAILABLE = True
except ImportError:
    try:
        from sb3_contrib.common.callbacks import WandbCallback
        WANDB_CALLBACK_AVAILABLE = True
    except ImportError:
        WANDB_CALLBACK_AVAILABLE = False
        print("⚠️  WandbCallback not available. WandB logging will use manual logging instead.")

# Add paths - IMPORTANT: Add current directory first to prioritize local packages
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import local packages BEFORE any modules that add 01sweep_uncertainty/utilities to path
# This prevents naming conflicts with evaluation.py in utilities
from config import RLConfig, get_default_config, update_config_from_dict
from evaluation.gt_tracker import GroundTruthTracker
from evaluation.metrics import RLPerformanceMetrics
from utils.env_utils import make_pointmaze_env, get_env_info
from wrappers.intrinsic_reward_wrapper import IntrinsicRewardWrapper
from uncertainty.integration import create_uncertainty_method
from uncertainty.gt_intrinsic import GTIntrinsicReward


class IntrinsicRewardCallback:
    """
    Callback to update uncertainty models during training.
    Updates models every N steps as specified in config.
    """
    
    def __init__(self, wrapper, update_frequency=1):
        """
        Args:
            wrapper: IntrinsicRewardWrapper instance
            update_frequency: Update every N steps
        """
        self.wrapper = wrapper
        self.update_frequency = update_frequency
        self.step_count = 0
    
    def __call__(self, locals_, globals_):
        """Called during training"""
        self.step_count += 1
        
        # Update uncertainty model if frequency reached
        if self.update_frequency > 0 and self.step_count % self.update_frequency == 0:
            visited_states = self.wrapper.get_visited_states()
            if len(visited_states) > 0:
                # Update uncertainty method
                if hasattr(self.wrapper.uncertainty_method, 'update_with_batch'):
                    self.wrapper.uncertainty_method.update_with_batch(visited_states)
                elif hasattr(self.wrapper.uncertainty_method, 'method'):
                    # If wrapped in adapter
                    if hasattr(self.wrapper.uncertainty_method, 'update_with_batch'):
                        self.wrapper.uncertainty_method.update_with_batch(visited_states)
        
        return True


def create_env(config, uncertainty_method, gt_tracker=None):
    """
    Create and wrap environment with intrinsic rewards.
    
    Args:
        config: RLConfig object
        uncertainty_method: Uncertainty method object
        gt_tracker: GroundTruthTracker (optional, for GT baseline)
        
    Returns:
        env: Wrapped environment
    """
    # Create base environment
    env = make_pointmaze_env(config.env_name, seed=config.a_seed)
    
    # Get maze map
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../01sweep_uncertainty/utilities'))
    from environment import get_maze_map
    try:
        maze_map = get_maze_map()
    except:
        maze_map = None
    
    # Wrap with intrinsic rewards
    if config.use_gt_baseline:
        # Use GT as intrinsic reward
        if gt_tracker is None:
            gt_tracker = GroundTruthTracker(
                grid_rows=config.grid_rows,
                grid_cols=config.grid_cols,
                maze_map=maze_map
            )
        gt_method = GTIntrinsicReward(
            grid_rows=config.grid_rows,
            grid_cols=config.grid_cols,
            maze_map=maze_map
        )
        env = IntrinsicRewardWrapper(
            env,
            uncertainty_method=gt_method,
            beta=config.beta,
            grid_rows=config.grid_rows,
            grid_cols=config.grid_cols,
            maze_map=maze_map
        )
    else:
        # Use uncertainty method
        env = IntrinsicRewardWrapper(
            env,
            uncertainty_method=uncertainty_method,
            beta=config.beta,
            grid_rows=config.grid_rows,
            grid_cols=config.grid_cols,
            maze_map=maze_map
        )
    
    return env, gt_tracker


def train(config: RLConfig):
    """
    Main training function.
    
    Args:
        config: RLConfig object with training configuration
    """
    print("=" * 60)
    print("RL Training with Intrinsic Rewards")
    print("=" * 60)
    print(f"Algorithm: {config.algorithm.upper()}")
    print(f"Uncertainty method: {config.uncertainty_method}")
    print(f"Beta (intrinsic coefficient): {config.beta}")
    print(f"Total timesteps: {config.total_timesteps}")
    print("=" * 60)
    
    # Set seeds
    np.random.seed(config.a_seed)
    torch.manual_seed(config.a_seed)
    if torch.cuda.is_available() and config.device == 'cuda':
        torch.cuda.manual_seed(config.a_seed)
    
    # Create uncertainty method (if not using GT baseline)
    uncertainty_method = None
    if not config.use_gt_baseline:
        print(f"\nCreating uncertainty method: {config.uncertainty_method}")
        uncertainty_method = create_uncertainty_method(
            config.uncertainty_method,
            config.uncertainty_config,
            device=config.device
        )
        print("✓ Uncertainty method created")
    
    # Create GT tracker
    gt_tracker = GroundTruthTracker(
        grid_rows=config.grid_rows,
        grid_cols=config.grid_cols
    )
    
    # Create training environment
    print("\nCreating training environment...")
    train_env, gt_tracker = create_env(config, uncertainty_method, gt_tracker)
    train_env = Monitor(train_env, config.log_dir)
    train_env = DummyVecEnv([lambda: train_env])
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env, _ = create_env(config, uncertainty_method, gt_tracker)
    eval_env = Monitor(eval_env, os.path.join(config.log_dir, 'eval'))
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Get environment info
    env_info = get_env_info(train_env.envs[0].env.env)
    print(f"\nEnvironment info:")
    print(f"  Observation space: {env_info['observation_space']}")
    print(f"  Action space: {env_info['action_space']}")
    
    # Create RL agent
    print(f"\nCreating {config.algorithm.upper()} agent...")
    # PointMaze uses Dict observation space, so we need MultiInputPolicy
    if config.algorithm.lower() == 'sac':
        model = SAC(
            'MultiInputPolicy',
            train_env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            tau=config.sac_config['tau'],
            gamma=config.sac_config['gamma'],
            train_freq=config.sac_config['train_freq'],
            gradient_steps=config.sac_config['gradient_steps'],
            learning_starts=config.sac_config['learning_starts'],
            tensorboard_log=config.tensorboard_log,
            verbose=config.verbose,
            device=config.device,
            seed=config.a_seed,
        )
    elif config.algorithm.lower() == 'ppo':
        model = PPO(
            'MultiInputPolicy',
            train_env,
            learning_rate=config.learning_rate,
            n_steps=config.ppo_config['n_steps'],
            batch_size=config.ppo_config['batch_size'],
            n_epochs=config.ppo_config['n_epochs'],
            gamma=config.ppo_config['gamma'],
            gae_lambda=config.ppo_config['gae_lambda'],
            clip_range=config.ppo_config['clip_range'],
            ent_coef=config.ppo_config['ent_coef'],
            tensorboard_log=config.tensorboard_log,
            verbose=config.verbose,
            device=config.device,
            seed=config.a_seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")
    
    print("✓ Agent created")
    
    # Create callbacks
    callbacks = []
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(config.log_dir, 'best_model'),
        log_path=os.path.join(config.log_dir, 'eval'),
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.eval_freq,
        save_path=os.path.join(config.log_dir, 'checkpoints'),
        name_prefix='rl_model',
    )
    callbacks.append(checkpoint_callback)
    
    # WandB callback (if enabled and available)
    if config.wandb_switch and wandb.run is not None:
        if WANDB_CALLBACK_AVAILABLE:
            wandb_callback = WandbCallback(
                gradient_save_freq=0,  # Don't save gradients (saves space)
                model_save_freq=0,      # Don't save models (saves space)
                verbose=1,
            )
            callbacks.append(wandb_callback)
        else:
            # WandB logging will happen through Monitor wrapper and manual logging
            print("ℹ️  Using manual WandB logging (WandbCallback not available)")
    
    # Note: Uncertainty model updates happen in the wrapper's step method
    # The wrapper calls update_with_batch every step, and the adapter handles
    # the update frequency internally
    
    # Train
    print(f"\nStarting training for {config.total_timesteps} steps...")
    print("=" * 60)
    
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        log_interval=10,
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    # Save final model
    final_model_path = os.path.join(config.log_dir, 'final_model')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Print statistics
    wrapper = train_env.envs[0].env.env
    if isinstance(wrapper, IntrinsicRewardWrapper):
        stats = wrapper.get_statistics()
        print("\nTraining statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")
        
        # Update GT tracker with final visit counts
        visit_counts = wrapper.get_visit_counts()
        gt_tracker.update_from_visit_counts(visit_counts)
        gt_stats = gt_tracker.get_statistics()
        print("\nGround truth statistics:")
        for key, value in gt_stats.items():
            print(f"  {key}: {value}")
        
        # Log final statistics to WandB if enabled
        if config.wandb_switch and wandb.run is not None:
            wandb.log({
                "final_stats": stats,
                "final_gt_stats": gt_stats
            })
    
    return model, gt_tracker


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train RL agent with intrinsic rewards')
    
    # Algorithm
    parser.add_argument('--algorithm', type=str, default='sac', choices=['sac', 'ppo'],
                       help='RL algorithm to use')
    
    # Uncertainty method
    parser.add_argument('--uncertainty_method', type=str, default='rnd_linear_ls',
                       help='Uncertainty method name')
    parser.add_argument('--use_gt_baseline', action='store_true',
                       help='Use GT as intrinsic reward baseline')
    
    # Intrinsic reward
    parser.add_argument('--beta', type=float, default=1.0,
                       help='Intrinsic reward coefficient')
    
    # Training
    parser.add_argument('--total_timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    
    # Evaluation
    parser.add_argument('--eval_freq', type=int, default=5000,
                       help='Evaluation frequency (steps)')
    parser.add_argument('--n_eval_episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Log directory')
    parser.add_argument('--tensorboard_log', type=str, default=None,
                       help='Tensorboard log directory')
    
    # Device and seed
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device for computation')
    parser.add_argument('--a_seed', type=int, default=42,
                       help='Random seed for data sampling and model initialization (consistent with 01-05 folders)')
    
    # Environment parameters
    parser.add_argument('--env_name', type=str, default='PointMaze_Large-v3',
                       help='Environment name')
    parser.add_argument('--grid_rows', type=int, default=9,
                       help='Number of grid rows')
    parser.add_argument('--grid_cols', type=int, default=12,
                       help='Number of grid columns')
    
    # WandB logging
    parser.add_argument('--wandb_switch', type=str, default='true', choices=['true', 'false'],
                       help='Enable WandB logging (true/false)')
    
    # Uncertainty config (as JSON string or individual args)
    parser.add_argument('--feature_dim', type=int, default=128,
                       help='Feature dimension for linear methods')
    parser.add_argument('--update_frequency', type=int, default=1,
                       help='Uncertainty model update frequency (steps)')
    parser.add_argument('--num_epochs_per_update', type=int, default=1,
                       help='Number of epochs per uncertainty model update')
    
    args = parser.parse_args()
    
    # Handle WandB sweep mode (if running in a sweep, wandb.run is already initialized)
    if wandb.run is not None:
        # We're in a WandB sweep, use config values and override args
        sweep_config = wandb.config
        print("🔄 Running in WandB sweep mode")
        
        # Override arguments with WandB config
        args.algorithm = sweep_config.get('algorithm', args.algorithm)
        args.beta = sweep_config.get('beta', args.beta)
        args.a_seed = sweep_config.get('a_seed', args.a_seed)
        args.total_timesteps = sweep_config.get('total_timesteps', args.total_timesteps)
        args.eval_freq = sweep_config.get('eval_freq', args.eval_freq)
        args.n_eval_episodes = sweep_config.get('n_eval_episodes', args.n_eval_episodes)
        args.log_dir = sweep_config.get('log_dir', args.log_dir)
        args.device = sweep_config.get('device', args.device)
        args.wandb_switch = sweep_config.get('wandb_switch', args.wandb_switch)
        # Note: uncertainty_method is set but not used when beta=0
        args.uncertainty_method = sweep_config.get('uncertainty_method', args.uncertainty_method)
        # Environment parameters
        args.env_name = sweep_config.get('env_name', args.env_name)
        args.grid_rows = sweep_config.get('grid_rows', args.grid_rows)
        args.grid_cols = sweep_config.get('grid_cols', args.grid_cols)
    
    # Initialize WandB if not in sweep and wandb_switch is enabled
    wandb_switch = args.wandb_switch.lower() == 'true'
    if wandb_switch and wandb.run is None:
        wandb.init(
            project="rl-rnd-integration",
            name=f"{args.algorithm}_plain_rl_seed{args.a_seed}",
            config=vars(args),
            tags=["plain_rl", args.algorithm]
        )
    
    # Create config
    config = get_default_config()
    config.algorithm = args.algorithm
    config.uncertainty_method = args.uncertainty_method
    config.use_gt_baseline = args.use_gt_baseline
    config.beta = args.beta
    config.total_timesteps = args.total_timesteps
    config.learning_rate = args.learning_rate
    config.eval_freq = args.eval_freq
    config.n_eval_episodes = args.n_eval_episodes
    config.log_dir = args.log_dir
    config.tensorboard_log = args.tensorboard_log
    config.device = args.device
    config.a_seed = args.a_seed
    config.wandb_switch = wandb_switch
    # Environment parameters (from args, which are set from wandb.config in sweep mode)
    config.env_name = args.env_name
    config.grid_rows = args.grid_rows
    config.grid_cols = args.grid_cols
    
    # Update uncertainty config
    config.uncertainty_config.update({
        'feature_dim': args.feature_dim,
        'update_frequency': args.update_frequency,
        'num_epochs_per_update': args.num_epochs_per_update,
    })
    
    # Create log directory
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Train
    train(config)


if __name__ == '__main__':
    main()

