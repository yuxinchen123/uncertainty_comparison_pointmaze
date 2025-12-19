"""
Evaluation script for RL performance comparison across different intrinsic reward methods.
"""
import os
import sys
import argparse
import numpy as np
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

from config import RLConfig, get_default_config
from wrappers.intrinsic_reward_wrapper import IntrinsicRewardWrapper
from uncertainty.integration import create_uncertainty_method
from uncertainty.gt_intrinsic import GTIntrinsicReward
from evaluation.gt_tracker import GroundTruthTracker
from evaluation.metrics import RLPerformanceMetrics
from utils.env_utils import make_pointmaze_env


def evaluate_model(model, env, n_episodes=10, deterministic=True):
    """
    Evaluate a trained model.
    
    Args:
        model: Trained RL model
        env: Environment (wrapped)
        n_episodes: Number of episodes to evaluate
        deterministic: Use deterministic policy
        
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    metrics = RLPerformanceMetrics()
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        episode_intrinsic = 0.0
        episode_extrinsic = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_return += reward
            episode_length += 1
            
            # Extract intrinsic/extrinsic rewards from info
            if 'intrinsic_reward' in info:
                episode_intrinsic += info['intrinsic_reward']
            if 'extrinsic_reward' in info:
                episode_extrinsic += info['extrinsic_reward']
        
        # Determine success (you may need to adjust this based on your task)
        success = episode_return > 0  # Placeholder - adjust based on task
        
        metrics.record_episode(
            episode_return=episode_return,
            episode_length=episode_length,
            intrinsic_reward=episode_intrinsic,
            extrinsic_reward=episode_extrinsic,
            success=success
        )
    
    return metrics.get_statistics()


def compare_methods(method_configs, config_base, n_episodes=10):
    """
    Compare multiple intrinsic reward methods.
    
    Args:
        method_configs: List of dictionaries with method configurations
            Each dict should have: 'name', 'uncertainty_method', 'uncertainty_config', 'beta'
        config_base: Base RLConfig object
        n_episodes: Number of episodes per method
        
    Returns:
        results: Dictionary mapping method names to evaluation metrics
    """
    results = {}
    
    for method_config in method_configs:
        method_name = method_config['name']
        print(f"\n{'=' * 60}")
        print(f"Evaluating method: {method_name}")
        print(f"{'=' * 60}")
        
        # Create config for this method
        config = get_default_config()
        config.__dict__.update(config_base.__dict__)
        config.uncertainty_method = method_config.get('uncertainty_method', config_base.uncertainty_method)
        config.uncertainty_config = method_config.get('uncertainty_config', config_base.uncertainty_config)
        config.beta = method_config.get('beta', config_base.beta)
        config.use_gt_baseline = method_config.get('use_gt_baseline', False)
        
        # Create environment
        uncertainty_method = None
        if not config.use_gt_baseline:
            uncertainty_method = create_uncertainty_method(
                config.uncertainty_method,
                config.uncertainty_config,
                device=config.device
            )
        
        # Create GT tracker
        gt_tracker = GroundTruthTracker(
            grid_rows=config.grid_rows,
            grid_cols=config.grid_cols
        )
        
        # Create environment
        from train_rl import create_env
        env, gt_tracker = create_env(config, uncertainty_method, gt_tracker)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        # Load model (if path provided)
        model_path = method_config.get('model_path', None)
        if model_path and os.path.exists(model_path):
            if config.algorithm.lower() == 'sac':
                model = SAC.load(model_path, env=env)
            elif config.algorithm.lower() == 'ppo':
                model = PPO.load(model_path, env=env)
            else:
                raise ValueError(f"Unknown algorithm: {config.algorithm}")
        else:
            print(f"Warning: Model path not found: {model_path}")
            print("Skipping this method...")
            continue
        
        # Evaluate
        metrics = evaluate_model(model, env, n_episodes=n_episodes)
        results[method_name] = metrics
        
        # Print results
        print(f"\nResults for {method_name}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Get visit statistics
        wrapper = env.envs[0].env.env
        if isinstance(wrapper, IntrinsicRewardWrapper):
            visit_counts = wrapper.get_visit_counts()
            gt_tracker.update_from_visit_counts(visit_counts)
            gt_stats = gt_tracker.get_statistics()
            print(f"\nVisit statistics:")
            for key, value in gt_stats.items():
                print(f"  {key}: {value}")
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Evaluate RL models with different intrinsic reward methods')
    
    parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                       help='Paths to trained models')
    parser.add_argument('--method_names', type=str, nargs='+', required=True,
                       help='Names of methods (one per model path)')
    parser.add_argument('--uncertainty_methods', type=str, nargs='+',
                       help='Uncertainty method names (one per model)')
    parser.add_argument('--betas', type=float, nargs='+',
                       help='Beta values (one per model)')
    parser.add_argument('--n_episodes', type=int, default=10,
                       help='Number of episodes per method')
    parser.add_argument('--algorithm', type=str, default='sac', choices=['sac', 'ppo'],
                       help='RL algorithm')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device for computation')
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.model_paths) != len(args.method_names):
        raise ValueError("Number of model paths must match number of method names")
    
    # Create method configs
    method_configs = []
    for i, (model_path, method_name) in enumerate(zip(args.model_paths, args.method_names)):
        config = {
            'name': method_name,
            'model_path': model_path,
        }
        
        if args.uncertainty_methods and i < len(args.uncertainty_methods):
            config['uncertainty_method'] = args.uncertainty_methods[i]
        
        if args.betas and i < len(args.betas):
            config['beta'] = args.betas[i]
        
        method_configs.append(config)
    
    # Base config
    config_base = get_default_config()
    config_base.algorithm = args.algorithm
    config_base.device = args.device
    
    # Compare methods
    results = compare_methods(method_configs, config_base, n_episodes=args.n_episodes)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("=" * 60)
    print(f"{'Method':<30} {'Mean Return':<15} {'Success Rate':<15}")
    print("-" * 60)
    for method_name, metrics in results.items():
        mean_return = metrics.get('mean_episode_return', 0.0)
        success_rate = metrics.get('success_rate', 0.0)
        print(f"{method_name:<30} {mean_return:<15.4f} {success_rate:<15.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()

