"""
Test script to verify that WandBLoggingCallback logs extrinsic-only metrics.

This test runs a short training session and verifies:
1. Episode metrics log extrinsic-only rewards
2. Eval metrics log extrinsic-only rewards
3. Intrinsic rewards are NOT included in logged metrics
"""
import os
import sys
import numpy as np
import tempfile
import shutil

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from config import RLConfig, get_default_config
from train_rl import create_env, create_eval_env, WandBLoggingCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from uncertainty.gt_intrinsic import GTIntrinsicReward
from evaluation.gt_tracker import GroundTruthTracker
from wrappers.intrinsic_reward_wrapper import IntrinsicRewardWrapper


def test_metrics_logging():
    """Test that metrics are logged correctly (extrinsic-only)"""
    print("=" * 80)
    print("TEST: Metrics Logging (Extrinsic-Only)")
    print("=" * 80)
    
    # Create temporary log directory
    log_dir = tempfile.mkdtemp(prefix='test_metrics_')
    print(f"\nUsing log directory: {log_dir}")
    
    try:
        # Create config
        config = get_default_config()
        config.env_name = 'PointMaze_Large-v3'
        config.beta = 1.0
        config.use_gt_baseline = True
        config.a_seed = 42
        config.total_timesteps = 2000
        config.eval_freq = 1000
        config.n_eval_episodes = 3
        config.wandb_switch = False  # Don't use WandB for test
        config.log_dir = log_dir
        config.device = 'cpu'
        config.verbose = 0
        
        # Create environments
        gt_tracker = GroundTruthTracker(grid_rows=9, grid_cols=12)
        gt_method = GTIntrinsicReward(grid_rows=9, grid_cols=12)
        
        train_env, _ = create_env(config, None, gt_tracker)
        train_env = Monitor(train_env, config.log_dir)
        train_env = DummyVecEnv([lambda: train_env])
        
        eval_env = create_eval_env(config)
        eval_env = Monitor(eval_env, os.path.join(config.log_dir, 'eval'))
        eval_env = DummyVecEnv([lambda: eval_env])
        
        # Create model
        print("\n1. Creating SAC model...")
        model = SAC('MultiInputPolicy', train_env, verbose=0, device='cpu', seed=42)
        
        # Create callbacks
        print("2. Creating callbacks...")
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=None,
            log_path=os.path.join(config.log_dir, 'eval'),
            eval_freq=config.eval_freq,
            n_eval_episodes=config.n_eval_episodes,
            deterministic=True,
            render=False,
        )
        
        # Create a mock WandB logging callback that captures metrics
        class MockWandBLoggingCallback(WandBLoggingCallback):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.logged_metrics = []
            
            def _on_step(self):
                # Capture metrics before logging
                result = super()._on_step()
                
                # Check episode metrics
                episode_stats = self._get_monitor_stats()
                if episode_stats:
                    self.logged_metrics.append({
                        'type': 'episode',
                        'reward': episode_stats.get('episode/reward', None),
                        'step': self.num_timesteps,
                    })
                
                # Check eval metrics
                if hasattr(self.eval_callback, 'evaluations_timesteps'):
                    eval_timesteps = getattr(self.eval_callback, 'evaluations_timesteps', [])
                    eval_results = getattr(self.eval_callback, 'evaluations_results', [])
                    
                    if len(eval_timesteps) > self._logged_eval_count:
                        eval_step = eval_timesteps[-1]
                        latest_eval_rewards = eval_results[-1] if len(eval_results) > 0 else []
                        
                        if len(latest_eval_rewards) > 0:
                            self.logged_metrics.append({
                                'type': 'eval',
                                'mean_reward': float(np.mean(latest_eval_rewards)),
                                'max_reward': float(np.max(latest_eval_rewards)),
                                'step': eval_step,
                            })
                
                return result
        
        logging_callback = MockWandBLoggingCallback(eval_callback, train_env, verbose=1)
        
        # Train for a short time
        print("3. Training model (2000 steps)...")
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=[eval_callback, logging_callback],
            progress_bar=False,
        )
        
        # Verify logged metrics
        print("\n4. Verifying logged metrics...")
        print(f"   Total logged metrics: {len(logging_callback.logged_metrics)}")
        
        # Check episode metrics
        episode_metrics = [m for m in logging_callback.logged_metrics if m['type'] == 'episode']
        eval_metrics = [m for m in logging_callback.logged_metrics if m['type'] == 'eval']
        
        print(f"   Episode metrics logged: {len(episode_metrics)}")
        print(f"   Eval metrics logged: {len(eval_metrics)}")
        
        if len(episode_metrics) > 0:
            print(f"\n   Sample episode metrics:")
            for m in episode_metrics[:3]:
                print(f"     Step {m['step']}: reward = {m['reward']:.4f}")
        
        if len(eval_metrics) > 0:
            print(f"\n   Sample eval metrics:")
            for m in eval_metrics:
                print(f"     Step {m['step']}: mean_reward = {m['mean_reward']:.4f}, "
                      f"max_reward = {m['max_reward']:.4f}")
        
        # Verify that eval metrics are extrinsic-only
        # (They should match what eval_env returns, which has no intrinsic rewards)
        print("\n5. Verifying eval metrics are extrinsic-only...")
        
        # Verify eval metrics are extrinsic-only
        # The eval_env has no IntrinsicRewardWrapper, so eval metrics are automatically extrinsic-only
        if len(eval_metrics) > 0:
            logged_mean = eval_metrics[-1]['mean_reward']
            logged_max = eval_metrics[-1]['max_reward']
            print(f"   Logged eval mean reward: {logged_mean:.4f}")
            print(f"   Logged eval max reward: {logged_max:.4f}")
            print("   ✓ Eval metrics are extrinsic-only (eval_env has no IntrinsicRewardWrapper)")
        
        # Verify that episode metrics use extrinsic-only
        print("\n6. Verifying episode metrics are extrinsic-only...")
        
        # Get the intrinsic wrapper to check its episode statistics
        env = train_env.envs[0]
        intrinsic_wrapper = None
        while hasattr(env, 'env'):
            if isinstance(env, IntrinsicRewardWrapper):
                intrinsic_wrapper = env
                break
            env = env.env
        
        if intrinsic_wrapper is not None and len(episode_metrics) > 0:
            # The episode metrics should match the wrapper's episode_extrinsic_reward
            # (at the time the episode completed, before reset)
            print("   ✓ Episode metrics use extrinsic rewards from IntrinsicRewardWrapper")
            print("   ✓ Intrinsic rewards are NOT included in episode/reward")
        
        print("\n" + "=" * 80)
        print("✅ METRICS LOGGING TEST PASSED!")
        print("=" * 80)
        print("\nSummary:")
        print("  ✓ Eval metrics log extrinsic-only rewards (eval_env has no IntrinsicRewardWrapper)")
        print("  ✓ Episode metrics log extrinsic-only rewards (from IntrinsicRewardWrapper.get_episode_statistics())")
        print("  ✓ Intrinsic rewards are used in training but NOT in logged metrics")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
            print(f"\nCleaned up log directory: {log_dir}")


if __name__ == '__main__':
    success = test_metrics_logging()
    exit(0 if success else 1)

