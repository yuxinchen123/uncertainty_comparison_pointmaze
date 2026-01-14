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
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Try to import WandbCallback (from wandb.integration.sb3)
try:
    from wandb.integration.sb3 import WandbCallback
    WANDB_CALLBACK_AVAILABLE = True
except ImportError:
    try:
        # Fallback: try sb3-contrib (older versions)
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
from utils.goal_utils import select_fixed_goal, select_diverse_goals
from wrappers.intrinsic_reward_wrapper import IntrinsicRewardWrapper
from wrappers.goal_wrapper import GoalWrapper
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


class DualEvalCallback(BaseCallback):
    """
    Custom callback that evaluates on both single-goal and continuation settings.
    Logs metrics separately for each evaluation type.
    """
    
    def __init__(self, eval_env_single_goal, eval_env_continuation, n_eval_episodes=10, 
                 eval_freq=5000, verbose=0):
        """
        Args:
            eval_env_single_goal: Evaluation environment with continuing_task=False
            eval_env_continuation: Evaluation environment with continuing_task=True
            n_eval_episodes: Number of episodes per evaluation
            eval_freq: Evaluate every N steps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env_single_goal = eval_env_single_goal
        self.eval_env_continuation = eval_env_continuation
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        
        # Track evaluation results
        self.eval_results_single_goal = []  # List of (step, rewards, lengths)
        self.eval_results_continuation = []  # List of (step, rewards, lengths)
        self.last_eval_step = -1
    
    def _on_step(self) -> bool:
        """Called at each step during training"""
        if self.num_timesteps % self.eval_freq != 0:
            return True
        
        if self.num_timesteps == self.last_eval_step:
            return True
        
        self.last_eval_step = self.num_timesteps
        
        # Evaluate on single-goal setting
        if self.verbose > 0:
            print(f"\nEvaluating on single-goal setting (step {self.num_timesteps})...")
        
        mean_reward_single, std_reward_single, episode_lengths_single = self._evaluate(
            self.eval_env_single_goal, 
            prefix="eval_single_goal"
        )
        
        self.eval_results_single_goal.append({
            'step': self.num_timesteps,
            'mean_reward': mean_reward_single,
            'std_reward': std_reward_single,
            'episode_lengths': episode_lengths_single,
        })
        
        # Evaluate on continuation setting
        if self.verbose > 0:
            print(f"Evaluating on continuation setting (step {self.num_timesteps})...")
        
        mean_reward_cont, std_reward_cont, episode_lengths_cont = self._evaluate(
            self.eval_env_continuation,
            prefix="eval_continuation"
        )
        
        self.eval_results_continuation.append({
            'step': self.num_timesteps,
            'mean_reward': mean_reward_cont,
            'std_reward': std_reward_cont,
            'episode_lengths': episode_lengths_cont,
        })
        
        return True
    
    def _evaluate(self, eval_env, prefix=""):
        """
        Evaluate the current policy on the given environment.
        
        Returns:
            mean_reward: Mean episode reward
            std_reward: Std of episode rewards
            episode_lengths: List of episode lengths
        """
        from stable_baselines3.common.evaluation import evaluate_policy
        
        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
            return_episode_rewards=True,
        )
        
        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        
        if self.verbose > 0:
            print(f"  {prefix}: mean_reward={mean_reward:.4f} ± {std_reward:.4f}, "
                  f"mean_length={np.mean(episode_lengths):.2f}")
        
        return mean_reward, std_reward, episode_lengths


class EnhancedEvalCallback(EvalCallback):
    """
    Enhanced evaluation callback that tracks both extrinsic and intrinsic+extrinsic rewards.
    Computes intrinsic rewards during evaluation to see if policy is maximizing total reward.
    """
    
    def __init__(self, eval_env, uncertainty_method=None, beta=1.0, 
                 grid_rows=9, grid_cols=12, maze_map=None, goal_mode='single',
                 num_goals=None, training_wrapper=None, *args, **kwargs):
        """
        Args:
            eval_env: Evaluation environment
            uncertainty_method: Uncertainty method for computing intrinsic rewards
            beta: Intrinsic reward coefficient
            grid_rows: Number of grid rows
            grid_cols: Number of grid columns
            maze_map: Maze map array
            goal_mode: 'single' or 'multi'
            num_goals: Number of goals for multi-goal mode
            training_wrapper: IntrinsicRewardWrapper from training env (for training visit counts)
            *args, **kwargs: Additional arguments for EvalCallback
        """
        super().__init__(eval_env, *args, **kwargs)
        self.uncertainty_method = uncertainty_method
        self.beta = beta
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.maze_map = maze_map
        self.goal_mode = goal_mode
        self.num_goals = num_goals
        self.training_wrapper = training_wrapper  # For accessing training visit counts
        
        # Track intrinsic rewards per evaluation
        self.evaluations_intrinsic_rewards = []  # List of lists (one per evaluation)
        self.evaluations_extrinsic_rewards = []  # List of lists (one per evaluation)
        self.evaluations_total_rewards = []  # List of lists (intrinsic + extrinsic)
        self.evaluations_length = []  # List of lists (episode lengths per evaluation)
        
        # Ensure evaluations_results is initialized (base callback may not initialize it immediately)
        if not hasattr(self, 'evaluations_results'):
            self.evaluations_results = []
        
        # Track last evaluation step to avoid duplicate evaluations
        self.last_eval_step = -1
        
        # Import utility for state to grid conversion
        import sys
        import os
        import importlib.util
        # Path from 06rl_integration to 01sweep_uncertainty/utilities
        # Use absolute path resolution
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_file_dir))  # Go up to RND
        utilities_path = os.path.join(project_root, '01sweep_uncertainty', 'utilities')
        evaluation_file = os.path.join(utilities_path, 'evaluation.py')
        if os.path.exists(evaluation_file):
            try:
                spec = importlib.util.spec_from_file_location("utilities_evaluation", evaluation_file)
                utilities_evaluation = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(utilities_evaluation)
                self.observation_to_grid = utilities_evaluation.observation_to_grid_notebook_exact
            except Exception as e:
                print(f"Warning: Could not import observation_to_grid_notebook_exact: {e}")
                self.observation_to_grid = None
        else:
            print(f"Warning: evaluation.py not found at {evaluation_file}")
            self.observation_to_grid = None
    
    def _state_to_grid(self, observation):
        """Map observation to grid cell (0-based)"""
        if self.observation_to_grid is not None:
            _, (row, col) = self.observation_to_grid(
                observation,
                grid_rows=self.grid_rows,
                grid_cols=self.grid_cols
            )
            return row - 1, col - 1
        else:
            # Fallback: simple extraction (not used in current implementation)
            if isinstance(observation, dict) and 'achieved_goal' in observation:
                x, y = observation['achieved_goal'][0], observation['achieved_goal'][1]
            elif isinstance(observation, np.ndarray):
                x, y = observation[0], observation[1]
            else:
                return 0, 0
            # Simple grid mapping (approximate)
            col = int((x + 6.0) / (12.0 / self.grid_cols))
            row = int((4.5 - y) / (9.0 / self.grid_rows))
            return max(0, min(row, self.grid_rows - 1)), max(0, min(col, self.grid_cols - 1))
    
    def _get_uncertainty(self, observation, goal_idx=None):
        """
        Get intrinsic reward (uncertainty) for an observation.
        Uses training visit counts to reflect what the agent was trained on.
        
        Args:
            observation: Current observation
            goal_idx: Current goal index (for multi-goal mode)
        """
        if self.uncertainty_method is None or self.beta == 0.0:
            return 0.0
        
        # Update GT method with training visit counts (for this goal if multi-goal)
        # This reflects the exploration state during training, which is what the agent learned
        if hasattr(self.uncertainty_method, 'update_visit_counts') and self.training_wrapper is not None:
            training_visit_counts = self.training_wrapper.get_visit_counts()
            if self.goal_mode == 'multi' and goal_idx is not None:
                # Multi-goal: use training visit counts for this goal
                if len(training_visit_counts.shape) == 3:
                    goal_visit_counts = training_visit_counts[goal_idx]
                    self.uncertainty_method.update_visit_counts(goal_visit_counts)
                else:
                    # Fallback: if shape is wrong, use all visit counts
                    self.uncertainty_method.update_visit_counts(training_visit_counts)
            else:
                # Single-goal: use training visit counts directly
                if len(training_visit_counts.shape) == 2:
                    self.uncertainty_method.update_visit_counts(training_visit_counts)
                else:
                    # Fallback: if shape is wrong, use first goal's counts or all
                    if len(training_visit_counts.shape) == 3:
                        self.uncertainty_method.update_visit_counts(training_visit_counts[0])
                    else:
                        self.uncertainty_method.update_visit_counts(training_visit_counts)
        
        # Extract position from observation
        # VecEnv returns batched observations, so we need to handle both batched and non-batched
        if isinstance(observation, dict) and 'achieved_goal' in observation:
            achieved_goal = observation['achieved_goal']
            # Handle batched (VecEnv) and non-batched observations
            if isinstance(achieved_goal, np.ndarray) and len(achieved_goal.shape) > 1:
                # Batched: shape (1, 2) or (batch_size, 2) - take first element
                position = achieved_goal[0, :2] if achieved_goal.shape[1] >= 2 else achieved_goal[0]
            else:
                # Non-batched: shape (2,) - take first 2 elements
                position = achieved_goal[:2] if len(achieved_goal) >= 2 else achieved_goal
        elif isinstance(observation, np.ndarray):
            # Handle batched and non-batched
            if len(observation.shape) > 1:
                position = observation[0, :2] if observation.shape[1] >= 2 else observation[0]
            else:
                position = observation[:2] if len(observation) >= 2 else observation
        else:
            return 0.0
        
        # Get uncertainty from method
        position_array = np.array([position]).reshape(1, -1)
        try:
            uncertainty = self.uncertainty_method.get_uncertainty(position_array)
            if isinstance(uncertainty, np.ndarray):
                uncertainty_val = float(uncertainty[0] if len(uncertainty) > 0 else 0.0)
            else:
                uncertainty_val = float(uncertainty)
            
            # Handle NaN and inf values
            if np.isnan(uncertainty_val) or np.isinf(uncertainty_val):
                return 0.0
            
            return uncertainty_val
        except:
            return 0.0
    
    def _on_step(self) -> bool:
        """
        Override to run evaluation once and collect all necessary information
        (extrinsic rewards, intrinsic rewards, episode lengths, total rewards).
        """
        # Check if it's time to evaluate (same logic as base EvalCallback)
        # Use num_timesteps for consistency with other callbacks in this codebase
        if (self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0 and 
            self.num_timesteps != self.last_eval_step):
            # Run evaluation once and collect all information
            episode_extrinsic_rewards = []
            episode_intrinsic_rewards = []
            episode_total_rewards = []
            episode_lengths = []
            
            for episode_idx in range(self.n_eval_episodes):
                # VecEnv.reset() returns just the observation, not (obs, info)
                obs = self.eval_env.reset()
                done = False
                episode_extrinsic = 0.0
                episode_intrinsic = 0.0
                episode_length = 0
                
                # Get current goal index (for multi-goal mode)
                goal_idx = None
                if self.goal_mode == 'multi':
                    if hasattr(self.eval_env, 'envs') and len(self.eval_env.envs) > 0:
                        env = self.eval_env.envs[0]
                        while hasattr(env, 'env'):
                            if hasattr(env, 'get_current_goal_idx'):
                                goal_idx = env.get_current_goal_idx()
                                break
                            env = env.env
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    # VecEnv.step() returns (obs, rewards, dones, infos)
                    # where rewards and dones are arrays, infos is a list of dicts
                    obs, rewards, dones, infos = self.eval_env.step(action)
                    done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones
                    reward = rewards[0] if isinstance(rewards, (list, np.ndarray)) else rewards
                    info = infos[0] if isinstance(infos, list) and len(infos) > 0 else {}
                    
                    episode_extrinsic += reward
                    episode_length += 1
                    
                    # Compute intrinsic reward using training visit counts
                    # This reflects what the agent was trained on
                    if self.uncertainty_method is not None and self.beta != 0.0:
                        intrinsic_reward = self._get_uncertainty(obs, goal_idx=goal_idx)
                        episode_intrinsic += intrinsic_reward * self.beta
                
                episode_extrinsic_rewards.append(episode_extrinsic)
                episode_intrinsic_rewards.append(episode_intrinsic)
                episode_total_rewards.append(episode_extrinsic + episode_intrinsic)
                episode_lengths.append(episode_length)
            
            # Store results (compatible with base EvalCallback interface)
            mean_reward = float(np.mean(episode_extrinsic_rewards))
            self.evaluations_results.append(episode_extrinsic_rewards)
            
            # Store our additional metrics
            self.evaluations_extrinsic_rewards.append(episode_extrinsic_rewards)
            self.evaluations_intrinsic_rewards.append(episode_intrinsic_rewards)
            self.evaluations_total_rewards.append(episode_total_rewards)
            self.evaluations_length.append(episode_lengths)
            
            # Update best model if needed (same logic as base EvalCallback)
            if self.best_mean_reward is None or mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
            
            # Log to logger if available
            if self.logger is not None:
                self.logger.record("eval/mean_reward", mean_reward)
                self.logger.record("eval/mean_ep_length", float(np.mean(episode_lengths)))
                if self.verbose > 0:
                    print(f"Eval num_timesteps={self.num_timesteps}, "
                          f"episode_reward={mean_reward:.2f} +/- {np.std(episode_extrinsic_rewards):.2f}, "
                          f"episode_length={np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")
            
            # Mark this step as evaluated
            self.last_eval_step = self.num_timesteps
        
        return True


class WandBLoggingCallback(BaseCallback):
    """
    Comprehensive callback to log all SB3 metrics directly to WandB.
    Reads from SB3's logger to capture all training, evaluation, rollout, and time metrics.
    Also tracks episode metrics from Monitor wrapper (episode/reward, episode/length, etc.)
    and enhanced evaluation metrics (mean, std, min, max).
    This bypasses TensorBoard syncing issues and works reliably in sweep mode.
    """
    
    def __init__(self, eval_callback, train_env, dual_eval_callback=None, verbose=0):
        """
        Args:
            eval_callback: EvalCallback instance to monitor for evaluation metrics (or None if using dual eval)
            train_env: Training environment (DummyVecEnv) to access Monitor wrapper
            dual_eval_callback: DualEvalCallback instance (if using dual evaluation)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.dual_eval_callback = dual_eval_callback
        self.train_env = train_env
        self.last_eval_step = -1
        self.last_logger_state = {}  # Track logger state to detect updates
        self._logged_eval_count = 0  # Track how many evaluations we've logged (for step alignment)
        self._logged_eval_steps = set()  # Track which evaluation steps we've already logged (to prevent duplicates)
        self.last_global_step_logged = -1  # Track last global_step we logged (for smooth plotting)
        self.global_step_log_interval = 1000  # Log global_step every N steps for smooth plots
        
        # Track episode statistics from Monitor
        self.last_episode_count = 0
        self.last_episode_rewards = []
        self.last_episode_lengths = []
        
        # Track extrinsic rewards separately (for logging episode/reward as extrinsic-only)
        # Monitor tracks reward_total (extrinsic + intrinsic), but we want extrinsic only
        self.last_episode_extrinsic_rewards = []  # Track extrinsic rewards per episode
        self.last_seen_episode_extrinsic = 0.0  # Track last seen episode extrinsic reward (before reset)
        self.last_seen_episode_intrinsic = 0.0  # Track last seen episode intrinsic reward (before reset)
        
        # Get Monitor wrapper and IntrinsicRewardWrapper from environment
        self.monitor = None
        self.intrinsic_wrapper = None
        if hasattr(train_env, 'envs') and len(train_env.envs) > 0:
            # Unwrap to get Monitor and IntrinsicRewardWrapper
            # DummyVecEnv -> Monitor -> IntrinsicRewardWrapper -> env
            env = train_env.envs[0]
            while hasattr(env, 'env'):
                if isinstance(env, Monitor):
                    self.monitor = env
                # Check if this is IntrinsicRewardWrapper
                if isinstance(env, IntrinsicRewardWrapper):
                    self.intrinsic_wrapper = env
                env = env.env
            if self.monitor is None and self.verbose > 0:
                print("⚠️  Could not find Monitor wrapper in environment")
            if self.intrinsic_wrapper is None and self.verbose > 0:
                print("⚠️  Could not find IntrinsicRewardWrapper in environment")
    
    def _get_monitor_stats(self):
        """
        Get episode statistics from Monitor wrapper.
        Returns both extrinsic-only metrics (episode/...) and extrinsic+intrinsic metrics (episode_extrinsic_intrinsic/...).
        """
        if self.monitor is None:
            return None
        
        try:
            # Monitor stores episode data in these attributes
            # Use get_episode_rewards() and get_episode_lengths() methods for reliable access
            episode_rewards = self.monitor.get_episode_rewards() if hasattr(self.monitor, 'get_episode_rewards') else getattr(self.monitor, 'episode_returns', [])
            episode_lengths = self.monitor.get_episode_lengths() if hasattr(self.monitor, 'get_episode_lengths') else getattr(self.monitor, 'episode_lengths', [])
            episode_count = len(episode_rewards)
            
            # Check if a new episode completed
            if episode_count > self.last_episode_count:
                # Get the latest episode data
                latest_length = episode_lengths[-1] if len(episode_lengths) > 0 else 0
                
                # Get extrinsic and intrinsic rewards from IntrinsicRewardWrapper
                # Note: When an episode completes, Monitor records it, but the wrapper might have
                # already been reset. However, Monitor records the episode BEFORE reset, so we check
                # the wrapper's current episode stats. If it's been reset (value is 0), we use the
                # last value we saw. Otherwise, we use the current value.
                if self.intrinsic_wrapper is not None:
                    episode_stats = self.intrinsic_wrapper.get_episode_statistics()
                    current_episode_extrinsic = episode_stats.get('episode_extrinsic_reward', 0.0)
                    current_episode_intrinsic = episode_stats.get('episode_intrinsic_reward', 0.0)
                    
                    # If the wrapper has been reset (current value is 0 or very small), use the last seen value
                    # Otherwise, use the current value (episode just completed, wrapper not reset yet)
                    if current_episode_extrinsic > 0.01:  # Episode not reset yet
                        latest_extrinsic_reward = current_episode_extrinsic
                        latest_intrinsic_reward = current_episode_intrinsic
                        self.last_seen_episode_extrinsic = current_episode_extrinsic
                        self.last_seen_episode_intrinsic = current_episode_intrinsic
                    else:  # Wrapper has been reset, use last seen value
                        latest_extrinsic_reward = self.last_seen_episode_extrinsic
                        latest_intrinsic_reward = self.last_seen_episode_intrinsic
                    
                    # Store for reference
                    self.last_episode_extrinsic_rewards.append(latest_extrinsic_reward)
                else:
                    # Fallback: if we can't access wrapper, use 0
                    latest_extrinsic_reward = 0.0
                    latest_intrinsic_reward = 0.0
                    if self.verbose > 0:
                        print(f"⚠️  Warning: No IntrinsicRewardWrapper found, using 0.0 for extrinsic reward")
                
                # Calculate extrinsic + intrinsic for the new metric group
                latest_total_reward = latest_extrinsic_reward + latest_intrinsic_reward
                
                self.last_episode_count = episode_count
                return {
                    'episode/reward': float(latest_extrinsic_reward),  # Extrinsic only!
                    'episode/intrinsic_reward': float(latest_intrinsic_reward),  # Intrinsic only!
                    'episode/length': int(latest_length),
                    'episode/number': episode_count,
                    'episode_extrinsic_intrinsic/reward': float(latest_total_reward),  # Extrinsic + intrinsic
                    'episode_extrinsic_intrinsic/length': int(latest_length),
                    'episode_extrinsic_intrinsic/number': episode_count,
                }
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️  Error reading Monitor stats: {e}")
        
        return None
    
    def _on_step(self) -> bool:
        """
        Called at each step during training.
        Following MRQ's approach: only log when events happen (evaluations, episode completions),
        not continuously. This ensures perfect step alignment across runs for smoother graphs.
        """
        if wandb.run is None:
            return True
        
        current_step = self.num_timesteps
        
        # 0. Log global_step at regular intervals for smooth plotting (independent of events)
        # This ensures the x-axis represents actual training steps and the line is smooth
        if current_step - self.last_global_step_logged >= self.global_step_log_interval:
            try:
                wandb.log({'global_step': current_step}, step=current_step, commit=True)
                self.last_global_step_logged = current_step
            except Exception as e:
                if self.verbose > 0:
                    print(f"⚠️  WandB global_step logging error: {e}")
        
        # 1. Check for dual evaluation results (if using dual eval)
        if self.dual_eval_callback is not None:
            # Check for new dual evaluation results
            if len(self.dual_eval_callback.eval_results_single_goal) > 0:
                latest_single = self.dual_eval_callback.eval_results_single_goal[-1]
                if latest_single['step'] == current_step:
                    # Log single-goal evaluation metrics
                    try:
                        wandb.log({'global_step': current_step}, step=current_step, commit=False)
                        wandb.log({'eval_single_goal/mean_reward': latest_single['mean_reward']}, step=current_step, commit=False)
                        wandb.log({'eval_single_goal/std_reward': latest_single['std_reward']}, step=current_step, commit=False)
                        if len(latest_single['episode_lengths']) > 0:
                            wandb.log({'eval_single_goal/mean_ep_length': float(np.mean(latest_single['episode_lengths']))}, step=current_step, commit=False)
                            wandb.log({'eval_single_goal/std_ep_length': float(np.std(latest_single['episode_lengths']))}, step=current_step, commit=False)
                            wandb.log({'eval_single_goal/min_ep_length': float(np.min(latest_single['episode_lengths']))}, step=current_step, commit=False)
                            wandb.log({'eval_single_goal/max_ep_length': float(np.max(latest_single['episode_lengths']))}, step=current_step, commit=False)
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"⚠️  WandB single-goal eval logging error: {e}")
            
            if len(self.dual_eval_callback.eval_results_continuation) > 0:
                latest_cont = self.dual_eval_callback.eval_results_continuation[-1]
                if latest_cont['step'] == current_step:
                    # Log continuation evaluation metrics
                    try:
                        wandb.log({'global_step': current_step}, step=current_step, commit=False)
                        wandb.log({'eval_continuation/mean_reward': latest_cont['mean_reward']}, step=current_step, commit=False)
                        wandb.log({'eval_continuation/std_reward': latest_cont['std_reward']}, step=current_step, commit=False)
                        # Note: For continuation, we track rewards (not lengths) as the main metric
                        # Episode lengths are still useful but rewards show cumulative performance
                        if len(latest_cont['episode_lengths']) > 0:
                            wandb.log({'eval_continuation/mean_ep_length': float(np.mean(latest_cont['episode_lengths']))}, step=current_step, commit=False)
                            wandb.log({'eval_continuation/std_ep_length': float(np.std(latest_cont['episode_lengths']))}, step=current_step, commit=False)
                        wandb.log({}, step=current_step, commit=True)
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"⚠️  WandB continuation eval logging error: {e}")
        
        # 2. Check for standard evaluation (if not using dual eval)
        elif self.eval_callback is not None and hasattr(self.eval_callback, 'evaluations_timesteps') and hasattr(self.eval_callback, 'evaluations_results'):
            eval_timesteps = getattr(self.eval_callback, 'evaluations_timesteps', [])
            eval_results = getattr(self.eval_callback, 'evaluations_results', [])
            
            # Check for any new evaluations we haven't logged yet
            # Process all new evaluations (in case multiple happened)
            # We iterate through all evaluations and use the set to prevent duplicates
            for i in range(len(eval_timesteps)):
                eval_step = eval_timesteps[i]
                
                # Skip if we've already logged this evaluation step
                if eval_step in self._logged_eval_steps:
                    continue
                
                latest_eval_rewards = eval_results[i] if i < len(eval_results) else []
                
                if len(latest_eval_rewards) > 0:
                    # Log each eval metric separately at the exact evaluation timestep (like MRQ)
                    # This ensures all runs log at exactly the same steps (5000, 10000, 15000, etc.)
                    try:
                        # Log global_step first to ensure x-axis represents actual training step
                        wandb.log({'global_step': eval_step}, step=eval_step, commit=False)
                        # Log extrinsic rewards (from environment)
                        wandb.log({'eval/mean_reward': float(np.mean(latest_eval_rewards))}, step=eval_step, commit=False)
                        wandb.log({'eval/std_reward': float(np.std(latest_eval_rewards))}, step=eval_step, commit=False)
                        wandb.log({'eval/max_reward': float(np.max(latest_eval_rewards))}, step=eval_step, commit=False)
                        wandb.log({'eval/min_reward': float(np.min(latest_eval_rewards))}, step=eval_step, commit=False)
                        
                        # Log intrinsic+extrinsic rewards if available (from EnhancedEvalCallback)
                        if isinstance(self.eval_callback, EnhancedEvalCallback):
                            if len(self.eval_callback.evaluations_total_rewards) > i:
                                total_rewards = self.eval_callback.evaluations_total_rewards[i]
                                if len(total_rewards) > 0:
                                    wandb.log({'eval_extrinsic_intrinsic/mean_reward': float(np.mean(total_rewards))}, step=eval_step, commit=False)
                                    wandb.log({'eval_extrinsic_intrinsic/std_reward': float(np.std(total_rewards))}, step=eval_step, commit=False)
                                    wandb.log({'eval_extrinsic_intrinsic/max_reward': float(np.max(total_rewards))}, step=eval_step, commit=False)
                                    wandb.log({'eval_extrinsic_intrinsic/min_reward': float(np.min(total_rewards))}, step=eval_step, commit=False)
                                
                                # Also log intrinsic rewards separately
                                if len(self.eval_callback.evaluations_intrinsic_rewards) > i:
                                    intrinsic_rewards = self.eval_callback.evaluations_intrinsic_rewards[i]
                                    if len(intrinsic_rewards) > 0:
                                        wandb.log({'eval/mean_intrinsic_reward': float(np.mean(intrinsic_rewards))}, step=eval_step, commit=False)
                        
                        # Get episode length if available
                        eval_lengths = getattr(self.eval_callback, 'evaluations_length', [])
                        if len(eval_lengths) > i and len(eval_lengths[i]) > 0:
                            wandb.log({'eval/mean_ep_length': float(np.mean(eval_lengths[i]))}, step=eval_step, commit=False)
                        
                        # Commit all eval metrics together
                        wandb.log({}, step=eval_step, commit=True)
                        
                        # Mark this evaluation step as logged (only after successful logging)
                        self._logged_eval_steps.add(eval_step)
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"⚠️  WandB eval logging error: {e}")
                        # If logging fails, don't add to set, so we'll retry on next call
        
        # Fallback: use last_mean_reward if evaluations_timesteps not available (older SB3 versions)
        elif hasattr(self.eval_callback, 'last_mean_reward'):
            if (current_step != self.last_eval_step and 
                self.eval_callback.last_mean_reward is not None):
                
                eval_results = getattr(self.eval_callback, 'evaluations_results', [])
                if len(eval_results) > 0:
                    latest_eval_rewards = eval_results[-1]
                    if len(latest_eval_rewards) > 0:
                        try:
                            # Log global_step first to ensure x-axis represents actual training step
                            wandb.log({'global_step': current_step}, step=current_step, commit=False)
                            wandb.log({'eval/mean_reward': float(np.mean(latest_eval_rewards))}, step=current_step, commit=False)
                            wandb.log({'eval/std_reward': float(np.std(latest_eval_rewards))}, step=current_step, commit=False)
                            wandb.log({'eval/max_reward': float(np.max(latest_eval_rewards))}, step=current_step, commit=False)
                            wandb.log({'eval/min_reward': float(np.min(latest_eval_rewards))}, step=current_step, commit=False)
                            wandb.log({}, step=current_step, commit=True)
                        except Exception as e:
                            if self.verbose > 0:
                                print(f"⚠️  WandB eval logging error: {e}")
                else:
                    try:
                        # Log global_step first to ensure x-axis represents actual training step
                        wandb.log({'global_step': current_step}, step=current_step, commit=False)
                        wandb.log({'eval/mean_reward': self.eval_callback.last_mean_reward}, step=current_step, commit=True)
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"⚠️  WandB eval logging error: {e}")
                
                self.last_eval_step = current_step
        
        # 3. Log episode metrics when episodes complete (like MRQ does when episodes end)
        # This ensures episode metrics are logged at consistent timesteps (when episodes actually end)
        # Track episode extrinsic and intrinsic rewards before checking monitor stats (in case wrapper gets reset)
        if self.intrinsic_wrapper is not None:
            episode_stats_temp = self.intrinsic_wrapper.get_episode_statistics()
            current_ep_extrinsic = episode_stats_temp.get('episode_extrinsic_reward', 0.0)
            current_ep_intrinsic = episode_stats_temp.get('episode_intrinsic_reward', 0.0)
            if current_ep_extrinsic > 0.01:  # Only update if episode not reset yet
                self.last_seen_episode_extrinsic = current_ep_extrinsic
                self.last_seen_episode_intrinsic = current_ep_intrinsic
        
        episode_stats = self._get_monitor_stats()
        if episode_stats:
            # Log each episode metric separately at the exact timestep when episode ended
            # This matches MRQ's approach: log_metric('episode/reward', ..., step=self.t)
            try:
                # Log global_step first to ensure x-axis represents actual training step
                wandb.log({'global_step': current_step}, step=current_step, commit=False)
                
                # Log extrinsic-only metrics (existing group)
                wandb.log({'episode/reward': episode_stats['episode/reward']}, step=current_step, commit=False)
                wandb.log({'episode/intrinsic_reward': episode_stats['episode/intrinsic_reward']}, step=current_step, commit=False)
                wandb.log({'episode/length': episode_stats['episode/length']}, step=current_step, commit=False)
                wandb.log({'episode/number': episode_stats['episode/number']}, step=current_step, commit=False)
                
                # Log extrinsic + intrinsic metrics (new group)
                wandb.log({'episode_extrinsic_intrinsic/reward': episode_stats['episode_extrinsic_intrinsic/reward']}, step=current_step, commit=False)
                wandb.log({'episode_extrinsic_intrinsic/length': episode_stats['episode_extrinsic_intrinsic/length']}, step=current_step, commit=False)
                wandb.log({'episode_extrinsic_intrinsic/number': episode_stats['episode_extrinsic_intrinsic/number']}, step=current_step, commit=False)
                
                wandb.log({'episode/current_t': current_step}, step=current_step, commit=True)
            except Exception as e:
                if self.verbose > 0:
                    print(f"⚠️  WandB episode logging error: {e}")
        
        return True


def create_env(config, uncertainty_method, gt_tracker=None):
    """
    Create and wrap environment with intrinsic rewards.
    Used for training environment where intrinsic rewards are needed.
    
    Args:
        config: RLConfig object
        uncertainty_method: Uncertainty method object
        gt_tracker: GroundTruthTracker (optional, for GT baseline)
        
    Returns:
        env: Wrapped environment with GoalWrapper and IntrinsicRewardWrapper
        gt_tracker: GroundTruthTracker (may be created if None)
    """
    # Create base environment
    # Set continuing_task=False so episodes terminate when goal is reached
    env = make_pointmaze_env(config.env_name, seed=config.a_seed, continuing_task=False)
    
    # Get maze map
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../01sweep_uncertainty/utilities'))
    from environment import get_maze_map
    try:
        maze_map = get_maze_map()
    except:
        maze_map = None
    
    # Select goals based on goal_mode
    print(f"  Goal mode: {config.goal_mode}")
    if config.goal_mode == 'single':
        # Single-goal mode: select fixed goal
        fixed_goal_cell = select_fixed_goal(env, config.a_seed, config.fixed_goal_cell, maze_map)
        print(f"  Selected fixed goal cell: {fixed_goal_cell}")
        goal_cells = None
        num_goals = None
    elif config.goal_mode == 'multi':
        # Multi-goal mode: select diverse goals
        goal_cells = select_diverse_goals(env, config.num_goals, config.a_seed, maze_map)
        print(f"  Selected {len(goal_cells)} diverse goals: {goal_cells}")
        fixed_goal_cell = None
        num_goals = len(goal_cells)
    else:
        raise ValueError(f"Invalid goal_mode: {config.goal_mode}. Must be 'single' or 'multi'")
    
    # Wrap with GoalWrapper (before IntrinsicRewardWrapper)
    env = GoalWrapper(
        env,
        goal_mode=config.goal_mode,
        fixed_goal_cell=fixed_goal_cell,
        goal_cells=goal_cells,
        grid_rows=config.grid_rows,
        grid_cols=config.grid_cols
    )
    
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
            maze_map=maze_map,
            goal_mode=config.goal_mode,
            num_goals=num_goals
        )
    else:
        # Use uncertainty method
        env = IntrinsicRewardWrapper(
            env,
            uncertainty_method=uncertainty_method,
            beta=config.beta,
            grid_rows=config.grid_rows,
            grid_cols=config.grid_cols,
            maze_map=maze_map,
            goal_mode=config.goal_mode,
            num_goals=num_goals
        )
    
    return env, gt_tracker


def create_eval_env(config, continuing_task=False):
    """
    Create evaluation environment WITHOUT intrinsic rewards.
    Evaluation metrics should only reflect extrinsic (task) performance.
    Uses the same goal selection as training (fixed goal for single, same N goals for multi).
    
    Args:
        config: RLConfig object
        continuing_task: If True, episode continues after reaching goal (new goal generated).
                        If False, episode terminates when goal is reached.
        
    Returns:
        env: Base environment with GoalWrapper (no IntrinsicRewardWrapper)
    """
    # Create base environment
    env = make_pointmaze_env(config.env_name, seed=config.a_seed, continuing_task=continuing_task)
    
    # Get maze map
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../01sweep_uncertainty/utilities'))
    from environment import get_maze_map
    try:
        maze_map = get_maze_map()
    except:
        maze_map = None
    
    # Select goals using same logic as training (same seed ensures same goals)
    if config.goal_mode == 'single':
        fixed_goal_cell = select_fixed_goal(env, config.a_seed, config.fixed_goal_cell, maze_map)
        goal_cells = None
    elif config.goal_mode == 'multi':
        goal_cells = select_diverse_goals(env, config.num_goals, config.a_seed, maze_map)
        fixed_goal_cell = None
    else:
        raise ValueError(f"Invalid goal_mode: {config.goal_mode}. Must be 'single' or 'multi'")
    
    # Wrap with GoalWrapper (no intrinsic rewards for evaluation)
    env = GoalWrapper(
        env,
        goal_mode=config.goal_mode,
        fixed_goal_cell=fixed_goal_cell,
        goal_cells=goal_cells,
        grid_rows=config.grid_rows,
        grid_cols=config.grid_cols
    )
    
    return env


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
    
    # Get maze map for evaluation callback
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../01sweep_uncertainty/utilities'))
    from environment import get_maze_map
    try:
        maze_map = get_maze_map()
    except:
        maze_map = None
    
    # Create evaluation environments (WITHOUT intrinsic rewards for pure task performance metrics)
    print("Creating evaluation environments (extrinsic rewards only)...")
    print(f"  Goal mode: {config.goal_mode}")
    eval_env = create_eval_env(config, continuing_task=False)
    eval_env = Monitor(eval_env, os.path.join(config.log_dir, 'eval'))
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # For multi-goal mode, we'll evaluate on all goals (handled in callback)
    eval_env_single = None
    eval_env_cont = None
    
    # Get environment info
    env_info = get_env_info(train_env.envs[0].env.env)
    print(f"\nEnvironment info:")
    print(f"  Observation space: {env_info['observation_space']}")
    print(f"  Action space: {env_info['action_space']}")
    
    # Detect observation space structure to determine policy type
    obs_space = train_env.observation_space
    if hasattr(obs_space, 'spaces') and isinstance(obs_space.spaces, dict):
        # Dict observation space - use MultiInputPolicy
        policy_type = 'MultiInputPolicy'
        obs_keys = list(obs_space.spaces.keys())
        print(f"  Using MultiInputPolicy with observation keys: {obs_keys}")
    else:
        # Non-dict observation space - use MlpPolicy
        policy_type = 'MlpPolicy'
        print(f"  Using MlpPolicy")
    
    # Create RL agent
    print(f"\nCreating {config.algorithm.upper()} agent...")
    
    # Enable TensorBoard logging if WandB is enabled (WandbCallback needs it)
    tensorboard_log_dir = config.tensorboard_log
    if config.wandb_switch and wandb.run is not None and WANDB_CALLBACK_AVAILABLE:
        if tensorboard_log_dir is None:
            tensorboard_log_dir = os.path.join(config.log_dir, 'tensorboard')
            os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    if config.algorithm.lower() == 'sac':
        model = SAC(
            policy_type,
            train_env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            tau=config.sac_config['tau'],
            gamma=config.sac_config['gamma'],
            train_freq=config.sac_config['train_freq'],
            gradient_steps=config.sac_config['gradient_steps'],
            learning_starts=config.sac_config['learning_starts'],
            tensorboard_log=tensorboard_log_dir,
            verbose=config.verbose,
            device=config.device,
            seed=config.a_seed,
        )
    elif config.algorithm.lower() == 'ppo':
        model = PPO(
            policy_type,
            train_env,
            learning_rate=config.learning_rate,
            n_steps=config.ppo_config['n_steps'],
            batch_size=config.ppo_config['batch_size'],
            n_epochs=config.ppo_config['n_epochs'],
            gamma=config.ppo_config['gamma'],
            gae_lambda=config.ppo_config['gae_lambda'],
            clip_range=config.ppo_config['clip_range'],
            ent_coef=config.ppo_config['ent_coef'],
            tensorboard_log=tensorboard_log_dir,
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
    # Use enhanced evaluation callback that tracks intrinsic+extrinsic rewards
    print(f"✓ Using enhanced evaluation callback (goal_mode: {config.goal_mode})")
    
    # Get uncertainty method, beta, and training wrapper from training environment
    # Training wrapper provides visit counts that reflect what the agent was trained on
    uncertainty_method_for_eval = None
    beta_for_eval = 0.0
    num_goals_for_eval = None
    training_wrapper_for_eval = None
    if hasattr(train_env, 'envs') and len(train_env.envs) > 0:
        env = train_env.envs[0]
        while hasattr(env, 'env'):
            if isinstance(env, IntrinsicRewardWrapper):
                uncertainty_method_for_eval = env.uncertainty_method
                beta_for_eval = env.beta
                num_goals_for_eval = env.num_goals if env.goal_mode == 'multi' else None
                training_wrapper_for_eval = env  # Keep reference for training visit counts
                break
            env = env.env
    
    eval_callback = EnhancedEvalCallback(
            eval_env,
        uncertainty_method=uncertainty_method_for_eval,
        beta=beta_for_eval,
        grid_rows=config.grid_rows,
        grid_cols=config.grid_cols,
        maze_map=maze_map,
        goal_mode=config.goal_mode,
        num_goals=num_goals_for_eval,
        training_wrapper=training_wrapper_for_eval,  # Pass training wrapper for training visit counts
            best_model_save_path=os.path.join(config.log_dir, 'best_model'),
            log_path=os.path.join(config.log_dir, 'eval'),
            eval_freq=config.eval_freq,
            n_eval_episodes=config.n_eval_episodes,
            deterministic=True,
            render=False,
        )
    callbacks.append(eval_callback)
    dual_eval_callback = None  # No longer used
    
    # WandB callback (if enabled and available)
    if config.wandb_switch and wandb.run is not None:
        # Always use custom callback for reliable direct logging (works in sweep mode)
        # This reads directly from SB3's logger and logs all metrics
        # Also tracks episode metrics from Monitor and enhanced evaluation metrics
        print("✓ Using custom WandB logging callback (reads directly from SB3 logger + Monitor episode stats)")
        wandb_logging_callback = WandBLoggingCallback(
            eval_callback, 
            train_env, 
            dual_eval_callback=None,  # No longer used
            verbose=1
        )
        callbacks.append(wandb_logging_callback)
        
        # Don't use official WandbCallback - it conflicts with explicit step logging
        # The official callback uses TensorBoard syncing which doesn't allow explicit step values
        # Our custom callback handles all logging with proper step alignment
        # if WANDB_CALLBACK_AVAILABLE:
        #     try:
        #         wandb_callback = WandbCallback(...)
        #         callbacks.append(wandb_callback)
        #     except Exception as e:
        #         pass
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.eval_freq,
        save_path=os.path.join(config.log_dir, 'checkpoints'),
        name_prefix='rl_model',
    )
    callbacks.append(checkpoint_callback)
    
    # Note: Uncertainty model updates happen in the wrapper's step method
    # The wrapper calls update_with_batch every step, and the adapter handles
    # the update frequency internally
    
    # Set model reference for dual eval callback (if using dual eval)
    if dual_eval_callback is not None:
        dual_eval_callback.model = model
    
    # Set model reference for dual eval callback (if using dual eval)
    if dual_eval_callback is not None:
        dual_eval_callback.model = model
    
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
    parser.add_argument('--use_gt_baseline', type=lambda x: str(x).lower() in ('true', '1', 'yes'),
                       default=False, nargs='?', const=True,
                       help='Use GT as intrinsic reward baseline (accepts True/False or flag)')
    
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
    parser.add_argument('--dual_eval', type=lambda x: str(x).lower() in ('true', '1', 'yes'),
                       default=True, nargs='?', const=True,
                       help='Enable dual evaluation (single-goal + continuation). Default: True')
    
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
        
        # Don't enable TensorBoard syncing - we use custom callback with explicit step logging
        # TensorBoard syncing conflicts with explicit step values in wandb.log()
        # Our custom callback handles all logging directly with proper step alignment
        # try:
        #     from wandb import tensorboard as wandb_tensorboard
        #     wandb_tensorboard.patch(save=False)
        # except Exception as e:
        #     pass
        
        # Override arguments with WandB config
        args.algorithm = sweep_config.get('algorithm', args.algorithm)
        args.beta = sweep_config.get('beta', args.beta)
        args.a_seed = sweep_config.get('a_seed', args.a_seed)
        args.total_timesteps = sweep_config.get('total_timesteps', args.total_timesteps)
        args.eval_freq = sweep_config.get('eval_freq', args.eval_freq)
        args.n_eval_episodes = sweep_config.get('n_eval_episodes', args.n_eval_episodes)
        args.dual_eval = sweep_config.get('dual_eval', args.dual_eval)
        args.log_dir = sweep_config.get('log_dir', args.log_dir)
        args.device = sweep_config.get('device', args.device)
        args.wandb_switch = sweep_config.get('wandb_switch', args.wandb_switch)
        # GT baseline flag
        args.use_gt_baseline = sweep_config.get('use_gt_baseline', args.use_gt_baseline)
        # Note: uncertainty_method is set but not used when beta=0 or use_gt_baseline=true
        args.uncertainty_method = sweep_config.get('uncertainty_method', args.uncertainty_method)
        # Environment parameters
        args.env_name = sweep_config.get('env_name', args.env_name)
        args.grid_rows = sweep_config.get('grid_rows', args.grid_rows)
        args.grid_cols = sweep_config.get('grid_cols', args.grid_cols)
    
    # Initialize WandB if not in sweep and wandb_switch is enabled
    wandb_switch = args.wandb_switch.lower() == 'true'
    if wandb_switch and wandb.run is None:
        # Determine TensorBoard log directory (will be set in train function)
        tensorboard_log_dir = os.path.join(args.log_dir, 'tensorboard')
        # Don't sync TensorBoard - we use custom callback for direct logging with explicit steps
        
        # Determine run name based on whether GT baseline is used
        if args.use_gt_baseline:
            run_name = f"{args.algorithm}_gt_beta{args.beta}_seed{args.a_seed}"
            tags = ["gt_baseline", args.algorithm]
        else:
            run_name = f"{args.algorithm}_plain_rl_seed{args.a_seed}"
            tags = ["plain_rl", args.algorithm]
        
        wandb.init(
            project="rl_integration",
            name=run_name,
            config=vars(args),
            tags=tags,
            sync_tensorboard=False,  # Disable TensorBoard syncing - we use custom callback with explicit steps
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
    config.dual_eval = args.dual_eval
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

