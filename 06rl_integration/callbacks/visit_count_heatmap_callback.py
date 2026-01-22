"""
Callback to periodically log visit count heatmaps to WandB.
"""
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional
import sys
import os

# Add utils to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))
from utils.heatmap_utils import create_visit_count_heatmap, create_multi_goal_heatmap
from wrappers.intrinsic_reward_wrapper import IntrinsicRewardWrapper


class VisitCountHeatmapCallback(BaseCallback):
    """
    Callback that periodically logs visit count heatmaps to WandB.
    
    Works for both training and evaluation by tracking visit counts
    from the IntrinsicRewardWrapper.
    """
    
    def __init__(
        self,
        train_env,
        eval_env=None,
        log_freq: int = 10000,
        maze_map: Optional[np.ndarray] = None,
        goal_cells: Optional[list] = None,
        start_cell: Optional[tuple] = None,
        beta: float = 1.0,
        verbose: int = 0,
    ):
        """
        Args:
            train_env: Training environment (DummyVecEnv) to access IntrinsicRewardWrapper
            eval_env: Optional evaluation environment (for eval visit counts)
            log_freq: Log heatmap every N steps (default: 10000)
            maze_map: Maze map array (1 = wall, 0 = open)
            goal_cells: List of (row, col) tuples for goal locations
            start_cell: Optional (row, col) tuple for start location
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env
        self.log_freq = log_freq
        self.maze_map = maze_map
        self.goal_cells = goal_cells
        self.start_cell = start_cell
        self.beta = beta  # Store beta for heatmap titles
        self.last_log_step = -1
        
        # Find IntrinsicRewardWrapper in training environment
        self.train_wrapper = None
        if hasattr(train_env, 'envs') and len(train_env.envs) > 0:
            env = train_env.envs[0]
            while hasattr(env, 'env'):
                if isinstance(env, IntrinsicRewardWrapper):
                    self.train_wrapper = env
                    break
                env = env.env
        
        # Find IntrinsicRewardWrapper in evaluation environment (if provided)
        self.eval_wrapper = None
        if eval_env is not None and hasattr(eval_env, 'envs') and len(eval_env.envs) > 0:
            env = eval_env.envs[0]
            while hasattr(env, 'env'):
                if isinstance(env, IntrinsicRewardWrapper):
                    self.eval_wrapper = env
                    break
                env = env.env
    
    def _log_heatmap(self, visit_counts: np.ndarray, prefix: str, step: int, goal_idx: Optional[int] = None):
        """
        Create and log a heatmap to WandB.
        
        Args:
            visit_counts: Visit count array (2D or 3D)
            prefix: Prefix for WandB log key (e.g., "train" or "eval")
            step: Current training step
            goal_idx: Optional goal index (for multi-goal mode)
        """
        if wandb.run is None:
            return
        
        try:
            # Handle multi-goal mode (3D array)
            if len(visit_counts.shape) == 3:
                # Create heatmaps for each goal
                # Ensure goal_cells is a list (not None) with correct length
                if self.goal_cells is not None and isinstance(self.goal_cells, list):
                    goal_cells = self.goal_cells
                else:
                    goal_cells = [None] * visit_counts.shape[0]
                
                figures = create_multi_goal_heatmap(
                    visit_counts,
                    maze_map=self.maze_map,
                    goal_cells=goal_cells,
                    title_prefix=f"{prefix.capitalize()} Visit Count (β={self.beta})",
                )
                
                # Log each goal's heatmap under media/ namespace
                for goal_idx, fig in enumerate(figures):
                    try:
                        wandb.log({
                            f"media/{prefix}/visit_count_heatmap/goal_{goal_idx + 1}": wandb.Image(fig)
                        }, step=step, commit=True)  # Explicitly commit images
                        if self.verbose > 0:
                            print(f"    ✓ Logged media/{prefix}/visit_count_heatmap/goal_{goal_idx + 1} to WandB at step {step}")
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"    ⚠️  Error logging media/{prefix}/visit_count_heatmap/goal_{goal_idx + 1}: {e}")
                    import matplotlib.pyplot as plt
                    plt.close(fig)
            
            # Handle single-goal mode (2D array)
            elif len(visit_counts.shape) == 2:
                goal_cell = None
                if self.goal_cells is not None and isinstance(self.goal_cells, list) and len(self.goal_cells) > 0:
                    # For single-goal, goal_cells should be a list with one element
                    goal_cell = self.goal_cells[0] if goal_idx is None else (
                        self.goal_cells[goal_idx] if goal_idx < len(self.goal_cells) else None
                    )
                
                fig = create_visit_count_heatmap(
                    visit_counts,
                    maze_map=self.maze_map,
                    title=f"{prefix.capitalize()} Visit Count Heatmap (β={self.beta}, Step {step})",
                    goal_cell=goal_cell,
                    start_cell=self.start_cell,
                )
                
                # Log heatmap image to WandB under media/ namespace
                try:
                    wandb.log({
                        f"media/{prefix}/visit_count_heatmap": wandb.Image(fig)
                    }, step=step, commit=True)  # Explicitly commit images
                    if self.verbose > 0:
                        print(f"    ✓ Logged media/{prefix}/visit_count_heatmap to WandB at step {step}")
                except Exception as e:
                    if self.verbose > 0:
                        print(f"    ⚠️  Error logging media/{prefix}/visit_count_heatmap: {e}")
                
                import matplotlib.pyplot as plt
                plt.close(fig)
        
        except Exception as e:
            if self.verbose > 0:
                import traceback
                print(f"⚠️  Error logging heatmap: {e}")
                traceback.print_exc()
    
    def _on_step(self) -> bool:
        """
        Called at each step during training.
        Logs heatmaps periodically.
        """
        if wandb.run is None:
            return True
        
        current_step = self.num_timesteps
        
        # Check if it's time to log
        if current_step - self.last_log_step >= self.log_freq:
            # Log training visit counts
            if self.train_wrapper is not None:
                train_visit_counts = self.train_wrapper.get_visit_counts()
                if train_visit_counts is not None:
                    # Check if there are any non-zero visit counts
                    if np.any(train_visit_counts > 0):
                        if self.verbose > 0:
                            print(f"  Logging training heatmap at step {current_step} (max visits: {np.max(train_visit_counts)})")
                        self._log_heatmap(train_visit_counts, "train", current_step)
                    elif self.verbose > 0:
                        print(f"  Skipping training heatmap at step {current_step} (all visit counts are zero)")
            
            # Log evaluation visit counts (if available)
            # Note: Eval visit counts accumulate across all evaluation episodes
            # They show where the learned policy visits during evaluation
            if self.eval_wrapper is not None:
                eval_visit_counts = self.eval_wrapper.get_visit_counts()
                if eval_visit_counts is not None:
                    # Check if there are any non-zero visit counts
                    if np.any(eval_visit_counts > 0):
                        if self.verbose > 0:
                            print(f"  Logging evaluation heatmap at step {current_step} (max visits: {np.max(eval_visit_counts)})")
                        self._log_heatmap(eval_visit_counts, "eval", current_step)
                    elif self.verbose > 0:
                        print(f"  Skipping evaluation heatmap at step {current_step} (all visit counts are zero)")
            
            self.last_log_step = current_step
        
        return True
