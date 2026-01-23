"""
Environment wrapper that adds intrinsic rewards based on uncertainty quantification methods.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os

# Add parent directories to path to import utilities
utilities_path = os.path.join(os.path.dirname(__file__), '../../01sweep_uncertainty/utilities')
sys.path.insert(0, utilities_path)

# Import from utilities/evaluation.py (not the local evaluation package)
# Use importlib to avoid naming conflicts with local evaluation package
import importlib.util
spec = importlib.util.spec_from_file_location("utilities_evaluation", 
                                               os.path.join(utilities_path, "evaluation.py"))
utilities_evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utilities_evaluation)

observation_to_grid_notebook_exact = utilities_evaluation.observation_to_grid_notebook_exact


class IntrinsicRewardWrapper(gym.Wrapper):
    """
    Wraps PointMaze environment to add intrinsic rewards based on uncertainty methods.
    
    The wrapper:
    - Maps continuous states to grid cells (9×12)
    - Computes intrinsic reward from uncertainty method
    - Combines extrinsic + intrinsic rewards
    - Maintains visit count table for GT calculation
    - Tracks states for uncertainty model training
    """
    
    def __init__(self, env, uncertainty_method, beta=1.0, grid_rows=9, grid_cols=12, maze_map=None,
                 goal_mode='single', num_goals=None):
        """
        Args:
            env: PointMaze environment (should be wrapped with GoalWrapper)
            uncertainty_method: Object with get_uncertainty(state) method
            beta: Intrinsic reward coefficient (r_total = r_extrinsic + beta * r_intrinsic)
            grid_rows: Number of grid rows (default 9)
            grid_cols: Number of grid columns (default 12)
            maze_map: Maze map array (if None, will be extracted from env)
            goal_mode: 'single' or 'multi' - determines visit count structure
            num_goals: Number of goals for multi-goal mode (required if goal_mode='multi')
        """
        super().__init__(env)
        self.uncertainty_method = uncertainty_method
        self.beta = beta
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.goal_mode = goal_mode
        self.num_goals = num_goals
        
        # Get maze map if not provided
        if maze_map is None:
            self.maze_map = self._get_maze_map()
        else:
            self.maze_map = maze_map
        
        # Ensure maze_map is a numpy array (convert from list if needed)
        if not isinstance(self.maze_map, np.ndarray):
            self.maze_map = np.array(self.maze_map)
        
        # Auto-detect grid dimensions from maze map (more reliable than config)
        actual_rows, actual_cols = self.maze_map.shape
        if actual_rows != grid_rows or actual_cols != grid_cols:
            if grid_rows != 9 or grid_cols != 12:  # Only warn if not default
                print(f"⚠️  Grid dimensions mismatch: config says {grid_rows}×{grid_cols}, "
                      f"but maze map is {actual_rows}×{actual_cols}. Using maze map dimensions.")
            self.grid_rows = actual_rows
            self.grid_cols = actual_cols
        else:
            self.grid_rows = grid_rows
            self.grid_cols = grid_cols
        
        # Visit count matrix for GT calculation (0-based indexing)
        # For single-goal: (grid_rows, grid_cols)
        # For multi-goal: (num_goals, grid_rows, grid_cols)
        if goal_mode == 'multi':
            if num_goals is None:
                raise ValueError("num_goals must be provided for multi-goal mode")
            self.visit_counts = np.zeros((num_goals, self.grid_rows, self.grid_cols), dtype=int)
        else:
            self.visit_counts = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        
        # Track states for uncertainty model training (online updates)
        self.visited_states = []
        
        # Statistics tracking (cumulative across all episodes)
        self.total_intrinsic_reward = 0.0
        self.total_extrinsic_reward = 0.0
        self.step_count = 0
        
        # Per-episode reward tracking (reset on each episode)
        self.episode_extrinsic_reward = 0.0
        self.episode_intrinsic_reward = 0.0
        
    def _get_maze_map(self):
        """Extract maze map from environment"""
        unwrapped_env = self.env.unwrapped
        if hasattr(unwrapped_env, 'maze') and hasattr(unwrapped_env.maze, 'maze_map'):
            maze_map = unwrapped_env.maze.maze_map
            # Convert to numpy array if it's a list
            if not isinstance(maze_map, np.ndarray):
                maze_map = np.array(maze_map)
            return maze_map
        else:
            # Fallback: create default map (all open)
            return np.zeros((self.grid_rows, self.grid_cols), dtype=int)
    
    def _get_current_goal_idx(self):
        """Get current goal index from GoalWrapper (for multi-goal mode)"""
        # GoalWrapper should be wrapping the environment
        if hasattr(self.env, 'get_current_goal_idx'):
            return self.env.get_current_goal_idx()
        return None
    
    def _state_to_grid(self, observation):
        """
        Map continuous state observation to grid cell.
        
        Args:
            observation: Gymnasium observation (dict with 'achieved_goal' or array)
            
        Returns:
            (row, col): 0-based grid indices
        """
        _, (row, col) = observation_to_grid_notebook_exact(
            observation, 
            grid_rows=self.grid_rows, 
            grid_cols=self.grid_cols
        )
        # Convert to 0-based indexing
        return row - 1, col - 1
    
    def _get_uncertainty(self, observation):
        """
        Get uncertainty for a given observation.
        
        Args:
            observation: Gymnasium observation (can be dict or VecEnv batched dict)
            
        Returns:
            uncertainty: Scalar uncertainty value
        """
        # Extract position from observation
        # Handle both regular and VecEnv batched observations
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
            raise ValueError(f"Invalid observation format: {type(observation)}")
        
        # Get uncertainty from method (expects 2D array of positions)
        position_array = np.array([position]).reshape(1, -1)
        uncertainty = self.uncertainty_method.get_uncertainty(position_array)
        
        # Return scalar, handling NaN and inf values
        if isinstance(uncertainty, np.ndarray):
            if len(uncertainty) == 0:
                return 0.0
            uncertainty_val = float(uncertainty[0])
        else:
            uncertainty_val = float(uncertainty)
        
        # Handle NaN and inf values (e.g., from walls or out-of-bounds)
        if np.isnan(uncertainty_val) or np.isinf(uncertainty_val):
            return 0.0  # No intrinsic reward for invalid cells
        
        return uncertainty_val
    
    def step(self, action):
        """
        Step environment and add intrinsic reward.
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        obs, reward_extrinsic, terminated, truncated, info = self.env.step(action)
        
        # Update visit counts
        row, col = self._state_to_grid(obs)
        if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            if self.maze_map[row, col] == 0:  # Only count open cells
                if self.goal_mode == 'multi':
                    # Get current goal index from GoalWrapper
                    goal_idx = self._get_current_goal_idx()
                    if goal_idx is not None:
                        self.visit_counts[goal_idx, row, col] += 1
                    # Debug: warn if goal_idx is None (shouldn't happen in multi-goal mode)
                    # elif self.step_count % 1000 == 0:  # Only print occasionally
                    #     print(f"WARNING: goal_idx is None in multi-goal mode at step {self.step_count}")
                else:
                    # Single-goal mode: single visit count matrix
                    self.visit_counts[row, col] += 1
        
        # Track state for uncertainty model training (before getting uncertainty)
        if isinstance(obs, dict) and 'achieved_goal' in obs:
            state = obs['achieved_goal'][:2].copy()
            self.visited_states.append(state)
        elif isinstance(obs, np.ndarray):
            state = obs[:2].copy() if len(obs) >= 2 else obs.copy()
            self.visited_states.append(state)
        else:
            state = None
        
        # CRITICAL: Update GT method's visit counts BEFORE getting uncertainty
        # This ensures GT method uses current visit counts, not stale ones
        # For multi-goal mode, pass the visit counts for the current goal
        visit_counts_to_update = self.visit_counts
        if self.goal_mode == 'multi':
            goal_idx = self._get_current_goal_idx()
            if goal_idx is not None:
                visit_counts_to_update = self.visit_counts[goal_idx]
        
        if hasattr(self.uncertainty_method, 'update_visit_counts'):
            self.uncertainty_method.update_visit_counts(visit_counts_to_update)
        elif hasattr(self.uncertainty_method, 'method') and hasattr(self.uncertainty_method.method, 'update_visit_counts'):
            # If wrapped in adapter, update the underlying method
            self.uncertainty_method.method.update_visit_counts(visit_counts_to_update)
        
        # Get intrinsic reward (skip if beta=0 since it will be multiplied by 0 anyway)
        if self.beta == 0.0:
            intrinsic_reward = 0.0
        else:
            intrinsic_reward = self._get_uncertainty(obs)
            
            # Update uncertainty method if it supports online updates
            # This happens every step (update_frequency is handled by the adapter)
            if state is not None:
                # Check if it's an adapter (has .method attribute) or direct method
                if hasattr(self.uncertainty_method, 'update_with_batch'):
                    self.uncertainty_method.update_with_batch(np.array([state]))
                elif hasattr(self.uncertainty_method, 'method') and hasattr(self.uncertainty_method.method, 'train_on_positions'):
                    # Direct method that supports training
                    # For now, we'll let the adapter handle this, but if called directly,
                    # we could train here. However, adapters are preferred.
                    pass
        
        # Combine rewards: r_total = r_extrinsic + beta * r_intrinsic
        reward_total = reward_extrinsic + self.beta * intrinsic_reward
        
        # Update statistics (cumulative and per-episode)
        self.total_extrinsic_reward += reward_extrinsic
        self.total_intrinsic_reward += intrinsic_reward
        self.episode_extrinsic_reward += reward_extrinsic
        self.episode_intrinsic_reward += intrinsic_reward
        self.step_count += 1
        
        # Add info about rewards
        info['intrinsic_reward'] = intrinsic_reward
        info['extrinsic_reward'] = reward_extrinsic
        info['total_reward'] = reward_total
        
        return obs, reward_total, terminated, truncated, info
    
    def reset(self, seed=None, options=None, **kwargs):
        """Reset environment and clear episode statistics"""
        # Merge options with kwargs for backward compatibility
        if options is not None:
            kwargs['options'] = options
        if seed is not None:
            kwargs['seed'] = seed
            
        obs, info = self.env.reset(**kwargs)
        
        # Clear episode-specific tracking (but keep visit counts for GT)
        self.visited_states = []
        self.episode_extrinsic_reward = 0.0
        self.episode_intrinsic_reward = 0.0
        
        return obs, info
    
    def get_visited_states(self):
        """Get list of states visited in current episode"""
        return np.array(self.visited_states) if len(self.visited_states) > 0 else np.array([]).reshape(0, 2)
    
    def get_visit_counts(self):
        """
        Get current visit count matrix.
        
        Returns:
            For single-goal: (grid_rows, grid_cols) array
            For multi-goal: (num_goals, grid_rows, grid_cols) array
        """
        return self.visit_counts.copy()
    
    def get_visit_counts_for_goal(self, goal_idx):
        """
        Get visit counts for a specific goal (multi-goal mode only).
        
        Args:
            goal_idx: Goal index (0-based)
            
        Returns:
            (grid_rows, grid_cols) visit count array for the specified goal
        """
        if self.goal_mode != 'multi':
            raise ValueError("get_visit_counts_for_goal() only available in multi-goal mode")
        if goal_idx < 0 or goal_idx >= self.num_goals:
            raise ValueError(f"Invalid goal_idx: {goal_idx}. Must be in [0, {self.num_goals})")
        return self.visit_counts[goal_idx].copy()
    
    def get_statistics(self):
        """Get reward statistics (cumulative across all episodes)"""
        return {
            'total_extrinsic_reward': self.total_extrinsic_reward,
            'total_intrinsic_reward': self.total_intrinsic_reward,
            'step_count': self.step_count,
            'avg_intrinsic_reward': self.total_intrinsic_reward / max(self.step_count, 1),
            'avg_extrinsic_reward': self.total_extrinsic_reward / max(self.step_count, 1),
        }
    
    def get_episode_statistics(self):
        """Get reward statistics for current episode only"""
        return {
            'episode_extrinsic_reward': self.episode_extrinsic_reward,
            'episode_intrinsic_reward': self.episode_intrinsic_reward,
        }

