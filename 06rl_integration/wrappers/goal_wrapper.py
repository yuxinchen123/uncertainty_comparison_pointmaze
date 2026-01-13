"""
Goal management wrapper for PointMaze environments.
Handles both single-goal (no goal conditioning) and multi-goal (goal conditioning) modes.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Optional
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.goal_utils import cell_to_continuous_coords


class GoalWrapper(gym.Wrapper):
    """
    Wrapper that manages goal locations for PointMaze environments.
    
    Single-goal mode:
    - Stores a fixed goal location
    - Removes 'desired_goal' from observation (no goal conditioning)
    - Sets the same goal on every reset
    
    Multi-goal mode:
    - Stores a list of N goal locations
    - Samples a goal uniformly at episode start
    - Keeps 'desired_goal' in observation (goal conditioning)
    - Sets sampled goal on reset
    """
    
    def __init__(self, env, goal_mode: str = 'single', 
                 fixed_goal_cell: Optional[Tuple[int, int]] = None,
                 goal_cells: Optional[List[Tuple[int, int]]] = None,
                 grid_rows: int = 9, grid_cols: int = 12):
        """
        Args:
            env: PointMaze environment
            goal_mode: 'single' or 'multi'
            fixed_goal_cell: For single-goal mode, (row, col) tuple (0-based)
            goal_cells: For multi-goal mode, list of (row, col) tuples (0-based)
            grid_rows: Number of grid rows
            grid_cols: Number of grid columns
        """
        super().__init__(env)
        self.goal_mode = goal_mode
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        if goal_mode == 'single':
            if fixed_goal_cell is None:
                raise ValueError("fixed_goal_cell must be provided for single-goal mode")
            self.fixed_goal_cell = fixed_goal_cell
            self.current_goal_cell = fixed_goal_cell
            self.goal_cells = None
        elif goal_mode == 'multi':
            if goal_cells is None or len(goal_cells) == 0:
                raise ValueError("goal_cells must be provided for multi-goal mode")
            self.goal_cells = goal_cells
            self.fixed_goal_cell = None
            self.current_goal_cell = None
            self.current_goal_idx = None
        else:
            raise ValueError(f"Invalid goal_mode: {goal_mode}. Must be 'single' or 'multi'")
        
        # Store original observation space
        self.original_obs_space = env.observation_space
        
        # Modify observation space for single-goal mode (remove desired_goal)
        if goal_mode == 'single':
            if isinstance(self.original_obs_space, spaces.Dict):
                # Create new dict without 'desired_goal'
                new_spaces = {}
                for key, space in self.original_obs_space.spaces.items():
                    if key != 'desired_goal':
                        new_spaces[key] = space
                self.observation_space = spaces.Dict(new_spaces)
            else:
                # If not a dict, keep as is
                self.observation_space = self.original_obs_space
        else:
            # Multi-goal mode: keep original observation space
            self.observation_space = self.original_obs_space
    
    def _cell_to_goal_coords(self, cell: Tuple[int, int]) -> np.ndarray:
        """Convert cell (row, col) to continuous goal coordinates"""
        return cell_to_continuous_coords(cell, self.grid_rows, self.grid_cols)
    
    def reset(self, seed=None, options=None):
        """
        Reset environment and set goal.
        
        For single-goal mode: always uses the fixed goal.
        For multi-goal mode: samples a goal uniformly from the list.
        """
        # Prepare reset options
        reset_options = options.copy() if options is not None else {}
        
        if self.goal_mode == 'single':
            # Single-goal: always use fixed goal
            goal_coords = self._cell_to_goal_coords(self.fixed_goal_cell)
            # Convert to 1-based indexing for goal_cell (PointMaze expects 1-based)
            reset_options['goal_cell'] = np.array([self.fixed_goal_cell[0] + 1, self.fixed_goal_cell[1] + 1], dtype=int)
            self.current_goal_cell = self.fixed_goal_cell
        else:
            # Multi-goal: sample a goal uniformly (use environment's RNG for randomness)
            # The set of goals is deterministic (selected at training start), but which goal
            # is sampled each episode should be random for training diversity
            self.current_goal_idx = np.random.randint(len(self.goal_cells))
            self.current_goal_cell = self.goal_cells[self.current_goal_idx]
            goal_coords = self._cell_to_goal_coords(self.current_goal_cell)
            # Convert to 1-based indexing for goal_cell (PointMaze expects 1-based)
            reset_options['goal_cell'] = np.array([self.current_goal_cell[0] + 1, self.current_goal_cell[1] + 1], dtype=int)
        
        # Reset environment with goal
        obs, info = self.env.reset(seed=seed, options=reset_options)
        
        # For single-goal mode, remove 'desired_goal' from observation
        if self.goal_mode == 'single':
            if isinstance(obs, dict) and 'desired_goal' in obs:
                obs = {k: v for k, v in obs.items() if k != 'desired_goal'}
        
        # Store current goal info in info dict
        info['goal_cell'] = self.current_goal_cell
        if self.goal_mode == 'multi':
            info['goal_idx'] = self.current_goal_idx
        
        return obs, info
    
    def step(self, action):
        """Step environment and filter observation if needed"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # For single-goal mode, remove 'desired_goal' from observation
        if self.goal_mode == 'single':
            if isinstance(obs, dict) and 'desired_goal' in obs:
                obs = {k: v for k, v in obs.items() if k != 'desired_goal'}
        
        return obs, reward, terminated, truncated, info
    
    def get_current_goal_cell(self) -> Optional[Tuple[int, int]]:
        """Get current goal cell (row, col) in 0-based indexing"""
        return self.current_goal_cell
    
    def get_current_goal_idx(self) -> Optional[int]:
        """Get current goal index (for multi-goal mode only)"""
        return self.current_goal_idx if self.goal_mode == 'multi' else None
    
    def get_all_goals(self) -> List[Tuple[int, int]]:
        """Get all goal cells"""
        if self.goal_mode == 'single':
            return [self.fixed_goal_cell]
        else:
            return self.goal_cells.copy()

