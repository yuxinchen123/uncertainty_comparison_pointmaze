"""
Ground Truth tracker for maintaining visit counts and computing GT uncertainty.
"""
import numpy as np
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
get_maze_map = utilities_evaluation.get_maze_map


class GroundTruthTracker:
    """
    Tracks visit counts and computes ground truth uncertainty during RL training.
    
    GT uncertainty = 1/√N(i) where N(i) is visit count per grid cell.
    """
    
    def __init__(self, grid_rows=9, grid_cols=12, maze_map=None):
        """
        Args:
            grid_rows: Number of grid rows (default 9)
            grid_cols: Number of grid columns (default 12)
            maze_map: Maze map array (if None, will be extracted)
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        if maze_map is None:
            self.maze_map = get_maze_map()
        else:
            self.maze_map = maze_map
        
        # Visit count matrix (0-based indexing)
        self.visit_counts = np.zeros((grid_rows, grid_cols), dtype=int)
    
    def update_from_visit_counts(self, visit_counts):
        """
        Update visit counts from external source (e.g., from wrapper).
        
        Args:
            visit_counts: 2D array of visit counts (grid_rows x grid_cols)
        """
        self.visit_counts = visit_counts.copy()
    
    def get_visit_counts(self):
        """Get current visit count matrix"""
        return self.visit_counts.copy()
    
    def compute_gt_uncertainty_matrix(self):
        """
        Compute GT uncertainty matrix: 1/√N(i) for each grid cell.
        
        Returns:
            uncertainty_matrix: 2D array (grid_rows x grid_cols)
                - Open cells with visits: 1/√N(i)
                - Open cells without visits: inf
                - Wall cells: nan
        """
        uncertainty_matrix = np.full((self.grid_rows, self.grid_cols), np.nan)
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if self.maze_map[row, col] == 0:  # Open cell
                    count = self.visit_counts[row, col]
                    if count > 0:
                        uncertainty_matrix[row, col] = 1.0 / np.sqrt(count)
                    else:
                        uncertainty_matrix[row, col] = np.inf
                # Wall cells remain NaN
        
        return uncertainty_matrix
    
    def get_statistics(self):
        """Get statistics about visit counts"""
        open_mask = self.maze_map == 0
        open_counts = self.visit_counts[open_mask]
        
        return {
            'total_visits': int(np.sum(self.visit_counts)),
            'visited_cells': int(np.sum(open_counts > 0)),
            'unvisited_cells': int(np.sum(open_counts == 0)),
            'max_visits': int(np.max(open_counts)) if len(open_counts) > 0 else 0,
            'min_visits': int(np.min(open_counts)) if len(open_counts) > 0 else 0,
            'mean_visits': float(np.mean(open_counts)) if len(open_counts) > 0 else 0.0,
        }

