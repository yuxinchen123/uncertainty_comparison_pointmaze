"""
Ground Truth uncertainty as an intrinsic reward method.
GT uncertainty = 1/√N(i) where N(i) is visit count per grid cell.
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


class GTIntrinsicReward:
    """
    Ground Truth uncertainty as an intrinsic reward method.
    
    This class provides the same interface as other uncertainty methods,
    but uses the ground truth formula: 1/√N(i) where N(i) is visit count.
    """
    
    def __init__(self, grid_rows=9, grid_cols=12, maze_map=None):
        """
        Args:
            grid_rows: Number of grid rows (default 9)
            grid_cols: Number of grid columns (default 12)
            maze_map: Maze map array (walls vs open cells)
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        # Ensure maze_map is a numpy array (convert from list if needed)
        if maze_map is not None and not isinstance(maze_map, np.ndarray):
            self.maze_map = np.array(maze_map)
        else:
            self.maze_map = maze_map
        
        # Visit count matrix (0-based indexing)
        self.visit_counts = np.zeros((grid_rows, grid_cols), dtype=int)
        
    def update_visit_counts(self, visit_counts):
        """
        Update visit counts from external source (e.g., from wrapper).
        
        Args:
            visit_counts: 2D array of visit counts (grid_rows x grid_cols)
        """
        self.visit_counts = visit_counts.copy()
    
    def _state_to_grid(self, observation):
        """Map observation to grid cell (0-based)"""
        _, (row, col) = observation_to_grid_notebook_exact(
            observation,
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols
        )
        return row - 1, col - 1
    
    def get_uncertainty(self, coordinates):
        """
        Get GT uncertainty for given coordinates.
        
        Args:
            coordinates: Array of shape (N, 2) with (x, y) positions
            
        Returns:
            uncertainties: Array of shape (N,) with uncertainty values
        """
        if len(coordinates.shape) == 1:
            coordinates = coordinates.reshape(1, -1)
        
        uncertainties = []
        for coord in coordinates:
            # Create observation dict format
            obs = {'achieved_goal': np.array([coord[0], coord[1]])}
            
            # Get grid cell
            row, col = self._state_to_grid(obs)
            
            # Check bounds and if cell is open
            if 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
                if self.maze_map is None or self.maze_map[row, col] == 0:  # Open cell
                    count = self.visit_counts[row, col]
                    if count > 0:
                        uncertainty = 1.0 / np.sqrt(count)
                    else:
                        # Unvisited cell: use large but finite value to avoid infinite rewards
                        # This gives maximum exploration bonus without causing numerical issues
                        uncertainty = 100.0  # Large finite value for unvisited cells
                else:
                    uncertainty = np.nan  # Wall cell
            else:
                uncertainty = np.nan  # Out of bounds
            
            # Append uncertainty to list
            uncertainties.append(uncertainty)
        
        return np.array(uncertainties)

