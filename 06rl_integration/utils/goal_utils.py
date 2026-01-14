"""
Goal selection utilities for PointMaze environments.
"""
import numpy as np
from typing import List, Tuple, Optional
import sys
import os

# Add parent directories to path to import utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../01sweep_uncertainty/utilities'))
from environment import get_maze_map


def get_valid_cells(env, maze_map=None):
    """
    Get all valid (non-wall) cell indices from the maze.
    Uses the environment's actual maze_map to ensure consistency.
    
    Args:
        env: PointMaze environment (or wrapped environment)
        maze_map: Optional maze map array. If None, extracted from env.
        
    Returns:
        valid_cells: List of (row, col) tuples for valid cells (0-based indexing)
    """
    if maze_map is None:
        # Extract maze map directly from environment instance
        # Unwrap to get the base environment
        unwrapped_env = env
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env
        unwrapped_env = unwrapped_env.unwrapped
        
        if hasattr(unwrapped_env, 'maze') and hasattr(unwrapped_env.maze, 'maze_map'):
            maze_map = unwrapped_env.maze.maze_map
        else:
            # Fallback: try to get from utilities
            maze_map = get_maze_map()
    
    # According to documentation, maze_map is list[list] where maze_map[i][j] is row i, col j
    # Keep it as list of lists to match environment's internal format
    # Get valid cells (where value is 0, meaning open cell)
    # Documentation: 0 = open cell, 1 = wall
    valid_cells = []
    if isinstance(maze_map, list):
        # Handle list of lists (environment's native format)
        for row in range(len(maze_map)):
            for col in range(len(maze_map[row])):
                # Check if cell is open (value is 0 or can be goal)
                cell_value = maze_map[row][col]
                if cell_value == 0 or cell_value == 'g' or cell_value == 'c':
                    valid_cells.append((row, col))
    else:
        # Handle numpy array (fallback)
        maze_map_array = np.array(maze_map) if not isinstance(maze_map, np.ndarray) else maze_map
        rows, cols = maze_map_array.shape
        for row in range(rows):
            for col in range(cols):
                cell_value = maze_map_array[row, col]
                if cell_value == 0 or cell_value == 'g' or cell_value == 'c':
                    valid_cells.append((row, col))
    
    return valid_cells


def cell_to_continuous_coords(cell: Tuple[int, int], grid_rows: int, grid_cols: int) -> np.ndarray:
    """
    Convert cell indices (i, j) to continuous coordinates (x, y) in MuJoCo space.
    Uses the same coordinate system as the notebook.
    
    Args:
        cell: (row, col) tuple with 0-based indexing
        grid_rows: Number of grid rows
        grid_cols: Number of grid columns
        
    Returns:
        coords: (x, y) continuous coordinates
    """
    row, col = cell
    
    # Environment boundaries (same as notebook)
    left_edge = -6.0
    top_edge = 4.5
    cell_width = 12.0 / grid_cols    # 1.0m for 12 cols
    cell_height = 9.0 / grid_rows    # 1.0m for 9 rows
    
    # Convert 0-based matrix indices to 1-based grid coordinates
    grid_row = row + 1
    grid_col = col + 1
    
    # Calculate grid center coordinates
    center_x = left_edge + (grid_col - 1 + 0.5) * cell_width
    center_y = top_edge - (grid_row - 1 + 0.5) * cell_height
    
    return np.array([center_x, center_y])


def select_fixed_goal(env, seed: int, goal_cell: Optional[Tuple[int, int]] = None, maze_map=None) -> Tuple[int, int]:
    """
    Select a fixed goal cell deterministically.
    
    Args:
        env: PointMaze environment
        seed: Random seed for reproducibility (uses config.a_seed)
        goal_cell: Optional (row, col) tuple to use directly. If None, randomly selects from valid cells.
        maze_map: Optional maze map array
        
    Returns:
        goal_cell: (row, col) tuple with 0-based indexing
    """
    if goal_cell is not None:
        # Validate it's a valid cell
        valid_cells = get_valid_cells(env, maze_map)
        if goal_cell not in valid_cells:
            raise ValueError(f"Provided goal_cell {goal_cell} is not a valid (non-wall) cell")
        return goal_cell
    
    # Get all valid cells
    valid_cells = get_valid_cells(env, maze_map)
    
    if len(valid_cells) == 0:
        raise ValueError("No valid cells found in maze")
    
    # Use seed to deterministically select a goal
    # Sort valid cells for reproducibility
    valid_cells_sorted = sorted(valid_cells)
    rng = np.random.RandomState(seed)
    goal_cell = valid_cells_sorted[rng.randint(len(valid_cells_sorted))]
    
    return goal_cell


def select_diverse_goals(env, n_goals: int, seed: int, maze_map=None) -> List[Tuple[int, int]]:
    """
    Select N diverse goals using k-means clustering on valid cells.
    
    Args:
        env: PointMaze environment
        n_goals: Number of goals to select
        seed: Random seed for reproducibility (uses config.a_seed)
        maze_map: Optional maze map array
        
    Returns:
        goal_cells: List of (row, col) tuples with 0-based indexing
    """
    from sklearn.cluster import KMeans
    
    # Get all valid cells
    valid_cells = get_valid_cells(env, maze_map)
    
    if len(valid_cells) == 0:
        raise ValueError("No valid cells found in maze")
    
    if n_goals > len(valid_cells):
        raise ValueError(f"Requested {n_goals} goals but only {len(valid_cells)} valid cells available")
    
    if n_goals == len(valid_cells):
        # Return all valid cells
        return valid_cells
    
    # Get grid dimensions from maze map
    if maze_map is None:
        unwrapped_env = env.unwrapped
        if hasattr(unwrapped_env, 'maze') and hasattr(unwrapped_env.maze, 'maze_map'):
            maze_map = unwrapped_env.maze.maze_map
        else:
            maze_map = get_maze_map()
    
    # Ensure maze_map is a numpy array (convert from list if needed)
    if not isinstance(maze_map, np.ndarray):
        maze_map = np.array(maze_map)
    
    grid_rows, grid_cols = maze_map.shape
    
    # Convert valid cells to continuous coordinates for clustering
    coordinates = np.array([cell_to_continuous_coords(cell, grid_rows, grid_cols) for cell in valid_cells])
    
    # Use k-means to find diverse goal locations
    # Set random_state for reproducibility
    kmeans = KMeans(n_clusters=n_goals, random_state=seed, n_init=10)
    kmeans.fit(coordinates)
    
    # Get cluster centers and snap to nearest valid cells
    cluster_centers = kmeans.cluster_centers_
    goal_cells = []
    
    for center in cluster_centers:
        # Find nearest valid cell to cluster center
        distances = np.linalg.norm(coordinates - center, axis=1)
        nearest_idx = np.argmin(distances)
        goal_cells.append(valid_cells[nearest_idx])
    
    # Remove duplicates (in case multiple centers map to same cell)
    goal_cells = list(dict.fromkeys(goal_cells))  # Preserves order
    
    # If we have fewer than n_goals due to duplicates, add more from remaining valid cells
    if len(goal_cells) < n_goals:
        remaining = [cell for cell in valid_cells if cell not in goal_cells]
        rng = np.random.RandomState(seed)
        needed = n_goals - len(goal_cells)
        if len(remaining) >= needed:
            additional = rng.choice(len(remaining), size=needed, replace=False)
            goal_cells.extend([remaining[i] for i in additional])
        else:
            # If not enough remaining, just use what we have
            goal_cells.extend(remaining)
    
    return goal_cells[:n_goals]  # Ensure we return exactly n_goals

