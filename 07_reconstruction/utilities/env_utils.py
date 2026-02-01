"""Environment utilities for PointMaze (07_reconstruction, self-contained)."""
import numpy as np


def get_maze_map(env):
    """Extract maze map from PointMaze environment (env.unwrapped.maze.maze_map)."""
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "maze") and hasattr(unwrapped.maze, "maze_map"):
        m = unwrapped.maze.maze_map
        return np.array(m) if not isinstance(m, np.ndarray) else m.copy()
    return None


def observation_to_grid(observation, grid_rows=9, grid_cols=12):
    """
    Map continuous (x, y) observation to grid cell (row, col).
    Matches PointMaze coordinate system: X[-6.0, 6.0], Y[-4.5, 4.5] -> grid_rows x grid_cols.
    Returns 0-based (row, col) indices for numpy array indexing.
    Uses observation[:2] (x, y) for dict obs from 'observation' key.
    """
    arr = observation["observation"][:2] if isinstance(observation, dict) else observation[:2]
    x, y = float(arr[0]), float(arr[1])

    left_edge = -6.0
    right_edge = 6.0
    bottom_edge = -4.5
    top_edge = 4.5
    total_width = 12.0
    total_height = 9.0
    cell_width = total_width / grid_cols
    cell_height = total_height / grid_rows

    # X -> col (1-based in notebook, convert to 0-based)
    if x < left_edge:
        col_1 = 1
    elif x >= right_edge:
        col_1 = grid_cols
    else:
        col_1 = int((x - left_edge) / cell_width) + 1
        if abs(x - (left_edge + (col_1 - 1) * cell_width)) < 1e-10 and col_1 > 1:
            col_1 = col_1 - 1
        col_1 = max(1, min(col_1, grid_cols))
    col = col_1 - 1

    # Y -> row (1-based in notebook, top=1; convert to 0-based)
    if y <= bottom_edge:
        row_1 = grid_rows
    elif y > top_edge:
        row_1 = 1
    else:
        y_offset_from_top = top_edge - y
        row_1 = int(y_offset_from_top / cell_height) + 1
        if abs(y % 1.0 - 0.5) < 1e-10 and row_1 > 1:
            row_1 = row_1 - 1
        row_1 = max(1, min(row_1, grid_rows))
    row = row_1 - 1

    return row, col


