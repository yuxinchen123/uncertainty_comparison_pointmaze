"""PointMaze environment utilities (07_reconstruction, self-contained)."""
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

    # X -> col (0-based)
    if x < left_edge:
        col = 0
    elif x >= right_edge:
        col = grid_cols - 1
    else:
        col = int((x - left_edge) / cell_width)
        col = min(col, grid_cols - 1)

    # Y -> row (0-based; row 0 = top, row grid_rows-1 = bottom, matches maze_map indexing)
    if y <= bottom_edge:
        row = grid_rows - 1
    elif y > top_edge:
        row = 0
    else:
        y_offset_from_top = top_edge - y
        row = int(y_offset_from_top / cell_height)
        row = min(row, grid_rows - 1)

    return row, col


def get_valid_cells(env, maze_map=None):
    """Get all open (non-wall) cells. Returns list of (row, col) 0-based tuples."""
    if maze_map is None:
        maze_map = get_maze_map(env)
    if maze_map is None:
        return []
    m = np.array(maze_map)
    valid = []
    for row in range(m.shape[0]):
        for col in range(m.shape[1]):
            v = m[row, col]
            if v == 0 or v == "g" or v == "r" or v == "c":
                valid.append((row, col))
    return valid


def select_fixed_goal_top_left(env, maze_map=None):
    """
    Return the visual top-left valid (open) cell as the fixed goal.
    Heatmap uses invert_yaxis(), so high row = visual top. Top-left = max row, min col.
    Returns (row, col) 0-based.
    """
    valid = get_valid_cells(env, maze_map)
    if not valid:
        raise ValueError("No valid cells in maze")
    return max(valid, key=lambda c: (c[0], -c[1]))


def select_fixed_goal_bottom_right(env, maze_map=None):
    """
    Return the visual bottom-right valid (open) cell as the fixed goal.
    Heatmap uses invert_yaxis(), so low row = visual bottom. Bottom-right = min row, max col.
    Returns (row, col) 0-based.
    """
    valid = get_valid_cells(env, maze_map)
    if not valid:
        raise ValueError("No valid cells in maze")
    return min(valid, key=lambda c: (c[0], -c[1]))


def select_fixed_cell(env, seed: int, exclude_cells=None, force_cell=None, maze_map=None):
    """
    Select a fixed cell deterministically from valid cells (by seed).
    exclude_cells: iterable of (row,col) to exclude (e.g. goal when picking start).
    force_cell: if provided, return it (after validation).
    Returns (row, col) 0-based.
    """
    valid = get_valid_cells(env, maze_map)
    if not valid:
        raise ValueError("No valid cells in maze")
    if force_cell is not None:
        if force_cell not in valid:
            raise ValueError(f"force_cell {force_cell} not in valid cells")
        return force_cell
    if exclude_cells:
        exclude_set = set(exclude_cells)
        valid = [c for c in valid if c not in exclude_set]
    if not valid:
        raise ValueError("No valid cells after exclusion")
    valid_sorted = sorted(valid)
    rng = np.random.RandomState(seed)
    return valid_sorted[rng.randint(len(valid_sorted))]
