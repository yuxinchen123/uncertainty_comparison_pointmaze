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


def velocity_to_grid(observation, n_bins=10):
    """
    Map continuous (vx, vy) velocity to grid bins (vx_bin, vy_bin).
    PointMaze (Gymnasium-Robotics) clips velocity to [-5, 5] m/s; vx, vy are clamped to [-5.0, 5.0] inside this function.
    observation can be dict with 'observation' key or flat array; velocity is taken from indices 2:4 (vx, vy).
    Returns 0-based (vx_bin, vy_bin) in [0, n_bins-1].
    """
    v_min, v_max = -5.0, 5.0
    arr = observation["observation"][2:4] if isinstance(observation, dict) else observation[2:4]
    vx = np.clip(float(arr[0]), v_min, v_max)
    vy = np.clip(float(arr[1]), v_min, v_max)
    span = v_max - v_min
    vx_bin = int((vx - v_min) / span * n_bins)
    vy_bin = int((vy - v_min) / span * n_bins)
    vx_bin = min(vx_bin, n_bins - 1)
    vy_bin = min(vy_bin, n_bins - 1)
    return vx_bin, vy_bin


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


def select_fixed_goal_bottom_left(env, maze_map=None):
    """
    Return the open cell at array (max row, min col). In world coordinates -- row 0 at the top,
    x increasing right, y increasing up (the main.tex / visit-count-heatmap convention) -- this is
    the lower-left open cell: max row = lowest y, min col = lowest x. Returns (row, col) 0-based.
    """
    valid = get_valid_cells(env, maze_map)
    if not valid:
        raise ValueError("No valid cells in maze")
    return max(valid, key=lambda c: (c[0], -c[1]))


def select_fixed_goal_top_right(env, maze_map=None):
    """
    Return the open cell at array (min row, max col). In world coordinates -- row 0 at the top,
    x increasing right, y increasing up (the main.tex / visit-count-heatmap convention) -- this is
    the upper-right open cell: min row = highest y, max col = highest x. Returns (row, col) 0-based.
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
