"""
Full list of observations for valid (non-wall) cells in a PointMaze-style maze_map.
Used to compute intrinsic reward vectors over the full state space for distance-to-GT.
"""
import numpy as np

# Match PointMaze_Large: X[-6, 6], Y[-4.5, 4.5], grid_rows x grid_cols
LEFT, RIGHT = -6.0, 6.0
BOTTOM, TOP = -4.5, 4.5
VELOCITY_N_BINS = 10
V_MIN, V_MAX = -5.0, 5.0


def _grid_to_xy(row: int, col: int, grid_rows: int, grid_cols: int) -> tuple:
    """Convert grid (row, col) to continuous (x, y) at cell center. Inverse of observation_to_grid."""
    cell_width = (RIGHT - LEFT) / grid_cols
    cell_height = (TOP - BOTTOM) / grid_rows
    x = LEFT + (col + 0.5) * cell_width
    y = TOP - (row + 0.5) * cell_height
    return (x, y)


def _velocity_bin_to_v(v_bin: int, n_bins: int = VELOCITY_N_BINS) -> float:
    """Center velocity for bin index in [0, n_bins-1]."""
    span = V_MAX - V_MIN
    return V_MIN + (v_bin + 0.5) * (span / n_bins)


def _valid_cells(maze_map: np.ndarray):
    """Yield (row, col) for non-wall cells. maze_map: 0 = open, 1 = wall."""
    grid_rows, grid_cols = maze_map.shape
    for row in range(grid_rows):
        for col in range(grid_cols):
            v = maze_map[row, col]
            if isinstance(v, (str, np.str_)):
                # PointMaze cell type symbols: g=goal, r=robot/start, c=cell (open), "0"=open (string)
                if v in ("g", "r", "c", "0"):
                    yield row, col
            elif int(v) == 0:
                yield row, col


def full_observation_list(maze_map: np.ndarray) -> list:
    """
    Return a list of observation vectors for every valid (non-wall) cell in the maze.
    Each (row, col) is expanded over all velocity bins (VELOCITY_N_BINS x VELOCITY_N_BINS),
    so each observation is [x, y, vx, vy] (position + velocity). Same ordering as
    PositionVelocityVisitCountWrapper for distance-to-GT comparison.
    """
    maze_map = np.asarray(maze_map)
    grid_rows, grid_cols = maze_map.shape
    out = []
    for row, col in _valid_cells(maze_map):
        x, y = _grid_to_xy(row, col, grid_rows, grid_cols)
        for vx_bin in range(VELOCITY_N_BINS):
            for vy_bin in range(VELOCITY_N_BINS):
                vx = _velocity_bin_to_v(vx_bin)
                vy = _velocity_bin_to_v(vy_bin)
                out.append(np.array([x, y, vx, vy], dtype=np.float32))
    return out
