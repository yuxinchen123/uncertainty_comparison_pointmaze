"""Build deterministic path-shaped grids without starting a JAX computation."""

from dataclasses import dataclass

import numpy as np


ACTION_NAMES = ("up", "right", "down", "left")
ACTION_OFFSETS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass(frozen=True)
class GridWorld:
    """Store a path's physical cells, exact transitions, and coordinate observations."""

    kind: str
    width: int
    turns: int
    transition_table: np.ndarray
    coordinates: np.ndarray
    positions: np.ndarray
    grid: np.ndarray
    path_distance: np.ndarray
    start_state: int = 0

    @property
    def state_count(self) -> int:
        """Return the number of traversable cells, including start and goal."""
        # Wall entries in the physical grid are not states.
        return int(self.positions.shape[0])

    @property
    def n_states(self) -> int:
        """Expose the state count for trainer configurations."""
        # Both names refer to the same set of accessible cells.
        return self.state_count

    @property
    def goal_state(self) -> int:
        """Return the final cell along the path."""
        # State identifiers follow increasing distance from the start.
        return self.state_count - 1


def _assemble_grid_world(kind: str, width: int, turns: int, positions: list) -> GridWorld:
    """Construct table transitions and centered coordinates from an ordered path."""
    # Before: [(0, 0), (0, 1), ...]; after: state identifiers in a physical grid.
    position_array = np.asarray(positions, dtype=np.int32)
    height = int(position_array[:, 0].max()) + 1
    grid = np.full((height, width), -1, dtype=np.int32)
    grid[position_array[:, 0], position_array[:, 1]] = np.arange(len(positions))

    # A wall collision stays in the same state; for a 1x2 path, right maps 0 to 1.
    transition_table = np.repeat(np.arange(len(positions), dtype=np.int32)[:, None], 4, axis=1)
    for state, (row, column) in enumerate(positions):
        for action, (row_offset, column_offset) in enumerate(ACTION_OFFSETS):
            next_row, next_column = row + row_offset, column + column_offset
            if 0 <= next_row < height and 0 <= next_column < width:
                neighbor = grid[next_row, next_column]
                if neighbor >= 0:
                    transition_table[state, action] = neighbor

    # Before: row/column integers; after: x points right and y points up in [-1, 1].
    center = np.asarray([(width - 1) / 2, (height - 1) / 2], dtype=np.float32)
    scale = np.float32(max(width - 1, height - 1) / 2)
    coordinates = np.column_stack((position_array[:, 1], -position_array[:, 0])).astype(np.float32)
    coordinates -= np.asarray([center[0], -center[1]], dtype=np.float32)
    coordinates /= scale
    return GridWorld(kind, width, turns, transition_table, coordinates, position_array,
                     grid, np.arange(len(positions), dtype=np.int32))


def make_straight_grid_world(width: int) -> GridWorld:
    """Make a one-cell-wide corridor starting left and ending right."""
    # Even widths avoid an exact zero coordinate with zero-bias random networks.
    if not isinstance(width, (int, np.integer)) or width < 2 or width % 2:
        raise ValueError("Straight-grid width must be an even integer of at least two.")
    return _assemble_grid_world("straight", int(width), 0, [(0, column) for column in range(width)])


def make_snake_grid_world(width: int, turns: int) -> GridWorld:
    """Make horizontal corridors separated by walls and joined at alternating ends."""
    # An even number of turns gives an odd number of corridors and a bottom-right goal.
    if not isinstance(width, (int, np.integer)) or width < 2 or width % 2:
        raise ValueError("Snake-grid width must be an even integer of at least two.")
    if not isinstance(turns, (int, np.integer)) or turns < 2 or turns % 2:
        raise ValueError("Snake-grid turns must be a positive even integer of at least two.")

    # Before: width=4, turns=2; after: three corridors plus two joining cells, 14 states.
    positions = []
    for corridor in range(turns + 1):
        row = 2 * corridor
        columns = range(width) if corridor % 2 == 0 else range(width - 1, -1, -1)
        positions.extend((row, column) for column in columns)
        if corridor < turns:
            positions.append((row + 1, width - 1 if corridor % 2 == 0 else 0))
    return _assemble_grid_world("snake", int(width), int(turns), positions)


def make_geometry(kind: str, width: int, turns: int = 0) -> GridWorld:
    """Choose a straight or snake geometry using literal experiment settings."""
    # Reject unused turn settings so recorded settings describe the actual environment.
    if kind == "straight":
        if turns != 0:
            raise ValueError("A straight grid must have zero turns.")
        return make_straight_grid_world(width)
    if kind == "snake":
        return make_snake_grid_world(width, turns)
    raise ValueError(f"Unknown grid kind: {kind!r}.")
