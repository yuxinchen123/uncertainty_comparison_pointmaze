"""Tests for rnd_exploration.envs.point_maze_utils (coordinate mapping + cell selection)."""
import numpy as np
import pytest

from rnd_exploration.envs.point_maze_utils import (
    get_maze_map,
    observation_to_grid,
    velocity_to_grid,
    get_valid_cells,
    select_fixed_goal_bottom_left,
    select_fixed_goal_top_right,
    select_fixed_cell,
)


# ---------------------------------------------------------------------------
# Small hand-built mazes (object dtype so int 0/1 and letter cells coexist).
# 4x4 ring of open cells:
#   row0 [1 1 1 1]
#   row1 [1 0 0 1]
#   row2 [1 0 0 1]
#   row3 [1 1 1 1]
# open cells = (1,1),(1,2),(2,1),(2,2)
# ---------------------------------------------------------------------------
SMALL_MAZE = np.array(
    [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
    ]
)
SMALL_VALID_SORTED = [(1, 1), (1, 2), (2, 1), (2, 2)]
ALL_WALLS = np.ones((3, 3), dtype=int)


# Fake env objects exercise get_maze_map without paying for a real gym.make().
class _FakeMaze:
    """Holds a maze_map attribute, mimicking env.unwrapped.maze."""

    def __init__(self, maze_map):
        self.maze_map = maze_map


class _FakeUnwrapped:
    """Stand-in for env.unwrapped that owns a .maze."""

    def __init__(self, maze_map):
        self.maze = _FakeMaze(maze_map)


class _FakeEnv:
    """Minimal env exposing .unwrapped, used to test get_maze_map's two branches."""

    def __init__(self, maze_map):
        self.unwrapped = _FakeUnwrapped(maze_map)


# ---------------------------------------------------------------------------
# observation_to_grid: (x, y) -> (row, col)
# ---------------------------------------------------------------------------
def test_observation_to_grid_golden_path():
    """Center, top-left and bottom-right of the X[-6,6] Y[-4.5,4.5] world map to expected cells."""
    # center of the world is (0,0): col=int(0+6)=6, row=int(4.5-0)=4
    assert observation_to_grid([0.0, 0.0]) == (4, 6)
    # top-left world corner (x=-6, y=+4.5) -> grid origin (row 0, col 0)
    assert observation_to_grid([-6.0, 4.5]) == (0, 0)
    # near bottom-right interior (x=5.9, y=-4.4) -> last row/col of 9x12 grid
    assert observation_to_grid([5.9, -4.4]) == (8, 11)


def test_observation_to_grid_accepts_dict_and_ignores_velocity():
    """A dict obs uses observation[:2]; the velocity tail (indices 2:) must not change the cell."""
    # only the first two entries (x, y) are read; the trailing velocity is ignored
    flat = observation_to_grid([0.0, 0.0])
    as_dict = observation_to_grid({"observation": np.array([0.0, 0.0, 9.0, -9.0])})
    assert as_dict == flat == (4, 6)


def test_observation_to_grid_edge_clipping_out_of_bounds():
    """Coordinates outside the world box clamp to the boundary rows/cols instead of going negative or overflowing."""
    # x at/over right edge -> last col (uses >=); x far left -> col 0
    assert observation_to_grid([6.0, 0.0])[1] == 11
    assert observation_to_grid([100.0, 0.0])[1] == 11
    assert observation_to_grid([-100.0, 0.0])[1] == 0
    # y boundaries: y==+4.5 stays row 0, y just above stays row 0; y<=-4.5 -> last row 8
    assert observation_to_grid([0.0, 4.5])[0] == 0
    assert observation_to_grid([0.0, 4.6])[0] == 0
    assert observation_to_grid([0.0, -4.5])[0] == 8
    assert observation_to_grid([0.0, -100.0])[0] == 8


# ---------------------------------------------------------------------------
# velocity_to_grid: (vx, vy) -> (vx_bin, vy_bin), clipped to [-5, 5]
# ---------------------------------------------------------------------------
def test_velocity_to_grid_golden_path():
    """Zero velocity lands in the middle bin (5); -5 maps to bin 0 for a 10-bin split of [-5,5]."""
    # span 10 over 10 bins -> bin index = int(v + 5)
    assert velocity_to_grid([0.0, 0.0, 0.0, 0.0]) == (5, 5)
    assert velocity_to_grid([0.0, 0.0, -5.0, -5.0]) == (0, 0)


def test_velocity_to_grid_clips_at_edges():
    """Velocities at or beyond +-5 clamp to the outermost bins (0 and n_bins-1=9), never out of range."""
    # +5 (and anything above) -> top bin 9; below -5 -> bottom bin 0
    assert velocity_to_grid([0.0, 0.0, 5.0, 5.0]) == (9, 9)
    assert velocity_to_grid([0.0, 0.0, 100.0, 100.0]) == (9, 9)
    assert velocity_to_grid([0.0, 0.0, -100.0, -100.0]) == (0, 0)
    # dict form reads indices 2:4 and gives the same answer as the flat form
    assert velocity_to_grid({"observation": np.array([0.0, 0.0, 0.0, 0.0])}) == (5, 5)


def test_velocity_to_grid_custom_n_bins_stays_in_range():
    """With a custom n_bins every returned bin is within [0, n_bins-1], including the clipped top edge."""
    # sweep a range of velocities; both bins must stay inside [0, n_bins)
    for v in np.linspace(-10.0, 10.0, 21):
        vx_bin, vy_bin = velocity_to_grid([0.0, 0.0, v, -v], n_bins=4)
        assert 0 <= vx_bin <= 3
        assert 0 <= vy_bin <= 3
    # the exact top edge clips to the last bin (3), not 4
    assert velocity_to_grid([0.0, 0.0, 5.0, 5.0], n_bins=4) == (3, 3)


# ---------------------------------------------------------------------------
# get_valid_cells: open (non-wall) cells of a maze map
# ---------------------------------------------------------------------------
def test_get_valid_cells_golden_path():
    """Open (0) cells of the 4x4 ring are returned; walls (1) are excluded; the count matches the zeros."""
    valid = get_valid_cells(None, maze_map=SMALL_MAZE)
    assert sorted(valid) == SMALL_VALID_SORTED
    # every returned cell is an open cell, none is a wall
    assert all(SMALL_MAZE[r, c] == 0 for r, c in valid)


def test_get_valid_cells_treats_goal_reset_combined_letters_as_open():
    """An object-dtype map with 'g'/'r'/'c' markers counts those as open along with int 0."""
    # object dtype keeps int 0 and the letter markers distinct (a str-dtype array would not match 0)
    letter_maze = np.array(
        [
            [1, 1, 1],
            [1, 0, "g"],
            [1, "r", "c"],
        ],
        dtype=object,
    )
    valid = sorted(get_valid_cells(None, maze_map=letter_maze))
    # 0 at (1,1), 'g' at (1,2), 'r' at (2,1), 'c' at (2,2) are all open
    assert valid == [(1, 1), (1, 2), (2, 1), (2, 2)]


def test_get_valid_cells_all_walls_is_empty():
    """A maze with no open cells returns an empty list rather than raising."""
    assert get_valid_cells(None, maze_map=ALL_WALLS) == []


# ---------------------------------------------------------------------------
# select_fixed_goal_bottom_left / select_fixed_goal_top_right
# ---------------------------------------------------------------------------
def test_select_fixed_goal_corners_golden_path():
    """Bottom-left = (max row, min col); top-right = (min row, max col) over the open cells."""
    # 4x4 ring: bottom-left open cell is (2,1), top-right open cell is (1,2)
    assert select_fixed_goal_bottom_left(None, maze_map=SMALL_MAZE) == (2, 1)
    assert select_fixed_goal_top_right(None, maze_map=SMALL_MAZE) == (1, 2)


def test_select_fixed_goal_raises_when_no_open_cells():
    """Both corner selectors raise ValueError on an all-wall maze (edge case: nothing to pick)."""
    with pytest.raises(ValueError):
        select_fixed_goal_bottom_left(None, maze_map=ALL_WALLS)
    with pytest.raises(ValueError):
        select_fixed_goal_top_right(None, maze_map=ALL_WALLS)


# ---------------------------------------------------------------------------
# select_fixed_cell: seeded deterministic pick, with exclude/force options
# ---------------------------------------------------------------------------
def test_select_fixed_cell_seeded_determinism():
    """Same seed gives the same valid cell every call and matches the documented RandomState draw."""
    # repeat with one seed -> identical result (deterministic)
    first = select_fixed_cell(None, seed=7, maze_map=SMALL_MAZE)
    second = select_fixed_cell(None, seed=7, maze_map=SMALL_MAZE)
    assert first == second
    assert first in SMALL_VALID_SORTED
    # result equals indexing sorted(valid) with the same RandomState(seed) draw it uses internally
    idx = np.random.RandomState(7).randint(len(SMALL_VALID_SORTED))
    assert first == SMALL_VALID_SORTED[idx]


def test_select_fixed_cell_different_seeds_can_differ_but_all_valid():
    """Across several seeds every pick is a valid open cell, and not all seeds collapse to one cell."""
    # collect picks for a spread of seeds
    picks = {select_fixed_cell(None, seed=s, maze_map=SMALL_MAZE) for s in range(20)}
    assert picks.issubset(set(SMALL_VALID_SORTED))
    # the seed actually drives variety (more than one distinct cell over 20 seeds)
    assert len(picks) > 1


def test_select_fixed_cell_force_and_exclude():
    """force_cell returns that cell when valid; an invalid force_cell or excluding all cells raises."""
    # force_cell short-circuits to the requested valid cell
    assert select_fixed_cell(None, seed=0, force_cell=(2, 2), maze_map=SMALL_MAZE) == (2, 2)
    # a force_cell that is a wall is rejected
    with pytest.raises(ValueError):
        select_fixed_cell(None, seed=0, force_cell=(0, 0), maze_map=SMALL_MAZE)
    # excluding every open cell leaves nothing to choose -> ValueError
    with pytest.raises(ValueError):
        select_fixed_cell(None, seed=0, exclude_cells=SMALL_VALID_SORTED, maze_map=SMALL_MAZE)


def test_select_fixed_cell_exclude_one_cell_narrows_pool():
    """Excluding a single cell removes it from the candidate pool while picks stay deterministic."""
    # exclude (1,1); the returned cell must never be the excluded one and remains seed-stable
    excluded = (1, 1)
    remaining = [c for c in SMALL_VALID_SORTED if c != excluded]
    result = select_fixed_cell(None, seed=3, exclude_cells=[excluded], maze_map=SMALL_MAZE)
    assert result != excluded
    assert result in remaining
    # the draw indexes the sorted remaining pool with RandomState(seed)
    idx = np.random.RandomState(3).randint(len(remaining))
    assert result == sorted(remaining)[idx]


# ---------------------------------------------------------------------------
# get_maze_map: extraction + copy semantics
# ---------------------------------------------------------------------------
def test_get_maze_map_returns_copy_and_converts_list():
    """get_maze_map returns an ndarray (converting a list) and a copy that does not alias the source array."""
    # list input is converted to ndarray
    list_map = [[1, 1], [1, 0]]
    out = get_maze_map(_FakeEnv(list_map))
    assert isinstance(out, np.ndarray)
    assert out.tolist() == list_map
    # ndarray input is copied: mutating the result leaves the source untouched
    src = np.array([[1, 1], [1, 0]])
    returned = get_maze_map(_FakeEnv(src))
    returned[0, 0] = 999
    assert src[0, 0] == 1


def test_get_maze_map_missing_maze_returns_none():
    """An env whose unwrapped object has no .maze yields None (edge case: not a PointMaze)."""

    # unwrapped without a .maze attribute -> None
    class _NoMaze:
        pass

    class _EnvNoMaze:
        unwrapped = _NoMaze()

    assert get_maze_map(_EnvNoMaze()) is None


# ---------------------------------------------------------------------------
# Real PointMaze_Large-v3 map (integration; one env build, kept fast)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def point_maze_env():
    """Build PointMaze_Large-v3 once for the module and close it afterwards."""
    # register the gymnasium-robotics envs, then make the large point maze
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
    env = gym.make("PointMaze_Large-v3")
    yield env
    env.close()


def test_real_maze_valid_cells_and_corners(point_maze_env):
    """On the real 9x12 PointMaze_Large map, valid cells match the zeros and corner goals are the expected open cells."""
    # the extracted map is 9x12 with walled corners
    maze_map = get_maze_map(point_maze_env)
    assert maze_map.shape == (9, 12)
    assert maze_map[0, 0] == 1 and maze_map[-1, -1] == 1
    # number of open cells equals the number of zero entries in the map
    valid = get_valid_cells(point_maze_env)
    assert len(valid) == int((np.array(maze_map) == 0).sum())
    # known corner goals for this fixed map layout
    assert select_fixed_goal_bottom_left(point_maze_env) == (7, 1)
    assert select_fixed_goal_top_right(point_maze_env) == (1, 10)
    # a seeded start excluding the goal is deterministic and never the goal cell
    goal = select_fixed_goal_bottom_left(point_maze_env)
    start_a = select_fixed_cell(point_maze_env, seed=0, exclude_cells=[goal])
    start_b = select_fixed_cell(point_maze_env, seed=0, exclude_cells=[goal])
    assert start_a == start_b != goal
