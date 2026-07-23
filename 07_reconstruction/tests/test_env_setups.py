"""Unit tests for the section-8 env setups: registry names/cells, exact-center resets,
reward shift, termination semantics, grid subdivision, the 29-d AntMaze state, and the
1/sqrt(n) visit-count decay."""
import numpy as np
import pytest
import gymnasium as gym
import gymnasium_robotics

from rnd_exploration.envs.env_setups import ENV_SETUPS, ENV_SETUP_NAMES
from rnd_exploration.envs.point_maze_utils import (
    get_valid_cells,
    observation_to_grid,
    select_corner_cell,
)
from rnd_exploration.envs.point_maze_wrappers import (
    PositionVisitCountWrapper,
    RewardShiftWrapper,
)
from rnd_exploration.methods import REGISTRY
from rnd_exploration.methods.visit_count import VisitCount

gym.register_envs(gymnasium_robotics)


def _registered_map(env_id):
    """Read an env id's maze_map array straight from the gym registry (no env instantiation)."""
    return np.array(gym.spec(env_id).kwargs["maze_map"])


def test_registry_names_and_count():
    """17 setups: 8 env ids x 2 start corners + the recorded previous setup."""
    assert len(ENV_SETUPS) == 17
    assert "initial_single_large_pointmaze_max_400" in ENV_SETUPS
    for name in ENV_SETUP_NAMES:
        assert ENV_SETUPS[name].name == name


def test_section8_cells_are_open_and_match_corners():
    """Every start/goal cell is an open cell of its registered map, and the section-8 cells equal
    the generic corner selection (UMaze pairs the U ends; others go corner-to-corner)."""
    for name, setup in ENV_SETUPS.items():
        maze_map = _registered_map(setup.env_id)
        valid = set(get_valid_cells(None, maze_map=maze_map))
        assert tuple(setup.start_cell) in valid, f"{name}: start {setup.start_cell} is not open"
        assert tuple(setup.goal_cell) in valid, f"{name}: goal {setup.goal_cell} is not open"
    # corner-derivation check on the two Large section-8 setups (row 0 top: bottom_left=(7,1), ...)
    large_map = _registered_map("PointMaze_Large-v3")
    assert select_corner_cell(None, "bottom_left", maze_map=large_map) == (7, 1)
    assert select_corner_cell(None, "top_left", maze_map=large_map) == (1, 1)
    assert select_corner_cell(None, "top_right", maze_map=large_map) == (1, 10)
    assert select_corner_cell(None, "bottom_right", maze_map=large_map) == (7, 10)
    assert ENV_SETUPS["PointMaze_Large-v3_start_bottom_left"].start_cell == (7, 1)
    assert ENV_SETUPS["PointMaze_Large-v3_start_bottom_left"].goal_cell == (1, 10)
    assert ENV_SETUPS["AntMaze_UMaze-v5_start_top_left"].start_cell == (1, 1)
    assert ENV_SETUPS["AntMaze_UMaze-v5_start_top_left"].goal_cell == (3, 1)


def test_previous_setup_recorded_verbatim():
    """initial_single_large_pointmaze_max_400 must equal the pre-section-8 configuration."""
    s = ENV_SETUPS["initial_single_large_pointmaze_max_400"]
    assert (s.env_id, s.start_cell, s.goal_cell) == ("PointMaze_Large-v3", (7, 1), (1, 10))
    assert s.continuing_task is True and s.reset_target is False
    assert s.position_noise_range == 0.25 and s.max_episode_steps == 400
    assert s.reward_shift == 0.0 and s.discount_factor == 0.999
    assert s.attach_xy_to_state is False


def test_exact_center_reset_with_zero_noise():
    """position_noise_range=0 (set on the built env) puts goal and start at exact cell centers."""
    env = gym.make("PointMaze_UMaze-v3", continuing_task=False, reset_target=False)
    env.unwrapped.position_noise_range = 0.0
    obs, _ = env.reset(seed=0, options={"goal_cell": np.array([1, 1]), "reset_cell": np.array([3, 1])})
    goal_center = env.unwrapped.maze.cell_rowcol_to_xy((1, 1))
    start_center = env.unwrapped.maze.cell_rowcol_to_xy((3, 1))
    np.testing.assert_allclose(env.unwrapped.goal, goal_center, atol=0.0)
    # the ball spawns at the start center exactly (no cell noise; qpos is set to reset_pos)
    np.testing.assert_allclose(obs["achieved_goal"], start_center, atol=1e-9)
    env.close()


def test_reward_shift_and_terminated_semantics():
    """RewardShiftWrapper adds -1 per step; compute_terminated fires only with continuing_task=False."""
    env = gym.make("PointMaze_UMaze-v3", continuing_task=False, reset_target=False)
    env.unwrapped.position_noise_range = 0.0
    shifted = RewardShiftWrapper(env, -1.0)
    shifted.reset(seed=0, options={"goal_cell": np.array([1, 1]), "reset_cell": np.array([3, 1])})
    _, reward, _, _, _ = shifted.step(np.zeros(2, dtype=np.float32))
    assert reward == -1.0  # far from the goal: sparse 0 shifted to -1
    # termination semantics straight from the env's own function: at the goal it terminates only
    # when the task is NOT continuing
    goal = np.asarray(env.unwrapped.goal)
    assert env.unwrapped.compute_terminated(goal, goal, {}) is True
    env_cont = gym.make("PointMaze_UMaze-v3", continuing_task=True)
    assert env_cont.unwrapped.compute_terminated(goal, goal, {}) is False
    env.close()
    env_cont.close()


@pytest.fixture(scope="module")
def ant_umaze():
    """One AntMaze env for the module (MuJoCo ant instantiation is slow)."""
    env = gym.make("AntMaze_UMaze-v5", continuing_task=False, reset_target=False,
                   include_cfrc_ext_in_observation=False)
    env.unwrapped.position_noise_range = 0.0
    yield env
    env.close()


def test_antmaze_observation_strips_xy_and_attach_restores_29d(ant_umaze):
    """AntMaze's observation key is 27-d without x, y; achieved_goal + observation = 29-d state."""
    obs, _ = ant_umaze.reset(seed=0, options={"goal_cell": np.array([1, 1]), "reset_cell": np.array([3, 1])})
    assert obs["observation"].shape == (27,)
    state = np.concatenate([obs["achieved_goal"], obs["observation"]])
    assert state.shape == (29,)
    # the re-attached head is the global torso x, y (equals the start cell center with zero noise)
    start_center = ant_umaze.unwrapped.maze.cell_rowcol_to_xy((3, 1))
    np.testing.assert_allclose(state[:2], start_center, atol=0.15)  # MuJoCo qpos init noise only


def test_subdivided_grid_16_squares_per_cell(ant_umaze):
    """subdivision=4 on a 4 m maze: grid 4x larger per axis, 16 open 1 m squares per open cell,
    and a cell-center position maps into the cell's own 4x4 sub-block."""
    cell_wrapper = PositionVisitCountWrapper(ant_umaze)
    sub_wrapper = PositionVisitCountWrapper(ant_umaze, subdivision=4)
    assert cell_wrapper.maze_map.shape == (5, 5)
    assert sub_wrapper.maze_map.shape == (20, 20)
    open_cells = (cell_wrapper.maze_map == 0).sum()
    open_squares = (sub_wrapper.maze_map == 0).sum()
    assert open_squares == 16 * open_cells
    assert sub_wrapper.cell_size == 1.0  # 4 m cell / subdivision 4
    # the center of cell (1, 1) lands in sub-rows/cols 4..7 (cell (1,1)'s own block)
    center = ant_umaze.unwrapped.maze.cell_rowcol_to_xy((1, 1))
    row, col = observation_to_grid(np.array([center[0], center[1]]), 20, 20, cell_size=1.0)
    assert 4 <= row <= 7 and 4 <= col <= 7
    assert sub_wrapper.maze_map[row, col] == 0


def test_observation_to_grid_backward_compatible():
    """Default arguments reproduce the old hardcoded PointMaze_Large mapping exactly."""
    # world (4.5, 3.0) = the top-right goal cell (1, 10); (-4.5, -3.0) = start (7, 1)
    assert observation_to_grid(np.array([4.5, 3.0, 0.0, 0.0])) == (1, 10)
    assert observation_to_grid(np.array([-4.5, -3.0, 0.0, 0.0])) == (7, 1)


def test_visit_count_decay_sqrt():
    """intrinsic_decay_rate -0.5 gives min(1, 1/sqrt(n)); count 4 -> 0.5; count 0 -> 1.0."""
    model = VisitCount(visit_count_wrapper=None, intrinsic_decay_rate=-0.5)
    assert model._count_to_bonus(4) == 0.5
    assert model._count_to_bonus(0) == 1.0
    assert model._count_to_bonus(1) == 1.0


def test_new_gt_algorithms_registered():
    """The two AntMaze ground-truth variants exist with the right wrapper kinds."""
    assert REGISTRY["gt_position_maze_cell"].gt_wrapper_kind == "position"
    assert REGISTRY["gt_position_1m"].gt_wrapper_kind == "position_1m"
    assert REGISTRY["gt_position_maze_cell"].kind == "visit_count"
    assert REGISTRY["gt_position_1m"].kind == "visit_count"


def test_apply_env_setup_cli_override_wins():
    """--env_setup fills env fields, but an explicitly passed flag keeps its CLI value."""
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "train_module", pathlib.Path(__file__).resolve().parents[1] / "train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    cfg = train.Config(env_setup="AntMaze_Large-v5_start_bottom_left", discount_factor=0.5)
    # explicit={'discount_factor'} simulates --discount_factor on the command line
    train._apply_env_setup(cfg, explicit={"discount_factor"})
    assert cfg.env_name == "AntMaze_Large-v5"
    assert cfg.continuing_task is False and cfg.reward_shift == -1.0
    assert cfg.env_max_episode == -1 and cfg.position_noise_range == 0.0
    assert cfg.attach_xy_to_state is True
    assert cfg.start_cell == "7,1" and cfg.goal_cell == "1,10"
    assert cfg.discount_factor == 0.5  # the explicit CLI value survived
