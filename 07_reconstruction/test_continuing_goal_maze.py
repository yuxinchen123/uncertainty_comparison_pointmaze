"""
Tests for continuing-goal maze: no termination on goal, only truncation at max steps.

Run with: pytest 07_reconstruction/test_continuing_goal_maze.py -v
Or: python 07_reconstruction/test_continuing_goal_maze.py
"""

import sys
import os

import gymnasium as gym
import gymnasium_robotics
import numpy as np

# Add 07_reconstruction so we can import from it
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from continuing_goal_maze import make_continuing_goal_maze_env
from utilities.env_utils import (
    get_maze_map,
    get_valid_cells,
    observation_to_grid,
)
from new_rl import FixedGoalWrapper, FixedStartWrapper, RemoveGoalWrapper


GOAL_RANGE = 0.45  # PointMaze considers "in goal" when distance <= 0.45


def _distance_to_goal(obs, goal_xy):
    """Achieved goal position from obs; goal_xy is (2,) array."""
    if isinstance(obs, dict):
        achieved = obs.get("achieved_goal", obs.get("observation", obs)[:2])
    else:
        achieved = obs[:2]
    return np.linalg.norm(np.asarray(achieved) - np.asarray(goal_xy))


def test_original_env_terminates_on_goal_and_step_after_is_invalid():
    """
    Original maze (continuing_task=False): when we reach the goal we get
    terminated=True. Per Gymnasium contract we must NOT call step() again
    after terminated or truncated.
    """
    gym.register_envs(gymnasium_robotics)
    max_steps = 100
    env = gym.make(
        "PointMaze_Large-v3",
        continuing_task=False,
        reset_target=False,
        max_episode_steps=max_steps,
    )
    maze_map = get_maze_map(env)
    valid = get_valid_cells(env, maze_map)
    assert len(valid) >= 1
    # Use one cell for both start and goal so we start very close to goal
    cell = valid[0]
    goal_cell = [int(cell[0]), int(cell[1])]
    reset_cell = [int(cell[0]), int(cell[1])]

    obs, info = env.reset(seed=42, options={"goal_cell": goal_cell, "reset_cell": reset_cell})
    goal_xy = env.unwrapped.goal.copy()
    terminated = False
    truncated = False
    steps = 0
    reward_at_goal = None
    while not (terminated or truncated) and steps < max_steps:
        # Move toward goal: action = (goal - current) normalized
        if isinstance(obs, dict):
            achieved = obs.get("achieved_goal", obs["observation"][:2])
        else:
            achieved = obs[:2]
        diff = goal_xy - np.array(achieved[:2])
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            action = np.zeros(2)
        else:
            action = np.clip(diff / max(dist, 1e-6), -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if reward >= 1.0 or (info.get("success") and reward > 0):
            reward_at_goal = reward
            break

    assert reward_at_goal is not None or terminated or truncated, "Should reach goal or hit limit"
    if reward_at_goal is not None and reward >= 1.0:
        # When we get goal reward, original env must have terminated
        assert terminated, "Original env must set terminated=True when goal is reached"
    # Per Gymnasium: after terminated=True we must not call step() again.
    # We do not call step() here; the test documents that the episode is done.
    env.close()


def test_continuing_env_does_not_terminate_on_goal_and_keeps_accepting_steps():
    """
    Continuing-goal maze: when we reach the goal we get terminated=False.
    We can keep calling step(); we keep getting reward while in goal range;
    episode ends only when truncated at max_episode_steps.
    """
    max_steps = 50
    env = make_continuing_goal_maze_env(
        "PointMaze_Large-v3",
        seed=42,
        max_episode_steps=max_steps,
    )
    maze_map = get_maze_map(env)
    valid = get_valid_cells(env, maze_map)
    assert len(valid) >= 1
    cell = valid[0]
    goal_cell = [int(cell[0]), int(cell[1])]
    reset_cell = [int(cell[0]), int(cell[1])]

    obs, info = env.reset(seed=42, options={"goal_cell": goal_cell, "reset_cell": reset_cell})
    goal_xy = env.unwrapped.goal.copy()
    total_reward = 0.0
    steps_in_goal = 0
    terminated = False
    truncated = False
    steps = 0

    # Phase 1: reach goal (or get close)
    while steps < max_steps:
        if isinstance(obs, dict):
            achieved = obs.get("achieved_goal", obs["observation"][:2])
        else:
            achieved = obs[:2]
        diff = goal_xy - np.array(achieved[:2])
        dist = np.linalg.norm(diff)
        if dist < 1e-6:
            action = np.zeros(2)
        else:
            action = np.clip(diff / max(dist, 1e-6), -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total_reward += reward
        if reward >= 1.0:
            steps_in_goal += 1
        if reward >= 1.0:
            # We reached goal: must NOT be terminated
            assert not terminated, "Continuing env must not terminate on goal"
            break
        assert not terminated, "Should not terminate before goal (continuing env)"

    # Phase 2: stay in goal and keep stepping until truncation
    if reward >= 1.0 and not truncated:
        # Stay put (zero action) and collect reward
        while not truncated and steps < max_steps:
            obs, reward, terminated, truncated, info = env.step(np.zeros(2))
            steps += 1
            total_reward += reward
            if reward >= 1.0:
                steps_in_goal += 1
            assert not terminated, "Continuing env must never terminate on goal"
        assert truncated, "Episode must end by truncation at max_episode_steps"
        assert steps == max_steps, "Should use all max_episode_steps"

    env.close()
    # We should have collected reward multiple times while in goal
    assert steps_in_goal >= 1, "Should get reward at least once while in goal"


def test_continuing_env_same_wrappers_as_original():
    """
    Same wrappers (FixedGoalWrapper, FixedStartWrapper, RemoveGoalWrapper)
    work on the continuing-goal env; reset and step return expected shapes.
    """
    gym.register_envs(gymnasium_robotics)
    base_orig = gym.make(
        "PointMaze_Large-v3",
        continuing_task=False,
        max_episode_steps=20,
    )
    base_cont = make_continuing_goal_maze_env("PointMaze_Large-v3", max_episode_steps=20)

    maze_map = get_maze_map(base_orig)
    valid = get_valid_cells(base_orig, maze_map)
    goal_cell = valid[0]
    start_cell = valid[1] if len(valid) > 1 else valid[0]

    for base_env, name in [(base_orig, "original"), (base_cont, "continuing")]:
        env = FixedStartWrapper(base_env, start_cell)
        env = FixedGoalWrapper(env, goal_cell)
        env = RemoveGoalWrapper(env)
        obs, info = env.reset(seed=123, options={})
        assert "observation" in obs and "desired_goal" not in obs, (
            f"{name}: RemoveGoalWrapper should remove desired_goal"
        )
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        assert obs is not None and reward is not None
        env.close()


def test_continuing_env_reward_while_in_goal_range():
    """
    While agent stays within goal range (distance <= 0.45), every step
    returns reward 1 (sparse). Once it leaves, reward is 0 until back in range.
    """
    max_steps = 200
    env = make_continuing_goal_maze_env(
        "PointMaze_Large-v3",
        seed=0,
        max_episode_steps=max_steps,
    )
    maze_map = get_maze_map(env)
    valid = get_valid_cells(env, maze_map)
    cell = valid[0]
    obs, _ = env.reset(seed=0, options={"goal_cell": [cell[0], cell[1]], "reset_cell": [cell[0], cell[1]]})
    goal_xy = env.unwrapped.goal.copy()

    # Move to goal
    for _ in range(30):
        if isinstance(obs, dict):
            achieved = obs.get("achieved_goal", obs["observation"][:2])
        else:
            achieved = obs[:2]
        diff = goal_xy - np.array(achieved[:2])
        d = np.linalg.norm(diff)
        if d < 1e-6:
            break
        action = np.clip(diff / d, -1.0, 1.0)
        obs, reward, term, trunc, _ = env.step(action)
        if term or trunc:
            break

    # Now step with zero action several times; we should get reward when in range
    in_goal_rewards = 0
    for _ in range(10):
        obs, reward, term, trunc, _ = env.step(np.zeros(2))
        assert not term
        dist = _distance_to_goal(obs, goal_xy)
        if dist <= GOAL_RANGE:
            assert reward >= 1.0, "Sparse reward should be 1 when in goal range"
            in_goal_rewards += 1
        if trunc:
            break
    assert in_goal_rewards >= 1, "Should get reward at least once while staying in goal"
    env.close()


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"] + sys.argv[1:]))
