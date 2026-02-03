"""
Continuing-goal PointMaze: reach goal does not end the episode.

Use this when you want:
- Episode continues after reaching the goal (no termination on goal).
- Reward is collected every step while the agent stays within the goal range
  (same reward logic as original: sparse 1 when distance <= 0.45, else 0).
- Episode ends only by truncation when max_episode_steps is reached.

Compatibility with original maze env:
- Same observation space, action space, and reward structure.
- Same reset(options={"goal_cell", "reset_cell"}) and wrappers (FixedGoalWrapper,
  FixedStartWrapper, RemoveGoalWrapper, VisitCountWrapper, etc.) work unchanged.
- The only difference is termination: original returns terminated=True when
  goal is reached (and you must not call step() again); this env never
  terminates on goal, so you can keep stepping and collecting reward until
  TimeLimit truncates.

Implementation: uses gymnasium_robotics PointMaze with continuing_task=True and
reset_target=False so the env never terminates and keeps the same goal when
reached. TimeLimit (max_episode_steps) is applied by gym.make() and causes
truncation only.
"""

import gymnasium as gym
import gymnasium_robotics


def make_continuing_goal_maze_env(
    env_name: str = "PointMaze_Large-v3",
    seed=None,
    max_episode_steps=None,
    **kwargs,
):
    """
    Create a PointMaze env that does not terminate on goal; only truncates at max steps.

    Args:
        env_name: Same IDs as original (e.g. PointMaze_Large-v3).
        seed: Optional; passed to env.reset(seed=seed) after creation.
        max_episode_steps: Optional; passed to gym.make (TimeLimit). If None,
            uses the env's default (e.g. 700 for PointMaze_Large-v3).
        **kwargs: Extra kwargs for gym.make (e.g. maze_map, reward_type).

    Returns:
        env: Same interface as gym.make(env_name, continuing_task=False), but
            with continuing_task=True and reset_target=False so the episode
            never terminates on goal and the goal stays fixed when reached.
    """
    gym.register_envs(gymnasium_robotics)
    # continuing_task=True: never terminate on goal; truncation only via TimeLimit.
    # reset_target=False: when agent reaches goal, do not resample a new goal
    # (goal stays the same so we keep getting reward while in range).
    env_kwargs = {
        "continuing_task": True,
        "reset_target": False,
        **kwargs,
    }
    if max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = max_episode_steps
    env = gym.make(env_name, **env_kwargs)
    if seed is not None:
        env.reset(seed=seed)
    return env
