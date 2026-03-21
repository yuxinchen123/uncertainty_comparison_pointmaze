"""
Test the impact of TerminateOnTimeLimitWrapper on SB3's done/timeout handling.

CRITICAL FINDING: TerminateOnTimeLimitWrapper breaks SB3's timeout correction.

In the old code (a558f7c), there is NO TerminateOnTimeLimitWrapper.
When max_episode_steps is reached with continuing_task=True:
  - env returns: terminated=False, truncated=True
  - SB3 DummyVecEnv sets info["TimeLimit.truncated"] = True
  - SB3 ReplayBuffer records timeout, so done is corrected to False
  - SAC critic bootstraps: target_Q = r + gamma * V(s')

In the new code (2ff924b), TerminateOnTimeLimitWrapper is added:
  - env returns: terminated=False, truncated=True
  - TerminateOnTimeLimitWrapper converts to: terminated=True, truncated=False
  - SB3 DummyVecEnv does NOT set info["TimeLimit.truncated"]
  - SB3 ReplayBuffer records done=True, no timeout correction
  - SAC critic does NOT bootstrap: target_Q = r  (value = 0 after episode end)

This means SAC thinks every episode truly ends, losing all future value.
With gamma=0.999 and max_episode_steps=400: gamma^400 ≈ 0.67, so 33% of
theoretical future value is lost at the boundary.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import numpy as np
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from env_wrapper.point_maze_wrappers import TerminateOnTimeLimitWrapper


def test_without_terminate_wrapper():
    """Old code path: no TerminateOnTimeLimitWrapper."""
    gym.register_envs(gymnasium_robotics)
    env = gym.make("PointMaze_Large-v3", continuing_task=True, reset_target=False, max_episode_steps=5)

    mon = Monitor(env, filename=None)
    vec_env = DummyVecEnv([lambda: mon])
    vec_env.seed(0)
    obs = vec_env.reset()

    for step in range(10):
        action = vec_env.action_space.sample()
        obs, rewards, dones, infos = vec_env.step([action])
        info = infos[0]
        tl_truncated = info.get("TimeLimit.truncated", False)
        terminal_obs = "terminal_observation" in info
        print(f"Step {step+1}: done={dones[0]}, TimeLimit.truncated={tl_truncated}, has_terminal_obs={terminal_obs}")
        if dones[0]:
            print(f"  -> Episode ended. SB3 sees TimeLimit.truncated={tl_truncated}")
            if tl_truncated:
                print(f"  -> SB3 will CORRECT done to False (bootstrap past timeout)")
            else:
                print(f"  -> SB3 treats as TRUE termination (no bootstrap)")
            break

    vec_env.close()


def test_with_terminate_wrapper():
    """New code path: WITH TerminateOnTimeLimitWrapper."""
    gym.register_envs(gymnasium_robotics)
    env = gym.make("PointMaze_Large-v3", continuing_task=True, reset_target=False, max_episode_steps=5)
    env = TerminateOnTimeLimitWrapper(env)

    mon = Monitor(env, filename=None)
    vec_env = DummyVecEnv([lambda: mon])
    vec_env.seed(0)
    obs = vec_env.reset()

    for step in range(10):
        action = vec_env.action_space.sample()
        obs, rewards, dones, infos = vec_env.step([action])
        info = infos[0]
        tl_truncated = info.get("TimeLimit.truncated", False)
        terminal_obs = "terminal_observation" in info
        print(f"Step {step+1}: done={dones[0]}, TimeLimit.truncated={tl_truncated}, has_terminal_obs={terminal_obs}")
        if dones[0]:
            print(f"  -> Episode ended. SB3 sees TimeLimit.truncated={tl_truncated}")
            if tl_truncated:
                print(f"  -> SB3 will CORRECT done to False (bootstrap past timeout)")
            else:
                print(f"  -> SB3 treats as TRUE termination (no bootstrap)")
            break

    vec_env.close()


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: WITHOUT TerminateOnTimeLimitWrapper (old code)")
    print("=" * 60)
    test_without_terminate_wrapper()

    print()
    print("=" * 60)
    print("TEST 2: WITH TerminateOnTimeLimitWrapper (new code)")
    print("=" * 60)
    test_with_terminate_wrapper()

    print()
    print("=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("With TerminateOnTimeLimitWrapper, SB3 treats every timeout as a")
    print("true termination. SAC's critic sets target_Q = reward (no bootstrap).")
    print("This fundamentally changes the MDP for continuing_task=True envs.")
    print()
    print("FIX: Remove TerminateOnTimeLimitWrapper from 04_many_exploration_method.py")
    print("     on lines where base_env and eval_base are created:")
    print("     DELETE: base_env = TerminateOnTimeLimitWrapper(base_env)")
    print("     DELETE: eval_base = TerminateOnTimeLimitWrapper(eval_base)")
