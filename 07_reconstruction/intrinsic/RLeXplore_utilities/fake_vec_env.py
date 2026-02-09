"""
Build a VectorEnv for rllte RND: use stable_baselines3 DummyVecEnv over a state-only
wrapper of the real env, so RND gets Box observation space and can run init_normalization()
when obs_norm_type="rms".
"""
import gymnasium as gym
import numpy as np
from typing import Any

from stable_baselines3.common.vec_env import DummyVecEnv


class StateOnlyWrapper(gym.ObservationWrapper):
    """
    Wraps an env with Dict observation (key "observation" = state) and exposes only
    that state as a Box observation. So observation_space becomes the state Box and
    reset/step return only the state vector.

    Input (underlying env): observation_space = Dict(observation=Box(...), ...).
    Output (this wrapper): observation_space = Box(...), obs = obs["observation"].
    """

    def __init__(self, env: gym.Env, key: str = "observation"):
        super().__init__(env)
        if not hasattr(env.observation_space, "spaces") or key not in env.observation_space.spaces:
            raise ValueError(
                f"StateOnlyWrapper expects observation_space with key '{key}'. "
                f"Got {type(env.observation_space)}."
            )
        self._key = key
        self.observation_space = env.observation_space.spaces[key]

    def observation(self, obs: Any) -> np.ndarray:
        return np.asarray(obs[self._key], dtype=np.float32)


def make_fake_vec_env_for_rnd(
    observation_space: Any,
    action_space: Any,
    env: Any,
    num_envs: int = 1,
) -> Any:
    """
    Input format:
        - observation_space: gymnasium.Space, typically a Dict with key "observation" (Box
          for state vector) as produced by e.g. RemoveGoalWrapper on PointMaze.
        - action_space: gymnasium.Space (e.g. Box for continuous actions).
        - env: gymnasium.Env with reset() and step() and Dict observation containing
          "observation". The returned VecEnv runs this env (wrapped to expose state only).
        - num_envs: integer, number of envs (usually 1).

    Output format:
        stable_baselines3.common.vec_env.DummyVecEnv wrapping StateOnlyWrapper(env).
        Real VecEnv with observation_space = Box (state), reset() and step() from SB3,
        so RND init_normalization() can run when obs_norm_type="rms".

    Why we need it:
        RND(envs=...) expects a VectorEnv to derive obs_shape. For Dict observation
        spaces we must expose only the state Box. Using SB3's DummyVecEnv gives a
        proper vectorized env; StateOnlyWrapper makes the underlying env return
        state-only so the VecEnv's observation_space is Box, and init_normalization()
        can run when obs_norm_type="rms".
    """
    def make_state_only():
        return StateOnlyWrapper(env)

    vec_env = DummyVecEnv([make_state_only])
    if not hasattr(vec_env, "single_observation_space"):
        vec_env.single_observation_space = vec_env.observation_space
    return vec_env
