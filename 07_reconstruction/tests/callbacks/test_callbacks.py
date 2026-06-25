"""Unit tests for rnd_exploration.callbacks (train_episode_stats, wandb_eval_logging, distance_logging).

These tests construct each callback without a SAC model and exercise the model-free code
paths directly (attribute wiring, the episode-count selection expression, reward
accumulation/windowing, the Monitor lookup, and the distance-logging dispatch). They never
run a real training loop: fake one-step vec-envs and a stub model stand in for SB3.
"""
import types

import numpy as np
import pytest

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from rnd_exploration.callbacks import (
    TrainEpisodeStatsCallback,
    WandbEvalLoggingCallback,
    DistanceLoggingCallback,
)
from rnd_exploration.callbacks import train_episode_stats as tes_mod
from rnd_exploration.callbacks import wandb_eval_logging as wel_mod
from rnd_exploration.callbacks import distance_logging as dl_mod


# ----------------------------------------------------------------------------------------
# Test doubles: a one-step vec-env and a constant-action model standing in for SB3 objects.
# ----------------------------------------------------------------------------------------
class _OneStepVecEnv:
    """Minimal SB3-style vec-env whose every episode terminates after a single step."""

    def __init__(self, obs_dim: int = 2):
        # one fixed observation vector reused for reset and every step
        self.obs = np.zeros(obs_dim, dtype=np.float32)

    def reset(self):
        """Return the fixed observation (VecEnv reset returns obs only, not a tuple)."""
        return self.obs

    def step(self, action):
        """Return (obs, rewards, dones, infos): reward 1.0, done immediately, empty info."""
        return self.obs, np.array([1.0]), np.array([True]), [{}]


class _ConstantModel:
    """Minimal model: predict always returns a fixed zero action and no state."""

    def predict(self, obs, deterministic=True):
        """Return a constant action and None state, ignoring the observation."""
        return np.zeros(1, dtype=np.float32), None


class _TinyGymEnv(gym.Env):
    """Tiny gymnasium env used only to give Monitor a real wrapped env to wrap."""

    def __init__(self):
        # smallest valid continuous spaces so Monitor can wrap without extra machinery
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        """Return a zero observation and empty info dict."""
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        """Return a zero-observation, zero-reward, terminated transition."""
        return np.zeros(2, dtype=np.float32), 0.0, True, False, {}


# ========================================================================================
# TrainEpisodeStatsCallback
# ========================================================================================
def test_train_episode_stats_construction():
    """Golden: documented kwargs land on attributes; edge: beta defaults to 0.0 when omitted."""
    # golden path: every documented kwarg is supplied and exposed as an attribute
    env = object()
    cb = TrainEpisodeStatsCallback(train_env=env, eval_freq=500, n_eval_episodes=7, use_wandb=True, beta=0.25)
    assert cb.train_env is env
    assert cb.eval_freq == 500
    assert cb.n_eval_episodes == 7
    assert cb.use_wandb is True
    assert cb.beta == 0.25
    # internal accumulators start empty/zero
    assert cb._monitor is None
    assert cb._ep_extrinsic == 0.0 and cb._ep_intrinsic == 0.0
    assert cb._episode_extrinsics == [] and cb._episode_intrinsics == []

    # edge: beta omitted -> documented default of 0.0
    cb_default = TrainEpisodeStatsCallback(train_env=env, eval_freq=1, n_eval_episodes=1, use_wandb=False)
    assert cb_default.beta == 0.0


def test_train_episode_stats_get_monitor_found_and_cached():
    """Golden: a Monitor in the env chain is located; edge: the lookup result is cached."""
    # golden path: vec-env whose first sub-env is a real Monitor wrapping a tiny gym env
    monitor = Monitor(_TinyGymEnv())
    vec_env = types.SimpleNamespace(envs=[monitor])
    cb = TrainEpisodeStatsCallback(train_env=vec_env, eval_freq=10, n_eval_episodes=3, use_wandb=False)
    found = cb._get_monitor()
    assert found is monitor
    assert cb._monitor is monitor

    # edge: a second call returns the cached object even if the env list is swapped out
    vec_env.envs = [object()]
    assert cb._get_monitor() is monitor


def test_train_episode_stats_get_monitor_missing_raises():
    """Edge: when no Monitor is present in the env chain a RuntimeError is raised."""
    # vec-env whose sub-env is a plain object (no `.env` chain, not a Monitor)
    vec_env = types.SimpleNamespace(envs=[object()])
    cb = TrainEpisodeStatsCallback(train_env=vec_env, eval_freq=10, n_eval_episodes=3, use_wandb=False)
    with pytest.raises(RuntimeError):
        cb._get_monitor()


def test_train_episode_stats_accumulates_and_weights_intrinsic():
    """Golden: extrinsic + beta*intrinsic accumulate per step; edge: a done flushes and resets."""
    # eval_freq=0 short-circuits the logging block so we isolate the accumulation logic
    cb = TrainEpisodeStatsCallback(train_env=object(), eval_freq=0, n_eval_episodes=5, use_wandb=False, beta=0.5)

    # golden path: a non-terminal step adds extrinsic (2.0) and beta-weighted intrinsic (0.5*4.0)
    cb.locals = {"infos": [{"extrinsic_reward": 2.0, "intrinsic_reward": 4.0}], "dones": [False]}
    assert cb._on_step() is True
    assert cb._ep_extrinsic == 2.0
    assert cb._ep_intrinsic == 2.0
    assert cb._episode_extrinsics == []

    # edge: a terminal step accumulates again then flushes the episode totals and resets
    cb.locals = {"infos": [{"extrinsic_reward": 2.0, "intrinsic_reward": 4.0}], "dones": [True]}
    assert cb._on_step() is True
    assert cb._episode_extrinsics == [4.0]
    assert cb._episode_intrinsics == [4.0]
    assert cb._ep_extrinsic == 0.0 and cb._ep_intrinsic == 0.0


def test_train_episode_stats_logs_windowed_means(monkeypatch):
    """Golden: logs means over the last n_eval_episodes; edge: no completed episodes -> no log."""
    # capture what would be logged instead of printing/sending to wandb
    captured = {}
    monkeypatch.setattr(tes_mod, "print_or_wandb_log", lambda use_wandb, summary, message: captured.update(summary=dict(summary)))

    # monitor with three completed episodes recorded; n_eval_episodes=2 keeps only the last two
    monitor = Monitor(_TinyGymEnv())
    monitor.episode_returns = [10.0, 20.0, 30.0]
    monitor.episode_lengths = [5, 7, 9]
    vec_env = types.SimpleNamespace(envs=[monitor])
    cb = TrainEpisodeStatsCallback(train_env=vec_env, eval_freq=4, n_eval_episodes=2, use_wandb=False)
    cb._episode_extrinsics = [1.0, 2.0, 3.0]
    cb._episode_intrinsics = [0.1, 0.2, 0.3]
    cb.locals = {}  # empty infos -> skip accumulation, isolate the windowed-logging path
    cb.num_timesteps = 4  # divisible by eval_freq so the logging block runs

    # golden path: each mean is taken over the last two entries only
    assert cb._on_step() is True
    s = captured["summary"]
    assert s["train/mean_total_reward"] == pytest.approx(25.0)
    assert s["train/mean_extrinsic_reward"] == pytest.approx(2.5)
    assert s["train/mean_intrinsic_reward"] == pytest.approx(0.25)
    assert s["train/mean_episode_length"] == pytest.approx(8.0)
    assert s["train/n_episodes_averaged"] == 2

    # edge: with no completed episodes the callback returns early and logs nothing new
    captured.clear()
    monitor.episode_returns = []
    monitor.episode_lengths = []
    assert cb._on_step() is True
    assert captured == {}


# ========================================================================================
# WandbEvalLoggingCallback
# ========================================================================================
def test_wandb_eval_construction_defaults():
    """Golden: documented kwargs are stored; edge: optional kwargs fall back to their defaults."""
    # golden path: full kwarg set, including the final-eval episode count
    eval_env = object()
    cb = WandbEvalLoggingCallback(
        eval_env=eval_env, eval_freq=1000, n_eval_episodes=5, use_wandb=True,
        visit_count_env=object(), goal_cell=(1, 2), start_cell=(0, 0), run_name="run-a",
        beta=0.1, total_timesteps=50000, n_eval_episodes_final=42,
    )
    assert cb.eval_env is eval_env
    assert cb.eval_freq == 1000
    assert cb.n_eval_episodes == 5
    assert cb.use_wandb is True
    assert cb.goal_cell == (1, 2) and cb.start_cell == (0, 0)
    assert cb.run_name == "run-a"
    assert cb.beta == 0.1
    assert cb.total_timesteps == 50000
    assert cb.n_eval_episodes_final == 42

    # edge: omit the optional kwargs and confirm the documented defaults
    cb_default = WandbEvalLoggingCallback(eval_env=eval_env, eval_freq=1, n_eval_episodes=3, use_wandb=False)
    assert cb_default.n_eval_episodes_final == 100
    assert cb_default.total_timesteps is None
    assert cb_default.visit_count_env is None
    assert cb_default.goal_cell is None and cb_default.start_cell is None
    assert cb_default.run_name == ""
    assert cb_default.beta == 0.0


def test_wandb_eval_selects_episode_count(monkeypatch):
    """Golden: final-eval count chosen once num_timesteps>=total; edge: normal count otherwise / when total is None."""
    # capture the logged summary so we can read which episode count drove the eval loop
    captured = {}
    monkeypatch.setattr(wel_mod, "print_or_wandb_log", lambda use_wandb, summary, message: captured.update(summary=dict(summary)))

    def run(total_timesteps, num_timesteps):
        """Build a callback over fake env+model, run one eval step, return the selected episode count."""
        captured.clear()
        cb = WandbEvalLoggingCallback(
            eval_env=_OneStepVecEnv(), eval_freq=1, n_eval_episodes=2, use_wandb=False,
            total_timesteps=total_timesteps, n_eval_episodes_final=5,
        )
        cb.model = _ConstantModel()
        cb.num_timesteps = num_timesteps  # eval_freq=1 makes any value divisible -> loop runs
        assert cb._on_step() is True
        return captured["summary"]["eval/n_eval_episodes"]

    # golden path: training has reached its budget -> final-eval count (5) is used
    assert run(total_timesteps=10, num_timesteps=10) == 5
    # edge: still below budget -> the regular eval count (2) is used
    assert run(total_timesteps=10, num_timesteps=4) == 2
    # edge: no total configured -> the regular eval count (2) is used regardless of step
    assert run(total_timesteps=None, num_timesteps=10) == 2


def test_wandb_eval_skips_when_not_due():
    """Golden: non-positive eval_freq returns early; edge: step not divisible by eval_freq also returns early."""
    # no model is attached, so any attempt to actually evaluate would raise -> early return is verified by success
    cb = WandbEvalLoggingCallback(eval_env=_OneStepVecEnv(), eval_freq=0, n_eval_episodes=2, use_wandb=False)
    cb.num_timesteps = 100
    assert cb._on_step() is True  # golden path: eval_freq <= 0 short-circuits

    # edge: positive eval_freq but the step count is not a multiple of it
    cb2 = WandbEvalLoggingCallback(eval_env=_OneStepVecEnv(), eval_freq=1000, n_eval_episodes=2, use_wandb=False)
    cb2.num_timesteps = 1234
    assert cb2._on_step() is True


# ========================================================================================
# DistanceLoggingCallback
# ========================================================================================
def test_distance_logging_construction():
    """Golden: documented kwargs are stored; edge: use_wandb=True without an active run does not error."""
    # golden path: all documented kwargs supplied; with no active wandb run define_metric is skipped
    model = object()
    vc_env = object()
    cb = DistanceLoggingCallback(
        algorithm="rnd", intrinsic_reward_model=model, visit_count_env_position_velocity=vc_env,
        eval_freq=2000, use_wandb=False,
    )
    assert cb.algorithm == "rnd"
    assert cb.intrinsic_reward_model is model
    assert cb.visit_count_env_position_velocity is vc_env
    assert cb.eval_freq == 2000
    assert cb.use_wandb is False

    # edge: use_wandb=True while no wandb run is active must construct without raising
    cb_wandb = DistanceLoggingCallback(
        algorithm="gt_position", intrinsic_reward_model=model, visit_count_env_position_velocity=vc_env,
        eval_freq=1, use_wandb=True,
    )
    assert cb_wandb.use_wandb is True


def test_distance_logging_on_step(monkeypatch):
    """Golden: at an eval step a metrics dict is logged with the step appended; edge: None metrics -> no log."""
    # capture the logged metrics and stub out the (model-dependent) distance computation
    captured = {}
    monkeypatch.setattr(dl_mod, "print_or_wandb_log", lambda use_wandb, metrics, message: captured.update(metrics=dict(metrics)))

    vc_env = types.SimpleNamespace(maze_map=np.zeros((3, 3)))
    cb = DistanceLoggingCallback(
        algorithm="rnd", intrinsic_reward_model=object(), visit_count_env_position_velocity=vc_env,
        eval_freq=5, use_wandb=False,
    )
    cb.num_timesteps = 5  # divisible by eval_freq so the distance block runs

    # golden path: a returned metrics dict is logged with the current step appended
    monkeypatch.setattr(dl_mod, "compute_intrinsic_vector_distance", lambda *a, **k: {"distance_to_gt/l2": 1.5})
    assert cb._on_step() is True
    assert captured["metrics"]["distance_to_gt/l2"] == 1.5
    assert captured["metrics"]["step"] == 5

    # edge: when the computation returns None nothing is logged
    captured.clear()
    monkeypatch.setattr(dl_mod, "compute_intrinsic_vector_distance", lambda *a, **k: None)
    assert cb._on_step() is True
    assert captured == {}
