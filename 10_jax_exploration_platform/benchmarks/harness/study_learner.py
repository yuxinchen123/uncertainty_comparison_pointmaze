"""Shared helpers of the PointMaze speed measurements: the study's learner, timers, records.

The learner is read from the study's example configuration file, so a measurement times the
program the study trains: clipped PPO, 4 epochs x 4 minibatches, 128-step rollouts, random network
distillation, the episode log and the coverage map on. Only the number of policies, the number of
environments per policy and the keys a measurement names are changed.
"""
import hashlib
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "scripts"))

STUDY_CONFIGURATION = (BASE / "configs/experiments" /
                       "pointmaze_large_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one"
                       "_example_steps-1024000.toml")


def study_configuration(copies: int, envs_per_copy: int, **overrides):
    """The study's learner at one rollout shape, as (PPOConfig, EnvConfig).

    Minibatches are gathered one epoch at a time from 128 environments per policy, as the study
    runs that shape. `overrides` replace trainer fields by name.

    before: copies 256, envs_per_copy 16, overrides {"batch_stats_f32": True}
    after:  the file's trainer with n_copies 256, n_envs 16, batch_stats_f32 True
    """
    from experiment_configuration import read_experiment_configuration
    from exploration_platform.agents.ppo.config import PPOConfig
    from exploration_platform.envs.pointmaze.pm_common import EnvConfig
    document = read_experiment_configuration(STUDY_CONFIGURATION)
    trainer = {**document["trainer"], "n_copies": copies, "n_envs": envs_per_copy,
               "gather_minibatches_per_epoch": envs_per_copy >= 128, **overrides}
    return PPOConfig(**trainer), EnvConfig(**document["environment"])


def call_seconds(fn, reps: int = 30, warmup: int = 3) -> list:
    """Seconds of each of `reps` calls of fn(), waiting for the card after every call.

    For functions that do not consume their arguments. The warm-up calls are not timed.
    """
    import jax
    for _ in range(warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - start)
    return times


def block_seconds(step, state, iterations: int):
    """Seconds per call over `iterations` chained calls with one wait at the end.

    `step` consumes its state and returns the next one, as the training loop calls it: the calls
    are sent without waiting for the card, so this is the rate a training run gets.
    Returns (seconds per call, the last state).
    """
    import jax
    start = time.perf_counter()
    for _ in range(iterations):
        state = step(state)
    jax.block_until_ready(state)
    return (time.perf_counter() - start) / iterations, state


def median(values: list) -> float:
    """The middle value of a list (the upper of the two middle values for an even count)."""
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def throughput_columns(copies: int, envs_per_copy: int, rollout_steps: int,
                       seconds_per_iteration: float) -> dict:
    """The five throughput columns of one setting.

    before: 256 policies x 16 environments x 128 steps, 0.050 s per iteration
    after:  total 10.49 million steps per second, 40.96 thousand per policy,
            0.0068 hours per million steps per policy
    """
    per_copy = envs_per_copy * rollout_steps / seconds_per_iteration
    return {"copies": copies, "seconds_per_iteration": seconds_per_iteration,
            "total_steps_per_second_millions": copies * per_copy / 1e6,
            "steps_per_second_per_copy_thousands": per_copy / 1e3,
            "hours_per_million_steps_per_copy": 1e6 / (3600.0 * per_copy)}


def source_digest() -> str:
    """SHA-256 over the platform's Python source files, in path order, so a record names its code."""
    digest = hashlib.sha256()
    for path in sorted((BASE / "src/exploration_platform").rglob("*.py")):
        digest.update(str(path.relative_to(BASE)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def machine_record() -> dict:
    """What a measurement ran on: host, card, driver, job, library version, code."""
    import jax
    query = ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"]
    smi = (subprocess.run(query, capture_output=True, text=True).stdout.strip()
           if shutil.which("nvidia-smi") else "no nvidia-smi on this host")
    return {"host": socket.gethostname(), "card": smi, "jax": jax.__version__,
            "device": str(jax.devices()[0]),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "git_commit": subprocess.run(["git", "-C", str(BASE), "rev-parse", "--short", "HEAD"],
                                         capture_output=True, text=True).stdout.strip(),
            "source_sha256": source_digest(),
            "started_pacific": subprocess.run(
                ["date", "+%Y-%m-%d %H:%M:%S PT"], capture_output=True, text=True,
                env={**os.environ, "TZ": "America/Los_Angeles"}).stdout.strip()}
