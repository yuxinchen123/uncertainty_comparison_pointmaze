"""Shared pieces of the RK4 AntMaze iteration profile: build the trainer as the training command
builds it, count the graphs Warp captures, time chained calls, and write records.

The trainer comes from the configuration file through `device/series.py` and
`scripts/run_training.py`, exactly as `diagnosis/2026-11-13_ant-integrators-in-mujoco-warp/scripts/
profile_integrators.py` of the AntMaze run folder builds it, so the program timed is the program a
unit runs (the file's 50 priming rollouts are cut to 2: priming only sets the input statistics).
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PLATFORM = Path("/p/rlprojects/RND/10_jax_exploration_platform")
for folder in ("device", "scripts", "src", "benchmarks/harness"):
    sys.path.insert(0, str(PLATFORM / folder))

CONFIGURATION = PLATFORM / "configs/experiments/antmaze_umaze_warp-physics_rk4_time-step-0.01_frame-skip-5_fault-guard-30-metres-and-100-radians-per-second_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-204800.toml"
ENVIRONMENTS_PER_POLICY = 16
ANNEAL_ITERATIONS = 10 ** 6  # a learning-rate horizon so long that the rate does not move here


def pacific_now() -> str:
    """The time now as a Pacific timestamp with its zone named."""
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S PT")


def say(text: str) -> None:
    """One flushed progress line with a Pacific time stamp."""
    print(f"[{pacific_now()}] {text}", flush=True)


def unit_configuration(copies: int, environments_per_policy: int = ENVIRONMENTS_PER_POLICY,
                       prime_iterations: int = 2):
    """(trainer arguments, PPOConfig, environment config) of the unit at one shape, as the
    doubling series builds its command line from the configuration file."""
    import series
    import run_training
    arguments = argparse.Namespace(configuration_file=CONFIGURATION, environment_family="antmaze",
                                   out_dir=Path("/tmp"), environments_per_policy=environments_per_policy,
                                   update_style="epoch_minibatch", bonus="rnd_next_state",
                                   trainer_argument=[f"--prime-iterations={prime_iterations}"])
    trainer_arguments = run_training.parse_args(series.trainer_command_line(arguments, copies))
    return (trainer_arguments, run_training.build_config(trainer_arguments),
            run_training.build_env_config(trainer_arguments))


def build_trainer(copies: int, environments_per_policy: int = ENVIRONMENTS_PER_POLICY,
                  prime_iterations: int = 2, env_overrides: dict = None, config_overrides: dict = None):
    """The unit's trainer at one shape, primed; returns (runner, state, environment config).

    env_overrides and config_overrides replace [environment] and [trainer] fields by name (for
    measurements only).
    before: 128 copies -> Runner with cfg.n_copies 128, cfg.n_envs 16, RK4 Warp physics
    """
    from dataclasses import replace
    from exploration_platform.training.runner import Runner

    # GOAL: the unit's configuration at this shape, with the measurement's replacements
    trainer_arguments, config, environment = unit_configuration(copies, environments_per_policy,
                                                                prime_iterations)
    if config_overrides:
        config = replace(config, **config_overrides)
    if env_overrides:
        environment = replace(environment, **env_overrides)
    if environment.physics != "warp" or environment.integrator != "RK4":
        raise ValueError(f"the file builds {environment.physics!r} physics with {environment.integrator!r}")
    runner = Runner(config, bonus=trainer_arguments.bonus, env_cfg=environment)
    state = runner.prime(runner.init_state(run_seed=0))
    return runner, state, environment


def lr_of(runner, iteration: int):
    """The learning-rate argument of one iteration under a horizon that does not move it."""
    return runner.lr_argument(iteration, ANNEAL_ITERATIONS)


class CaptureCounter:
    """Counts the CUDA graphs Warp captures inside the JAX foreign calls (graph mode WARP).

    A capture happens the first time a call site sees a new set of buffer addresses; one in the
    timed rounds means a graph was captured again during training, which costs far more than a
    replay. The count is read from the per-callable caches of the vendored Warp FFI.
    """

    def __init__(self):
        """Wrap ScopedCapture.__enter__ of the vendored warp so every capture is counted."""
        import warp as wp
        from mujoco.mjx.third_party.warp._src.jax import ffi as warp_ffi
        self.count = 0
        counter = self
        original = wp.ScopedCapture.__enter__

        def counted_enter(scoped, *args, **kwargs):
            """Count one capture, then enter it as Warp does."""
            counter.count += 1
            return original(scoped, *args, **kwargs)

        wp.ScopedCapture.__enter__ = counted_enter
        self.module = warp_ffi


def chained_seconds(step, state, calls: int):
    """Seconds per call over `calls` chained calls with one wait at the end; (seconds, state)."""
    import jax
    start = time.perf_counter()
    for _ in range(calls):
        state = step(state)
    jax.block_until_ready(state)
    return (time.perf_counter() - start) / calls, state


def throughput(copies: int, environments: int, rollout_steps: int, seconds: float) -> dict:
    """The five throughput columns of one setting (totals in millions, per copy in thousands)."""
    per_copy = environments * rollout_steps / seconds
    return {"copies": copies, "seconds_per_iteration": seconds,
            "total_steps_per_second_millions": copies * per_copy / 1e6,
            "steps_per_second_per_copy_thousands": per_copy / 1e3,
            "hours_per_million_steps_per_copy": 1e6 / (3600.0 * per_copy)}


def machine() -> dict:
    """Host, card, job, library versions and the code's commit, for every record."""
    import jax
    import mujoco
    import warp
    card = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip() if os.path.exists("/usr/bin/nvidia-smi") else ""
    commit = subprocess.run(["git", "-C", str(PLATFORM), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    return {"host": socket.gethostname(), "card": card, "device": str(jax.devices()[0]),
            "jax": jax.__version__, "mujoco": mujoco.__version__, "warp": warp.config.version,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""), "git_commit": commit,
            "written_at": datetime.now().astimezone().isoformat(timespec="seconds")}


def write_record(path: Path, record: dict) -> None:
    """Write the record through a temporary file, so a reader never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=1))
    temporary.replace(path)


def median(values) -> float:
    """The median of a list of numbers."""
    return statistics.median(values)
