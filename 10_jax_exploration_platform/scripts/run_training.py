"""Run one training work unit and write its shard into the platform's standard run folder.

One call = one work unit. A run folder may hold many units (that is what a sweep is); this script
writes one shard, `data/<unit_id>.jsonl`, one line per record, flushed as it goes, and then calls
the aggregator so the run-level `metrics.jsonl` and `summary.json` exist even for a single unit.

A unit may itself be a sweep across copy groups: `--learning-rates` and `--intrinsic-weights` take
the cross product, `--copies-per-cell` says how many copies each cell gets, and the copies of one
cell are seeded in pairs with the copies of every other cell, so a difference between two cells is
the swept values' doing.

Records are PHASE-BLOCKED EPISODE WINDOWS. The task is continuing, so an episode ends only at the
400-step truncation and every copy shares one episode clock; an iteration collects 128 of those 400
steps, so where inside the episode an iteration looks — its `episode_phase`, `((iteration - 1) *
rollout_steps) % episode_steps` — cycles with period 25 iterations. One iteration's extrinsic reward
is therefore a sample of one 128-step window of the episode and can read zero for every copy while
copies are solving. A record here covers a WINDOW of iterations spanning a whole number of those
25-iteration cycles, and reports the reward SUMMED over the window per copy, so the episode clock
cancels out and the whole run's episode returns are recomputable exactly from about a hundred
records instead of nineteen thousand. Each record still carries its own `episode_phase` values, per
the learning-outcome campaign's convention.

The launch-time files (`manifest.yaml`, `config_resolved.yaml`, `command.txt`) are written before
the first iteration, per the `experiment-background` skill. `experiment_background.md` is written by
hand at the same moment and is not this script's job.

No model state is saved by default. Resuming is then per UNIT: a unit whose shard already ends in a
`unit_complete` record is skipped, so re-running the same command continues a job at the first
unfinished unit and never repeats finished work, and an unfinished unit starts again from zero.

With `--checkpoint-seconds` (or `--time-limit-seconds`) the unit's whole state is saved to
`checkpoints/<unit_id>.npz` at window ends and when the process is told to stop. Re-running the
same command then continues the unit from that iteration: window records later than the checkpoint
are dropped from the shard and written again, and the results equal an uninterrupted unit's.

Episodes that end at the goal (`--terminate-at-goal`, or a named `--reward-setting`) have no shared
episode clock, so such a unit needs `--episode-log`: each window record then carries, per copy, the
episodes that ended in the window, the arrivals, and the sums of their returns and lengths.

AntMaze runs its physics through MJX's JAX implementation by default. `--physics warp` (or
`physics = "warp"` in a configuration file's `[environment]`) runs it through MuJoCo Warp, which
needs the interpreter /p/rlprojects/RND/.venvs/platform_jax_warp/bin/python. Such a unit names the
physics in its environment specification, writes MuJoCo Warp's kernel folder into its log, records
the largest use of the shared contact pool in every window record and in `unit_complete`, and
stops with a message when MuJoCo Warp dropped a contact or a constraint row.

A bonus family's own options (the fields of its configuration class, `bonuses/rnd/config.py` for
random network distillation) are changed with `--bonus-options name=value,...` or a configuration
file's optional `[bonus_options]` table; they hold for the whole process, never per copy. The
`unit_start` record holds every option with its value under `bonus_options`. A reward divisor
held after a fraction of the run (`reward_divisor=held_after_fraction`) counts that fraction of
the iterations the learning rate anneals over (`--lr-anneal-iterations`, or `--iterations`). A
reward divisor held after the first arrival (`reward_divisor=held_after_first_arrival`) is held
per copy from the rollout after the one in which the copy first reaches the goal; it needs
`--episode-log` and an environment whose episodes end at the goal. A fast last layer that stops
at the first arrival (`fast_last_layer_step=0.1,fast_last_layer_until_first_arrival=true`) scores
a copy's rollouts with the fast last layer up to and including the one in which the copy first
reaches the goal, and every later rollout by the predictor alone; it has the same two needs. Add
`predictor_divisor_after_first_arrival=true` so that a copy that has arrived is divided by a
second statistic kept on the predictor's own scores (the divisor random network distillation
would have), not by the one built on the fast last layer's scores, which is many times smaller.
With
`record_bonus_on_maze_grid=true` each window record also carries, per copy, the bonus at one probe
observation per open maze cell, the divisor of the intrinsic reward, and the window's visits to
each cell (`training/maze_grid.py`); one `maze_grid_at_start` record holds the same fields before
the first training iteration.

Example:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python run_training.py \
    --run-dir <runs/...> --unit-id unit-1 --bonus rnd_next_state \
    --learning-rates 1e-3,1e-4,1e-5 --intrinsic-weights 1e-5,1e-4 --copies-per-cell 256 \
    --iterations 19531 --phase-iterations 200 --track-coverage
"""
import argparse
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PLATFORM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLATFORM_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from experiment_configuration import (DEFAULT_SETTING_NAMES, REWARD_SETTING_NAMES,
                                      REWARD_SETTINGS, parse_with_configuration_file,
                                      reward_mappings)

PACIFIC = ZoneInfo("America/Los_Angeles")

# identities of the pieces this run is made of; they go into the manifest so a later change to any
# of them is visible in the record rather than silent
# one row per environment this script can run: the specification name a manifest carries, and
# the environment-configuration factory the composer dispatches on (None = the PointMaze default)
ENVIRONMENTS = {
    "pointmaze-large": {"spec": "pointmaze_large_cont400_nonoise@1", "cfg": None,
                        "episode_steps": 400,
                        "resolved": {"map_name": "large", "start_cell": "(7, 1)",
                                     "goal_cell": "(1, 10)", "position_noise": 0.0,
                                     "continuing_task": True, "goal_radius": 0.45,
                                     "reward_shift": 0.0}},
    "antmaze-umaze": {"spec": "antmaze_umaze_cont700_nonoise@1", "cfg": "umaze",
                      "episode_steps": 700,
                      "resolved": {"map_name": "umaze", "start_cell": "(3, 1)",
                                   "goal_cell": "(1, 1)", "continuing_task": True,
                                   "goal_radius": 0.45, "reward_shift": 0.0}},
    "antmaze-medium": {"spec": "antmaze_medium_cont1000_nonoise@1", "cfg": "medium",
                       "episode_steps": 1000,
                       "resolved": {"map_name": "medium", "start_cell": "(6, 1)",
                                    "goal_cell": "(1, 6)", "continuing_task": True,
                                    "goal_radius": 0.45, "reward_shift": 0.0}},
    "antmaze-large": {"spec": "antmaze_large_cont1000_nonoise@1", "cfg": "large",
                      "episode_steps": 1000,
                      "resolved": {"map_name": "large", "start_cell": "(7, 1)",
                                   "goal_cell": "(1, 10)", "continuing_task": True,
                                   "goal_radius": 0.45, "reward_shift": 0.0}},
    "montezuma": {"spec": "montezuma_oc4500_sticky@1", "cfg": "montezuma",
                  "episode_steps": 4500,
                  "resolved": {"game": "montezumarevenge (JAXAtari)",
                               "sticky_actions": 0.25, "frame_skip": 4, "frame_stack": 4,
                               "clip_reward": True, "reward_shift": 0.0}},
}

AGENT_SPECS = {
    "pointmaze_current": "ppo_full_batch_continuing_intrinsic@3",
    "pointmaze_historical": "ppo_full_batch@1",
    "antmaze_historical": "ppo_full_batch@1",
    "antmaze_current": "ppo_full_batch_continuing_intrinsic@3",
    "montezuma_object_historical": "ppo_full_batch@1",
}

# set by the stop signals when checkpoints are on; the training loop reads it after each iteration
STOP = {"requested": False}
# a unit with the Warp AntMaze physics reads its contact pool at window ends and, between them,
# once this many seconds have passed since the last read (each read waits for the device once)
POOL_CHECK_SECONDS = 60.0


def env_config_of(env_name: str):
    """The environment-configuration object the composer dispatches on; None = PointMaze default."""
    map_name = ENVIRONMENTS[env_name]["cfg"]
    if map_name is None:
        return None
    if map_name == "montezuma":
        from exploration_platform.envs.atari_montezuma.jax_montezuma import MontezumaConfig
        return MontezumaConfig()
    from exploration_platform.envs.antmaze.am_common import preset
    return preset(map_name)

# the update schedule each bonus family follows, for the manifest's record
UPDATE_SCHEDULES = {
    "rnd_next_state": "per_rollout@1",
    "none": "none@1",
    "gt_position_velocity_sqrt": "post_rollout_update@1",
    "gt_position_velocity_linear": "post_rollout_update@1",
}


def pacific_now() -> str:
    """The current time as the reader sees it: Pacific, with the zone named."""
    return datetime.now().astimezone(PACIFIC).strftime("%Y-%m-%d %H:%M PT")


def as_yaml(value, indent: int = 0) -> str:
    """Write a dictionary of scalars, lists and dictionaries as YAML text.

    before: {"a": 1, "b": {"c": [2, 3]}} ; after:
      a: 1
      b:
        c:
          - 2
          - 3
    """
    pad = "  " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(as_yaml(item, indent + 1))
            else:
                lines.append(f"{pad}{key}: {scalar_yaml(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {scalar_yaml(item)}" for item in value)
    return f"{pad}{scalar_yaml(value)}"


def scalar_yaml(value) -> str:
    """One scalar in YAML form: booleans lower case, empty stays empty, strings quoted if needed."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    needs_quotes = text == "" or text[0] in "&*?|-<>=!%@`{[" or ": " in text or text.endswith(":")
    return f'"{text}"' if needs_quotes else text


def git_state(repo_root: Path) -> dict:
    """The commit this run's code is at, and whether anything was uncommitted at launch."""
    # a worker submitted under a collaborator's account runs git on a repository it does not own,
    # which git refuses unless a global config names the directory safe (git 2.34 on the compute
    # nodes ignores the setting on the command line), so these calls read a config that does
    git = ["git", "-C", str(repo_root)]
    env = {**os.environ,
           "GIT_CONFIG_GLOBAL": str(Path(__file__).resolve().parent / "git_safe_directory.gitconfig")}
    run = lambda *a: subprocess.run(git + list(a), capture_output=True, text=True, env=env)
    head = run("rev-parse", "HEAD")
    if head.returncode:
        raise RuntimeError(f"git could not read the repository's commit: {head.stderr.strip()}")
    commit = head.stdout.strip()
    status = run("status", "--porcelain").stdout.strip()
    describe = run("describe", "--tags", "--always").stdout.strip()
    return {"commit": commit, "dirty": bool(status), "version_tag": describe}


def hardware_record() -> dict:
    """What this unit ran on: the machine, the card, the driver, and the library versions."""
    import jax
    devices = jax.devices()
    # a machine with no graphics card has no nvidia-smi at all, so ask before calling it
    query = ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"]
    smi = (subprocess.run(query, capture_output=True, text=True)
           if shutil.which("nvidia-smi") else None)
    return {
        "record": "job",
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "interpreter": sys.executable,
        "devices": [str(d) for d in devices],
        "device_kind": devices[0].device_kind,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "nvidia_smi": (smi.stdout.strip() if smi and smi.returncode == 0
                       else "no nvidia-smi on this host"),
    }


def number_list(text: str) -> tuple:
    """Parse a comma-separated list of numbers; an empty string is an empty list.

    before: "1e-3,1e-4,1e-5" ; after: (0.001, 0.0001, 1e-05)
    before: ""               ; after: ()
    """
    return tuple(float(part) for part in text.split(",") if part.strip())


def parse_args(argv=None) -> argparse.Namespace:
    """Every knob this run exposes; the defaults are the trainer's own."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="the run folder; created if missing")
    # required, and deliberately without a default: a job that loses its arguments must fail rather
    # than write a shard full of the runner's defaults under a plausible-looking name
    parser.add_argument("--unit-id", required=True, help="names this unit's data shard")
    parser.add_argument("--description", default="", help="the human sentence for the manifest")
    parser.add_argument("--configuration-file", type=Path,
                        help="complete environment-specific TOML; explicit CLI options override its settings")
    parser.add_argument("--copies", type=int, default=128,
                        help="copies, when neither knob is swept; a sweep derives the count")
    parser.add_argument("--configuration-preset", default="environment_default",
                        choices=["environment_default", *sorted(AGENT_SPECS)],
                        help="named environment-specific return and no-bonus choices; "
                             "pixel CleanRL Atari uses its separate training command")
    parser.add_argument("--envs-per-copy", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--update-style", default="full_batch",
                        choices=["full_batch", "epoch_minibatch"])
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="the single rate, used when --learning-rates is empty")
    parser.add_argument("--learning-rates", default="",
                        help="comma-separated rates to sweep, e.g. 1e-3,1e-4,1e-5")
    parser.add_argument("--intrinsic-coefficient", type=float, default=1.0,
                        help="the single intrinsic weight, used when --intrinsic-weights is empty")
    parser.add_argument("--intrinsic-weights", default="",
                        help="comma-separated weights on the intrinsic advantage to sweep")
    parser.add_argument("--copies-per-cell", type=int, default=0,
                        help="copies each (rate, weight) cell gets; required for a sweep")
    parser.add_argument("--parameter-settings", type=Path,
                        help="JSON mapping each bonus name to explicit [rate, weight] pairs")
    parser.add_argument("--flat-params", action="store_true")
    parser.add_argument("--scan-unroll", type=int, default=0)
    parser.add_argument("--no-aggregate", action="store_true",
                        help="let the run's central monitor write the aggregate")
    parser.add_argument("--bonus", default="rnd_next_state",
                        help="which intrinsic-reward family; see bonuses/registry.py")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--copy-seed-offset", type=int, default=0,
                        help="shift on every copy's seed index, so one logical run's copies can "
                             "be cut into chunks that run on separate cards; chunk j of k passes "
                             "j x (copies / k) and holds that slice of the copies")
    parser.add_argument("--window-iterations", type=int, default=200,
                        help="iterations per recorded episode window; one record covers one window")
    parser.add_argument("--env", default="pointmaze-large", choices=sorted(ENVIRONMENTS),
                        help="which environment specification this unit trains on")
    parser.add_argument("--episode-steps", type=int, default=0,
                        help="the environment's truncation cap, which sets the episode clock; "
                             "0 takes the chosen environment's own cap")
    parser.add_argument("--track-coverage", action="store_true",
                        help="keep a per-copy visited-cell map on the device")
    # the environment (added 2026-10-19); the defaults are the large-maze task every earlier run used
    parser.add_argument("--map", default="large", choices=["umaze", "open", "medium", "large"])
    parser.add_argument("--start-cell", default="7,1", help="row,column of the start; row 0 on top")
    parser.add_argument("--goal-cell", default="1,10", help="row,column of the goal")
    parser.add_argument("--terminate-at-goal", action="store_true",
                        help="end the episode on the first step inside the goal radius")
    parser.add_argument("--reward-shift", type=float, default=0.0,
                        help="added to every step's task reward (-1: 0 at the goal, -1 elsewhere)")
    parser.add_argument("--goal-reward", default="per_step",
                        choices=["per_step", "remaining_steps"],
                        help="remaining_steps: ending at the goal pays cap - step + 1 at once")
    # the agent knobs the reward-decline study varies; the defaults are PPOConfig's
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--gamma-ext", type=float, default=0.999)
    parser.add_argument("--ext-coef", type=float, default=2.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--diagnostics", action="store_true",
                        help="record policy, critic, advantage and per-episode numbers per window")
    parser.add_argument("--intrinsic-weight-schedule", default="constant",
                        choices=["constant", "linear_decay"])
    parser.add_argument("--success-gate-rate", type=float, default=0.0,
                        help="> 0: intrinsic weight x (1 - running goal-reaching rate)")
    parser.add_argument("--normalize-extrinsic-reward", action="store_true")
    parser.add_argument("--clip-per-network", action="store_true")
    parser.add_argument("--logstd-floor", type=float, default=float("-inf"),
                        help="raise the policy's log standard deviation to at least this")
    parser.add_argument("--critic-lr-scale", type=float, default=1.0,
                        help="the critic's step as a fraction of the step Adam computed")
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--normalize-agent-obs", action="store_true",
                        help="whiten the actor's and critic's input with running statistics")
    parser.add_argument("--no-bootstrap-on-truncation", action="store_true",
                        help="treat the episode cap as the end of the task return")
    parser.add_argument("--lr-anneal-iterations", type=int, default=0,
                        help="iterations over which the learning rate anneals linearly to zero; "
                             "0 = this run's own length. A longer horizon makes a short run use "
                             "the early part of a long run's schedule")
    # the PointMaze and AntMaze study (2026-11-11); every default leaves a run as it was
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="clip coefficient of the epoch-minibatch update (CleanRL: 0.1)")
    parser.add_argument("--prime-iterations", type=int, default=10,
                        help="rollouts of random actions that start the bonus's input statistics "
                             "before training (CleanRL: 50)")
    parser.add_argument("--entropy-coefficients", default="",
                        help="comma-separated entropy coefficients to sweep, one cell each")
    parser.add_argument("--clip-coefficients", default="",
                        help="comma-separated clip coefficients to sweep, one cell each")
    parser.add_argument("--value-coefficients", default="",
                        help="comma-separated value coefficients to sweep, one cell each")
    parser.add_argument("--reward-setting", default="environment", choices=list(REWARD_SETTINGS),
                        help="environment: the environment's own reward. arrival_plus_one: the "
                             "episode ends at the goal, which pays 1. step_cost: the episode "
                             "ends at the goal, every step costs 1 / (step limit), arrival pays "
                             "0. arrival_plus_one_and_step_cost: both in one process, each cell "
                             "once per reward on the same seeds")
    parser.add_argument("--physics", default="jax", choices=["jax", "warp"],
                        help="AntMaze only: the implementation of the physics step. jax: MJX's "
                             "JAX implementation. warp: MuJoCo Warp, which needs the interpreter "
                             "of the platform_jax_warp environment")
    parser.add_argument("--episode-log", action="store_true",
                        help="record per copy and window the episodes that ended, the arrivals, "
                             "and the sums of returns, lengths and steps to the goal")
    parser.add_argument("--gather-minibatches-per-epoch", action="store_true",
                        help="epoch-minibatch updates hold one shuffled copy of the batch at a "
                             "time instead of one per epoch")
    parser.add_argument("--checkpoint-seconds", type=float, default=0.0,
                        help="> 0: save the unit's whole state at the first window end after "
                             "this many seconds since the last save, and when the process is "
                             "told to stop; re-running the command continues from the save")
    parser.add_argument("--time-limit-seconds", type=float, default=0.0,
                        help="> 0: after this many seconds, save the unit's state and stop")
    parser.add_argument("--put-state-on-card", action="store_true",
                        help="before the first iteration and after every window end, put the "
                             "training state on the card as the arrays a compiled iteration "
                             "returns (jax.device_put), so the iteration is compiled once instead "
                             "of again on its second call and after the first window end. Changes "
                             "no number")
    parser.add_argument("--compilation-cache-dir", default="",
                        help="a folder for JAX's persistent compilation cache: a process that "
                             "compiles a program another process on the same card model already "
                             "compiled reads it from there instead of compiling it again. Changes "
                             "no number. Empty (the default): no cache")
    parser.add_argument("--bonus-options", default="",
                        help="options of the bonus family this unit changes, as comma-separated "
                             "name=value pairs, e.g. even_start=true,reward_divisor=none (the "
                             "names are the fields of the family's configuration, "
                             "bonuses/rnd/config.py); a pair replaces the same name of a "
                             "configuration file's [bonus_options] table")
    return parse_with_configuration_file(parser, argv)


def bonus_options_of(args: argparse.Namespace) -> dict:
    """The bonus family's options this unit changes: the configuration file's, then the command's.

    before: file [bonus_options] even_start = true ; --bonus-options reward_divisor=none
    after:  {"even_start": True, "reward_divisor": "none"}
    A value from the command line stays text here; the registry reads it in the option's type.
    """
    document = getattr(args, "experiment_configuration_data", None)
    options = dict(document["bonus_options"]) if document is not None else {}
    for pair in args.bonus_options.split(","):
        if not pair.strip():
            continue
        name, separator, value = (part.strip() for part in pair.partition("="))
        if not (name and separator and value):
            raise ValueError(f"--bonus-options takes name=value pairs; {pair!r} is not one")
        options[name] = value
    return options


def cell(text: str) -> tuple:
    """Parse a row,column cell. before: "6,1"; after: (6, 1)."""
    row, col = (int(part) for part in text.split(","))
    return row, col


def build_env_config(args: argparse.Namespace):
    """The environment this unit trains in, from the command line."""
    from dataclasses import replace
    from exploration_platform.envs.pointmaze.pm_common import EnvConfig, validate_env_config
    episode_steps = args.episode_steps or ENVIRONMENTS[args.env]["episode_steps"]
    document = getattr(args, "experiment_configuration_data", None)
    # a named reward setting is defined on an episode that ends at the goal and pays 1 there
    named_reward = args.reward_setting != "environment"
    if named_reward and (args.reward_shift != 0 or args.goal_reward != "per_step"):
        raise ValueError(f"--reward-setting {args.reward_setting} starts from a reward of 1 on "
                         "arrival and 0 elsewhere: leave --reward-shift at 0 and --goal-reward "
                         "at per_step")
    terminate_at_goal = args.terminate_at_goal or named_reward
    if args.physics != "jax" and not args.env.startswith("antmaze-"):
        raise ValueError(f"--physics {args.physics} is built for AntMaze only; {args.env} has one "
                         "implementation")
    if args.env != "pointmaze-large":
        if (args.map, args.start_cell, args.goal_cell, args.goal_reward) != (
                "large", "7,1", "1,10", "per_step"):
            raise ValueError("PointMaze map and goal options cannot change another environment")
        antmaze = args.env.startswith("antmaze-")
        if terminate_at_goal and not antmaze:
            raise ValueError("ending the episode at the goal is built for PointMaze and AntMaze; "
                             f"{args.env} has no such option")
        environment = env_config_of(args.env)
        if document is not None:
            environment = replace(environment, **document["environment"])
        changes = dict(max_episode_steps=episode_steps, reward_shift=args.reward_shift)
        if antmaze:
            changes.update(continuing_task=not terminate_at_goal, physics=args.physics)
        return replace(environment, **changes)
    # Retain file-only environment settings such as noise and goal radius, then apply CLI values.
    fields = dict(document["environment"]) if document is not None else {}
    fields.update(map_name=args.map, start_cell=cell(args.start_cell),
                  goal_cell=cell(args.goal_cell), max_episode_steps=episode_steps,
                  continuing_task=not terminate_at_goal,
                  reward_shift=args.reward_shift, goal_reward=args.goal_reward)
    env_cfg = EnvConfig(**fields)
    validate_env_config(env_cfg)
    return env_cfg


def env_spec(env_cfg, reward_setting: str = "environment") -> str:
    """The environment's identity string; the original task keeps its original name.

    before: large map, continuing, cap 400, shift 0, per_step -> pointmaze_large_cont400_nonoise@1
    before: medium map, terminating, cap 250, shift -1 -> pointmaze_medium_term250_nonoise_shift-1@1
    before: large map, terminating, cap 400, reward setting step_cost
            -> pointmaze_large_term400_nonoise_reward-step-cost@1
    before: AntMaze U-maze, terminating, cap 700, MuJoCo Warp physics, reward setting step_cost
            -> antmaze_umaze_term700_nonoise_physics-warp_reward-step-cost@1
    (AntMaze with the JAX physics keeps the name it had before the physics could be chosen)
    """
    environment_type = type(env_cfg).__name__
    if environment_type == "MontezumaConfig":
        name = f"montezuma_oc{env_cfg.max_episode_steps}_sticky"
        if env_cfg.reward_shift:
            name += f"_shift{env_cfg.reward_shift:g}"
        return name + "@1"
    kind = "cont" if env_cfg.continuing_task else "term"
    family = "antmaze" if environment_type == "AntMazeConfig" else "pointmaze"
    name = f"{family}_{env_cfg.map_name}_{kind}{env_cfg.max_episode_steps}_nonoise"
    if family == "pointmaze" and (env_cfg.map_name, env_cfg.start_cell, env_cfg.goal_cell) != (
            "large", (7, 1), (1, 10)):
        name += (f"_start{env_cfg.start_cell[0]}-{env_cfg.start_cell[1]}"
                 f"_goal{env_cfg.goal_cell[0]}-{env_cfg.goal_cell[1]}")
    if family == "antmaze" and env_cfg.physics != "jax":
        name += f"_physics-{env_cfg.physics}"
    if env_cfg.reward_shift:
        name += f"_shift{env_cfg.reward_shift:g}"
    if family == "pointmaze" and env_cfg.goal_reward != "per_step":
        name += f"_{env_cfg.goal_reward}"
    if reward_setting != "environment":
        name += "_reward-" + reward_setting.replace("_", "-")
    return name + "@1"


def specs_of(bonus: str, config, env_cfg, reward_setting: str = "environment") -> dict:
    """Record the actual environment and its named algorithm configuration, plus the bonus."""
    # an older name for a family records the family's own name, so two runs written against
    # different vocabularies are still comparable by their manifests
    from exploration_platform.bonuses.registry import ALIASES
    name = ALIASES.get(bonus, bonus)
    if name not in UPDATE_SCHEDULES:
        raise ValueError(f"no update schedule recorded for bonus {bonus!r}; add it to "
                         f"UPDATE_SCHEDULES so the run's manifest names one")
    return {"env": env_spec(env_cfg, reward_setting),
            "agent": AGENT_SPECS[config.configuration_preset],
            "configuration_preset": config.configuration_preset,
            "bonus": f"{name}@1", "update_schedule": UPDATE_SCHEDULES[name]}


def build_config(args: argparse.Namespace):
    """Preserve the existing CLI construction and retain complete optional file settings."""
    from dataclasses import replace
    from experiment_configuration import additional_trainer_fields
    from exploration_platform.agents.ppo.presets import resolve_configuration

    # Without the new flag this is exactly the existing constructor, with no replaced defaults.
    configuration = _build_config_values(args)
    if getattr(args, "experiment_configuration_data", None) is None:
        return configuration
    configuration = replace(configuration, **additional_trainer_fields(args))
    return resolve_configuration(configuration, build_env_config(args))


def cross_product_cells(lists: dict, reward_pairs: tuple):
    """The cells of a sweep over several per-copy settings, and the name of each column.

    lists maps a setting name to the values to try (an empty list leaves that setting a scalar).
    The first name varies slowest. Two reward mappings add the reward scale and offset as the
    last two columns, so each cell appears once per reward setting.

    before: {"learning_rate": [1e-4], "int_coef": [], "ent_coef": [0.001, 0.01]}, one mapping
    after:  (("learning_rate", "ent_coef"), ((1e-4, 0.001), (1e-4, 0.01)))
    before: {"int_coef": [1.0, 4.0]}, mappings ((1.0, 0.0), (0.0025, -0.0025))
    after:  (("int_coef", "ext_reward_scale", "ext_reward_offset"),
             ((1.0, 1.0, 0.0), (1.0, 0.0025, -0.0025), (4.0, 1.0, 0.0), (4.0, 0.0025, -0.0025)))
    """
    from itertools import product
    names = tuple(name for name, values in lists.items() if values)
    axes = [[(value,) for value in lists[name]] for name in names]
    if len(reward_pairs) == 2:
        names += REWARD_SETTING_NAMES
        axes.append(list(reward_pairs))
    return names, tuple(sum(combination, ()) for combination in product(*axes))


def planned_iterations(args: argparse.Namespace) -> int:
    """The planned length of the run that the trainer configuration carries; 0 when nothing in
    the trainer reads it.

    The intrinsic weight's schedule reads this run's own length. A reward divisor held after a
    fraction of the run reads the length the learning-rate anneal runs over, so a unit holds at
    the same iteration however often it is stopped and continued.
    before: --iterations 16384, --bonus-options reward_divisor=held_after_fraction,... ; after: 16384
    before: --iterations 16384, neither a weight schedule nor a held divisor         ; after: 0
    """
    from exploration_platform.bonuses.registry import family_configuration
    family = family_configuration(args.bonus, bonus_options_of(args))
    held_divisor = getattr(family, "reward_divisor", None) == "held_after_fraction"
    weight_schedule = args.intrinsic_weight_schedule != "constant"
    anneal_iterations = args.lr_anneal_iterations or args.iterations
    if not held_divisor:
        return args.iterations if weight_schedule else 0
    if weight_schedule and anneal_iterations != args.iterations:
        raise ValueError("the intrinsic weight's schedule runs over --iterations and the held "
                         "reward divisor over --lr-anneal-iterations; the trainer carries one "
                         "planned length, so the two must be equal when both are used")
    return anneal_iterations


def _build_config_values(args: argparse.Namespace):
    """The trainer configuration this unit runs, sweeping the knobs whose lists are non-empty."""
    from exploration_platform.agents.ppo.config import PPOConfig
    from exploration_platform.training.sweep import sweep_config
    from exploration_platform.agents.ppo.presets import resolve_configuration
    from exploration_platform.bonuses.registry import ALIASES
    env_cfg = build_env_config(args)
    selected = resolve_configuration(PPOConfig(configuration_preset=args.configuration_preset),
                                     env_cfg)
    rates, weights = number_list(args.learning_rates), number_list(args.intrinsic_weights)
    more_lists = {"ent_coef": number_list(args.entropy_coefficients),
                  "clip_coef": number_list(args.clip_coefficients),
                  "vf_coef": number_list(args.value_coefficients)}
    no_bonus = ALIASES.get(args.bonus, args.bonus) == "none"
    task_only = no_bonus and selected.no_bonus_advantage_stream == "task_only"
    intrinsic_coefficient = 0. if task_only else args.intrinsic_coefficient

    # the reward the learner sees. The environment's own reward keeps the scale and offset of a
    # configuration file (1 and 0 without one); a named setting fixes them from the step limit;
    # both named settings at once put them into every cell instead.
    document = getattr(args, "experiment_configuration_data", None)
    reward_pairs = reward_mappings(args.reward_setting, env_cfg.max_episode_steps)
    if len(reward_pairs) == 1:
        reward_scale, reward_offset = reward_pairs[0]
    elif document is not None and not reward_pairs:
        reward_scale = document["trainer"]["ext_reward_scale"]
        reward_offset = document["trainer"]["ext_reward_offset"]
    else:
        reward_scale, reward_offset = 1.0, 0.0
    common = dict(configuration_preset=selected.configuration_preset,
                  intrinsic_bootstrap_observation=selected.intrinsic_bootstrap_observation,
                  no_bonus_advantage_stream=selected.no_bonus_advantage_stream,
                  n_envs=args.envs_per_copy, num_steps=args.rollout_steps,
                  base_seed=args.base_seed, track_coverage=args.track_coverage,
                  copy_seed_offset=args.copy_seed_offset, flat_params=args.flat_params,
                  scan_unroll=args.scan_unroll,
                  ent_coef=args.ent_coef, gamma_ext=args.gamma_ext, ext_coef=args.ext_coef,
                  max_grad_norm=args.max_grad_norm, update_epochs=args.update_epochs,
                  num_minibatches=args.num_minibatches, diagnostics=args.diagnostics,
                  int_coef_schedule=args.intrinsic_weight_schedule,
                  total_iterations=planned_iterations(args),
                  success_gate_rate=args.success_gate_rate,
                  normalize_ext_reward=args.normalize_extrinsic_reward,
                  clip_per_network=args.clip_per_network, logstd_floor=args.logstd_floor,
                  critic_lr_scale=args.critic_lr_scale, vf_coef=args.vf_coef,
                  bootstrap_on_truncation=not args.no_bootstrap_on_truncation,
                  normalize_agent_obs=args.normalize_agent_obs,
                  clip_coef=args.clip_coef, prime_iterations=args.prime_iterations,
                  episode_log=args.episode_log,
                  gather_minibatches_per_epoch=args.gather_minibatches_per_epoch,
                  ext_reward_scale=reward_scale, ext_reward_offset=reward_offset)
    # Reject PointMaze-only study options before a run directory or its manifest is written.
    resolve_configuration(PPOConfig(**common), env_cfg)
    if 0 < args.lr_anneal_iterations < args.iterations:
        raise ValueError("--lr-anneal-iterations shorter than the run would anneal the learning "
                         "rate below zero")
    # the object-observation Montezuma configuration has no goal and always runs to its step cap
    if not getattr(env_cfg, "continuing_task", True) and not (args.episode_log
                                                             or args.diagnostics):
        raise ValueError("an episode that ends at the goal needs --episode-log (or "
                         "--diagnostics): episodes then end at different steps, and only an "
                         "episode log counts them")
    # Before: two rates x two weights = four cells; after: only the listed cells train.
    settings_from_file = getattr(args, "experiment_parameter_settings", ())
    if args.parameter_settings or settings_from_file:
        if rates or weights or any(more_lists.values()):
            raise ValueError("explicit parameter settings cannot be combined with sweep lists")
        if args.parameter_settings:
            mapping = json.loads(args.parameter_settings.read_text())
            if args.bonus not in mapping or not mapping[args.bonus]:
                raise ValueError(f"no parameter settings listed for {args.bonus}")
            rows, names = mapping[args.bonus], DEFAULT_SETTING_NAMES
        else:
            rows = settings_from_file
            names = getattr(args, "experiment_setting_names", DEFAULT_SETTING_NAMES)
        settings = tuple(tuple(float(value) for value in row) for row in rows)
        if any(len(row) != len(names) for row in settings) or args.copies_per_cell < 1:
            raise ValueError(f"explicit settings require one value for each of {list(names)} "
                             "per cell, and copies-per-cell")
        if len(reward_pairs) == 2 and not set(REWARD_SETTING_NAMES) <= set(names):
            # every listed cell once per reward setting, on the same seeds
            names = names + REWARD_SETTING_NAMES
            settings = tuple(row + pair for row in settings for pair in reward_pairs)
        if task_only and "int_coef" in names and any(
                row[names.index("int_coef")] != 0 for row in settings):
            raise ValueError("no-bonus settings must use intrinsic weight zero")
        # a cell that names the learning rate replaces the scalar, which keeps its default as
        # the recorded files expect; otherwise the scalar is the rate every copy trains at
        scalars = {} if "learning_rate" in names else {"learning_rate": args.learning_rate}
        return PPOConfig(n_copies=len(settings) * args.copies_per_cell,
                         update_style=args.update_style, parameter_settings=settings,
                         setting_names=names, copies_per_group=args.copies_per_cell,
                         int_coef=intrinsic_coefficient, **scalars, **common)
    if task_only and any(weight != 0 for weight in weights):
        raise ValueError("no-bonus sweeps must use intrinsic weight zero")
    # a sweep over more than rates and weights, or over both reward settings, is written as
    # explicit cells: the cross product of every list, each column named
    if any(more_lists.values()) or len(reward_pairs) == 2:
        if args.copies_per_cell <= 0:
            raise ValueError("a sweep needs --copies-per-cell")
        names, settings = cross_product_cells(
            {"learning_rate": rates, "int_coef": weights, **more_lists}, reward_pairs)
        return PPOConfig(n_copies=len(settings) * args.copies_per_cell,
                         update_style=args.update_style, parameter_settings=settings,
                         setting_names=names, copies_per_group=args.copies_per_cell,
                         learning_rate=args.learning_rate, int_coef=intrinsic_coefficient,
                         **common)
    # a sweep derives its copy count from the cells; a single configuration takes --copies
    if rates or weights:
        if args.copies_per_cell <= 0:
            raise ValueError("a sweep needs --copies-per-cell")
        return sweep_config(learning_rates=rates, betas=weights,
                            copies_per_group=args.copies_per_cell, style=args.update_style,
                            learning_rate=args.learning_rate, int_coef=intrinsic_coefficient,
                            **common)
    return PPOConfig(n_copies=args.copies, update_style=args.update_style,
                     learning_rate=args.learning_rate, int_coef=intrinsic_coefficient,
                     **common)


def write_launch_files(run_dir: Path, args: argparse.Namespace, config, env_cfg) -> None:
    """Write command.txt, config_resolved.yaml and manifest.yaml before the first iteration."""
    # the exact command, so the run can be repeated without reconstructing it from prose. Arguments
    # are quoted the way a shell needs them, or an argument holding spaces (the description) would
    # come back as several arguments.
    # before: sys.argv = ["run_training.py", "--description", "a proof run"]
    # after:  PYTHONNOUSERSITE=1 <python> run_training.py --description 'a proof run'
    (run_dir / "command.txt").write_text(
        "PYTHONNOUSERSITE=1 " + shlex.quote(sys.executable) + " "
        + " ".join(shlex.quote(argument) for argument in sys.argv) + "\n")

    # the full configuration after defaults and overrides — what actually ran, not what was typed
    resolved = {"trainer": {field: getattr(config, field)
                            for field in sorted(config.__dataclass_fields__)},
                "environment": {"specification": env_spec(env_cfg, args.reward_setting),
                                **{field: (str(getattr(env_cfg, field))
                                           if isinstance(getattr(env_cfg, field), tuple)
                                           else getattr(env_cfg, field))
                                   for field in env_cfg.__dataclass_fields__}},
                "runtime": {"interpreter": sys.executable, "unit_id": args.unit_id,
                            "iterations": args.iterations,
                            "window_iterations": args.window_iterations,
                            "run_seed": args.run_seed,
                            "reward_setting": args.reward_setting}}
    document = getattr(args, "experiment_configuration_data", None)
    if document is not None:
        resolved["configuration_file"] = {name: document[name] for name in ("path", "sha256", "metadata")}
    # the bonus family's own options, every one with its value (absent for a family with none)
    from exploration_platform.bonuses.registry import bonus_options_record
    bonus_record = bonus_options_record(args.bonus, bonus_options_of(args))
    if bonus_record:
        resolved["bonus_options"] = bonus_record
    (run_dir / "config_resolved.yaml").write_text(as_yaml(resolved) + "\n")

    # machine-readable identity of the run: which code, which component versions, which shape
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "description": args.description or "one training unit on the JAX exploration platform",
        "git": git_state(PLATFORM_ROOT.parent),
        "specs": specs_of(args.bonus, config, env_cfg, args.reward_setting),
        "compile_signature": {
            "copies": config.n_copies, "envs_per_copy": config.n_envs,
            "rollout_steps": config.num_steps, "update_style": config.update_style,
            "epochs": config.update_epochs, "minibatches": config.num_minibatches,
            "dtype": "float32", "stats_dtype": "float64",
        },
        "seeding": {"base_seed": config.base_seed, "run_seed": args.run_seed,
                    "mode": config.sweep_seed_mode if config.copies_per_group else "distinct"},
        "artifacts": {"metrics": "metrics.jsonl", "summary": "summary.json",
                      "shards": "data/*.jsonl"},
    }
    (run_dir / "manifest.yaml").write_text(as_yaml(manifest) + "\n")


def keep_records_through(shard: Path, iteration: int) -> int:
    """Drop the window records later than a checkpoint, so a resumed shard holds each window once.

    before: windows ending at iterations 3, 6 and 9, and a checkpoint taken at iteration 7
    after:  the windows ending at 3 and 6 stay; the one ending at 9 will be written again
    Returns the number of records kept. The file is replaced only once the new one is whole.
    """
    kept = [line for line in shard.read_text().splitlines() if line.strip()
            and not (json.loads(line).get("record") == "episode_window"
                     and json.loads(line)["last_iteration"] > iteration)]
    temporary = shard.with_name(shard.name + ".tmp")
    temporary.write_text("".join(line + "\n" for line in kept))
    os.replace(temporary, shard)
    return len(kept)


def pool_record(report):
    """The use of MuJoCo Warp's shared contact pool in a record's words; None without a pool.

    before: {"nacon_peak": 9731, "ncollision_peak": 24110, "naconmax": 786432, "nefc_peak": 39,
             "njmax": 160, "overflow_bits": 0, "dropped": False, "reasons": []}
    after:  {"contacts_largest": 9731, "candidate_pairs_largest": 24110, "pool_slots": 786432,
             "pool_fraction_largest": 0.0307, "constraint_rows_largest": 39,
             "constraint_row_limit": 160}
    Contacts and candidate pairs each have a pool of `pool_slots`; the fraction is the fuller one.
    """
    if report is None:
        return None
    fullest = max(report["nacon_peak"], report["ncollision_peak"])
    return {"contacts_largest": report["nacon_peak"],
            "candidate_pairs_largest": report["ncollision_peak"],
            "pool_slots": report["naconmax"],
            "pool_fraction_largest": round(fullest / report["naconmax"], 4),
            "constraint_rows_largest": report["nefc_peak"],
            "constraint_row_limit": report["njmax"]}


def checkpoint_identity(args: argparse.Namespace, config, env_cfg, sweep) -> dict:
    """What must agree before a checkpoint may continue this unit: everything that shapes it."""
    from dataclasses import asdict
    from exploration_platform.bonuses.registry import bonus_options_record
    # the bonus family's options enter only when the unit changes one, so a unit that changes
    # none keeps the identity it had before the options existed
    changed = bonus_options_of(args)
    return {"trainer_config": asdict(config), "environment_config": asdict(env_cfg),
            "environment": env_spec(env_cfg, args.reward_setting), "bonus": args.bonus,
            **({"bonus_options": bonus_options_record(args.bonus, changed)} if changed else {}),
            "run_seed": args.run_seed, "copy_seed_index": list(sweep.copy_seed_index),
            "iterations": args.iterations, "window_iterations": args.window_iterations,
            "lr_anneal_iterations": args.lr_anneal_iterations or args.iterations}


def main() -> None:
    """Create the run folder, train, write one record per phase, then aggregate."""
    args = parse_args()
    process_start = time.time()
    if args.compilation_cache_dir:
        # JAX's persistent compilation cache, every program in it; JAX decides at its first
        # compilation whether a process uses the cache, so this comes before any other import
        import jax
        jax.config.update("jax_compilation_cache_dir", args.compilation_cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    if args.episode_steps == 0:
        args.episode_steps = ENVIRONMENTS[args.env]["episode_steps"]
    run_dir = Path(args.run_dir).resolve()
    for sub in ("data", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    from exploration_platform.bonuses.registry import bonus_options_record, make_bonus
    from exploration_platform.training import episode_log, maze_grid
    from exploration_platform.training.checkpoint import (check_identity, read_checkpoint,
                                                          restore_state, save_checkpoint)
    from exploration_platform.training.runner import Runner
    import jax
    import jax.numpy as jnp

    config = build_config(args)
    env_cfg = build_env_config(args)
    # the bonus family's options this unit changes, and the family's whole option record; an
    # unknown option or a combination that is not built is refused here
    bonus_options = bonus_options_of(args)
    bonus_record = bonus_options_record(args.bonus, bonus_options)
    if not (run_dir / "manifest.yaml").exists():
        write_launch_files(run_dir, args, config, env_cfg)

    shard = run_dir / "data" / f"{args.unit_id}.jsonl"
    log = open(run_dir / "logs" / f"{args.unit_id}.log", "a", buffering=1)

    def say(message: str) -> None:
        """Print to the terminal and to the unit's log, both immediately."""
        line = f"[{pacific_now()}] {message}"
        print(line, flush=True)
        log.write(line + "\n")

    # checkpoints are on only when one of their two options is given; without them this unit
    # runs exactly as it always did
    checkpointing = args.checkpoint_seconds > 0 or args.time_limit_seconds > 0
    checkpoint_path = run_dir / "checkpoints" / f"{args.unit_id}.npz"
    saved = None

    # a shard whose last line says the unit finished is left alone: re-running is then a no-op, and
    # that is what makes a job's command resumable — it restarts at the first unfinished unit
    if shard.exists():
        existing = [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]
        if any(record.get("record") == "unit_complete" for record in existing):
            say(f"resume: {args.unit_id} is already complete in {shard.name} "
                f"({len(existing)} records on disk); skipping it")
            return
        if checkpointing and checkpoint_path.exists():
            saved = read_checkpoint(checkpoint_path)
        else:
            say(f"resume: {args.unit_id} has a partial shard with {len(existing)} records; no "
                f"model state is saved, so this is a fresh attempt appended to the same file")
    elif checkpointing and checkpoint_path.exists():
        raise ValueError(f"{checkpoint_path} exists but the unit's shard {shard} does not; "
                         "delete the checkpoint to start this unit again")

    # one line per record, opened for appending and flushed immediately, so the file is the live
    # progress signal and a re-run never truncates what is already there
    out = open(shard, "a", buffering=1)

    def emit(record: dict) -> None:
        """Append one record to the shard and push it to disk before returning."""
        record["written_at"] = datetime.now().astimezone().isoformat()
        out.write(json.dumps(record) + "\n")
        out.flush()
        os.fsync(out.fileno())

    def emit_job(**extra) -> None:
        """Record what this attempt runs on, as the attempt's first line in the shard."""
        hardware = hardware_record()
        hardware.update(unit_id=args.unit_id,
                        attempt_started=datetime.now().astimezone().isoformat(), **extra)
        emit(hardware)
        say(f"unit {args.unit_id} on {hardware['host']} / {hardware['device_kind']}, "
            f"jax {hardware['jax']}")

    if saved is None:
        emit_job()

    # build the trainer, then time the compilation of the first iteration separately from the rest,
    # because the first iteration pays for compiling the program. A resumed unit hands the
    # trainer the bonus's saved fixed arrays, so the program holds the numbers it started with.
    build_start = time.time()
    trainer = Runner(config, bonus=make_bonus(args.bonus, bonus_options), env_cfg=env_cfg,
                     bonus_constants=saved["bonus_constants"] if saved else None)
    if args.compilation_cache_dir or args.put_state_on_card:
        say(f"JAX compilation cache: {args.compilation_cache_dir or 'off'}; state put on the card "
            f"before calls: {'yes' if args.put_state_on_card else 'no'}")

    def on_card(tree):
        """With --put-state-on-card, the tree as arrays placed on the card (the form a compiled
        iteration returns); otherwise the tree as it is. The values do not change.

        before: a state whose arrays were made outside a compiled program (reset, restore, a
        window's counters set back to zero); a jitted call on it compiles once more
        after:  the same values, placed on jax.devices()[0]
        """
        return jax.device_put(tree, jax.devices()[0]) if args.put_state_on_card else tree
    config = trainer.cfg
    sweep = trainer.sweep
    # the program that scores the maze-grid probes, only when the bonus's options ask for it
    maze_probe = (maze_grid.build_probe(config, trainer.env, trainer.bonus, trainer.layout)
                  if trainer.bonus.record_on_maze_grid else None)
    kernel_folder = getattr(trainer.env, "kernel_folder", None)
    if kernel_folder is not None:
        # MuJoCo Warp: where its compiled kernels are kept on this node, and the pool's size
        environments = config.n_copies * config.n_envs
        say(f"MuJoCo Warp physics: kernel folder {kernel_folder}, holding "
            f"{sum(1 for entry in os.scandir(kernel_folder) if entry.is_dir())} compiled kernel "
            f"folders before this unit's first step; shared contact pool {trainer.env.naconmax:,} "
            f"slots ({trainer.env.naconmax / environments:g} per environment x {environments:,} "
            f"environments), at most {trainer.env.njmax} constraint rows per environment")

    def checked_pool():
        """The environment's pool use as a record entry (None without a pool); a drop is written
        to the unit's log and stops the unit."""
        try:
            return pool_record(trainer.check_environment(state))
        except RuntimeError as error:
            say(f"STOPPED: {error}")
            raise

    identity = checkpoint_identity(args, config, env_cfg, sweep)
    if saved is not None:
        # the checkpoint must belong to this unit before the shard is touched; the window
        # records later than it are then dropped, and this attempt's lines follow the kept ones
        from dataclasses import asdict
        check_identity(saved["identity"], identity,
                       environment_defaults=asdict(type(env_cfg)()),
                       bonus_option_defaults=bonus_options_record(args.bonus))
        state = restore_state(trainer.init_state(run_seed=args.run_seed), saved["leaves"])
        start_iteration = int(np.asarray(state.step)) + 1
        out.close()
        kept = keep_records_through(shard, start_iteration - 1)
        out = open(shard, "a", buffering=1)
        emit_job(resumed_from_iteration=start_iteration - 1)

    steps_per_iteration = config.num_steps * config.n_copies * config.n_envs
    steps_per_copy_per_iteration = config.num_steps * config.n_envs
    # the episode clock: how many iterations it takes for the 128-step window to return to the
    # same place inside the 400-step episode. A window spanning a whole number of these carries
    # every part of the episode exactly once, so its reward sum is free of the clock. An episode
    # that also ends at the goal has no such clock: with the episode log on, the fields derived
    # from it are then written as null and the episode counts come from the log.
    ends_at_goal = not getattr(env_cfg, "continuing_task", True)
    fixed_clock = not (ends_at_goal and config.episode_log)
    episodes_per_copy_per_iteration = (steps_per_copy_per_iteration / args.episode_steps
                                       if fixed_clock else None)
    clock_iterations = int(args.episode_steps // np.gcd(config.num_steps, args.episode_steps))

    def episode_phase(iteration: int) -> int:
        """Where inside the episode this iteration's rollout window begins, in environment steps.

        before: iteration 19400, 128-step rollouts, 400-step episodes
        after:  272 — that iteration covered episode steps 272 to 400 and then 0 to 128 of the next
        """
        return ((iteration - 1) * config.num_steps) % args.episode_steps

    # the unit's own identity card: which cell each copy belongs to, so the aggregator can group
    # the per-copy arrays of every later record without re-deriving the sweep. A resumed attempt
    # writes a short record instead, so the unit keeps one identity card and readers that take
    # the records after the last `unit_start` still see every window.
    configuration_document = getattr(args, "experiment_configuration_data", None)
    if saved is not None:
        emit({"record": "unit_resume", "unit_id": args.unit_id,
              "resumed_from_iteration": start_iteration - 1, "records_kept": kept,
              "checkpoint": str(checkpoint_path)})
        say(f"resume: continuing {args.unit_id} from the checkpoint at iteration "
            f"{start_iteration - 1} of {args.iterations}; {kept} records kept in {shard.name}")
    else:
        emit({"record": "unit_start", "unit_id": args.unit_id, "bonus": args.bonus,
              "environment": env_spec(env_cfg, args.reward_setting),
              "diagnostics": config.diagnostics,
              # this unit's whole configuration: units of one run folder may differ in their
              # flags, while the folder's config_resolved.yaml holds only the first unit's
              "trainer_config": {field: (list(value) if isinstance(value, tuple) else value)
                                 for field, value in ((f, getattr(config, f))
                                                      for f in sorted(config.__dataclass_fields__))},
              "environment_config": {field: (list(value) if isinstance(value, tuple) else value)
                                     for field, value in ((f, getattr(env_cfg, f))
                                                          for f in env_cfg.__dataclass_fields__)},
              "copies": config.n_copies, "envs_per_copy": config.n_envs,
              "rollout_steps": config.num_steps, "iterations": args.iterations,
              "window_iterations": args.window_iterations, "episode_steps": args.episode_steps,
              "episode_clock_iterations": int(clock_iterations) if fixed_clock else None,
              "copies_per_cell": config.copies_per_group,
              # one row per cell, one value per name in cell_setting_names
              "cell_settings": [list(setting) for setting in sweep.group_settings],
              "cell_setting_names": list(sweep.setting_names),
              "copy_cell": sweep.copy_group.tolist(),
              "copy_seed_index": list(sweep.copy_seed_index),
              # the slice of a chunked run this process holds; 0 and the whole count for an
              # unchunked run, so an aggregator can tell chunks apart and check that they cover
              # the run exactly
              "copy_seed_offset": config.copy_seed_offset,
              "copy_seed_index_first": int(min(sweep.copy_seed_index)),
              "copy_seed_index_last": int(max(sweep.copy_seed_index)),
              "sweep_seed_mode": config.sweep_seed_mode if sweep.is_sweep else "distinct",
              "episodes_per_copy_per_iteration": episodes_per_copy_per_iteration,
              "steps_per_iteration": steps_per_iteration,
              "lr_anneal_iterations": args.lr_anneal_iterations or args.iterations,
              "reward_setting": args.reward_setting, "episode_log": config.episode_log,
              # every option of the bonus family with its value ({} for a family with none), so
              # two units of one family can be told apart; and, when the bonus is recorded on
              # the maze grid, the (row, column) cell of each probe, in the order of the
              # per-probe lists of the window records
              "bonus_options": bonus_record,
              **({"maze_grid_probe_cells": [list(cell) for cell in
                                            maze_grid.open_cell_list(trainer.env)]}
                 if maze_probe is not None else {}),
              **({"configuration_file": {name: configuration_document[name]
                                          for name in ("path", "sha256", "metadata")}}
                 if configuration_document is not None else {})})
    say(f"{config.n_copies} copies x {config.n_envs} environments x {config.num_steps} steps, "
        f"{args.iterations} iterations, update style {config.update_style}, "
        f"{len(sweep.group_settings)} cells of {config.copies_per_group or config.n_copies} copies")

    # the window accumulators live on the device: adding to them is dispatched without waiting, so
    # the loop never synchronises except on the iterations that close a window
    reward_in_window = jnp.zeros((config.n_copies,))
    rint_in_window = jnp.zeros((config.n_copies,))
    # the diagnostics' window sums, also on the device; None until the first iteration names them
    diag_in_window = None
    window_index = 0
    window_first_iteration = 1
    if saved is None:
        state = on_card(trainer.init_state(run_seed=args.run_seed))
        prime_start = time.time()
        state = trainer.prime(state)
        jax.block_until_ready(state.obs)
        checked_pool()
        start_iteration = 1
        say(f"building and priming done in {time.time() - build_start:.1f}s "
            f"(priming alone {time.time() - prime_start:.1f}s)")
        if maze_probe is not None:
            # the bonus on the maze grid before the first training iteration (no visit yet)
            emit({"record": "maze_grid_at_start", "unit_id": args.unit_id,
                  **maze_grid.read_window(state, maze_probe)})
    else:
        # the driver's own sums and counters continue from where the checkpoint was taken
        driver = saved["driver"]
        reward_in_window = jnp.asarray(driver["reward_in_window"])
        rint_in_window = jnp.asarray(driver["rint_in_window"])
        if "diagnostics" in driver:
            diag_in_window = jax.tree.map(jnp.asarray, driver["diagnostics"])
        window_index = int(driver["window_index"])
        window_first_iteration = int(driver["window_first_iteration"])
        say(f"building and restoring done in {time.time() - build_start:.1f}s")

    # the reward the learner sees is scale x the environment's reward + offset, per copy; the
    # episode log sums the environment's reward, and these turn its sums into the learner's
    per_copy_number = lambda name: np.asarray(
        sweep.per_copy_values.get(name, (getattr(config, name),) * config.n_copies), np.float64)
    reward_scale, reward_offset = (per_copy_number("ext_reward_scale"),
                                   per_copy_number("ext_reward_offset"))

    last_checkpoint = time.time()

    def save(reason: str) -> None:
        """Write the unit's whole state and the driver's partial window sums to the checkpoint."""
        nonlocal last_checkpoint
        checked_pool()
        driver = {"reward_in_window": np.asarray(reward_in_window),
                  "rint_in_window": np.asarray(rint_in_window),
                  "window_index": np.asarray(window_index),
                  "window_first_iteration": np.asarray(window_first_iteration)}
        if diag_in_window is not None:
            driver["diagnostics"] = {name: np.asarray(value)
                                     for name, value in diag_in_window.items()}
        save_checkpoint(checkpoint_path, state, identity, trainer.bonus.constants, driver)
        last_checkpoint = time.time()
        say(f"checkpoint at iteration {int(np.asarray(state.step))} ({reason}) in "
            f"{checkpoint_path.name}")

    if checkpointing:
        import signal
        for number in (signal.SIGTERM, signal.SIGINT):
            signal.signal(number, lambda *_: STOP.update(requested=True))

    state = on_card(state)
    start = time.time()
    last_pool_check = start
    first_iteration_seconds = None
    for iteration in range(start_iteration, args.iterations + 1):
        iteration_start = time.time()
        state, metrics = trainer.iterate(
            state, trainer.lr_argument(iteration, args.lr_anneal_iterations or args.iterations))
        reward_in_window = reward_in_window + metrics["reward_ext_sum"]
        rint_in_window = rint_in_window + metrics["rint_mean"]
        if config.diagnostics:
            diag_in_window = (metrics["diagnostics"] if diag_in_window is None else
                              jax.tree.map(jnp.add, diag_in_window, metrics["diagnostics"]))
        if first_iteration_seconds is None:
            jax.block_until_ready(metrics["loss"])
            first_iteration_seconds = time.time() - iteration_start
            say(f"first iteration (includes compiling the program) "
                f"{first_iteration_seconds:.1f}s")

        # a Warp AntMaze unit in which MuJoCo Warp dropped a contact or a constraint row stops
        # within about a minute, not only at the next window end
        if kernel_folder is not None and time.time() - last_pool_check >= POOL_CHECK_SECONDS:
            checked_pool()
            last_pool_check = time.time()

        # one record per window, and always one at the last iteration; those are the only
        # iterations that wait for the device, so recording does not serialise the loop
        window_end = iteration % args.window_iterations == 0 or iteration == args.iterations
        if window_end:
            jax.block_until_ready(metrics["loss"])
            iterations_in_window = iteration - window_first_iteration + 1
            elapsed = time.time() - start
            reward = np.asarray(reward_in_window)
            record = {
                "record": "episode_window",
                "unit_id": args.unit_id,
                "window_index": window_index,
                "first_iteration": window_first_iteration,
                "last_iteration": iteration,
                "iterations_in_window": iterations_in_window,
                "episode_phase_first_iteration": episode_phase(window_first_iteration),
                "episode_phase_last_iteration": episode_phase(iteration),
                "episode_clock_cycles_in_window": (iterations_in_window / clock_iterations
                                                   if fixed_clock else None),
                # a window covering a whole number of clock cycles sees every part of the episode
                # the same number of times, so its reward sum carries no episode-clock artifact;
                # a window that does not is excluded from every score
                "phase_blocked": (iterations_in_window % clock_iterations == 0
                                  if fixed_clock else None),
                "episodes_per_copy_in_window": (
                    iterations_in_window * episodes_per_copy_per_iteration
                    if fixed_clock else None),
                "episodes_per_copy_cumulative": (iteration * episodes_per_copy_per_iteration
                                                 if fixed_clock else None),
                "seconds_since_first_iteration": elapsed,
                "env_steps": iteration * steps_per_iteration,
                "env_steps_per_copy": iteration * steps_per_copy_per_iteration,
                "loss": float(np.asarray(metrics["loss"])),
                "reward_ext_sum_per_copy": reward.tolist(),
                "rint_mean_per_copy":
                    (np.asarray(rint_in_window) / iterations_in_window).tolist(),
            }
            # the Warp AntMaze's largest pool use so far; a unit in which MuJoCo Warp dropped
            # a contact or a constraint row stops here, before this window is written
            pool = checked_pool()
            if pool is not None:
                record["physics_pool"] = pool
            if config.track_coverage:
                record["coverage_per_copy"] = trainer.coverage(state).tolist()
            if config.diagnostics:
                # episode_* fields stay sums over the window (episodes ended, reached, ...); every
                # other field is the mean over the window's iterations. Six significant digits.
                # before: {"episode_episodes": [3., 2.], "approx_kl": [0.002, 0.004]} summed over
                #         2 iterations; after: episode_episodes [3, 2], approx_kl [0.001, 0.002]
                record["diagnostics_per_copy"] = {
                    name: [float(f"{x:.6g}") for x in np.asarray(
                        value if name.startswith("episode_") else value / iterations_in_window)]
                    for name, value in diag_in_window.items()}
                diag_in_window = None
            if config.episode_log:
                # the episodes that ended in this window, per copy, read from the state and set
                # back to zero there. Returns of the reward the learner sees follow from the
                # environment's: scale x (sum of returns) + offset x (sum of lengths).
                # before: 2 episodes of lengths 30 and 400, one arrival, environment returns 1
                #         and 0, scale 1/400 and offset -1/400
                # after:  return_sum_per_copy 1/400 - 430/400 = -1.0725
                window_log = episode_log.read_window(state.agent_state["episode_log"])
                state = state._replace(agent_state={
                    **state.agent_state,
                    "episode_log": episode_log.clear_window(state.agent_state["episode_log"])})
                record.update({
                    "episodes_per_copy": window_log["episodes"].tolist(),
                    "arrivals_per_copy": window_log["arrivals"].tolist(),
                    "return_sum_per_copy": (reward_scale * window_log["return_sum"]
                                            + reward_offset * window_log["length_sum"]).tolist(),
                    "environment_return_sum_per_copy": window_log["return_sum"].tolist(),
                    "length_sum_per_copy": window_log["length_sum"].tolist(),
                    "steps_to_goal_sum_per_copy": window_log["steps_to_goal_sum"].tolist(),
                    "first_goal_step_per_copy": window_log["first_goal_step"].tolist()})
            fault_window = getattr(trainer.env, "fault_window", None)
            if fault_window is not None:
                # AntMaze only: the window's steps with the torso in the ceiling zone and its
                # largest torso speed, per copy, read from the environment state and set back to
                # zero there (envs/antmaze/faults.py)
                fault_record, env_state = fault_window(state.env_state)
                record.update(fault_record)
                state = state._replace(env_state=env_state)
            if maze_probe is not None:
                # the bonus at each open cell's probe as the window ends, the divisor, and the
                # window's visits to each cell, per copy; the visit counts go back to zero
                record.update(maze_grid.read_window(state, maze_probe))
                state = state._replace(agent_state=maze_grid.clear_window(state.agent_state))
            emit(record)
            coverage_note = (f", maze coverage mean {np.mean(record['coverage_per_copy']):.3f}"
                             if config.track_coverage else "")
            episode_note = (f", {int(np.sum(record['episodes_per_copy']))} episodes ended and "
                            f"{int(np.sum(record['arrivals_per_copy']))} reached the goal"
                            if config.episode_log else "")
            clock_note = (f"phase blocked {record['phase_blocked']}" if fixed_clock
                          else "no episode clock: episodes end at the goal")
            pool_note = (f", contact pool at most {pool['pool_fraction_largest']:.1%} used"
                         if pool is not None else "")
            say(f"window {window_index} (iterations {window_first_iteration}-{iteration} of "
                f"{args.iterations}, {clock_note}) "
                f"loss {record['loss']:.4f}, extrinsic reward per copy summed over the window "
                f"mean {reward.mean():.3f}{coverage_note}{episode_note}{pool_note}, elapsed {elapsed:.1f}s")
            reward_in_window = jnp.zeros((config.n_copies,))
            rint_in_window = jnp.zeros((config.n_copies,))
            window_index += 1
            window_first_iteration = iteration + 1
            # the window's counters were set back to zero outside the compiled program
            state = on_card(state)

        # save when told to stop, at the time limit, or at a window end once the interval has
        # passed; a stop ends the process here, and the same command continues the unit later
        if checkpointing and iteration < args.iterations:
            now = time.time()
            stopping = STOP["requested"] or (args.time_limit_seconds > 0
                                             and now - process_start >= args.time_limit_seconds)
            if stopping:
                save("stop signal" if STOP["requested"] else "time limit")
                out.close()
                log.close()
                raise SystemExit(f"unit {args.unit_id} stopped at iteration {iteration} of "
                                 f"{args.iterations} with its state saved; run the same command "
                                 "again to continue it")
            if window_end and args.checkpoint_seconds > 0 and (
                    now - last_checkpoint >= args.checkpoint_seconds):
                save("interval")

    jax.block_until_ready(state.agent_params)
    total_seconds = time.time() - start
    # the steady-state rate excludes the first iteration, which paid for compiling the program
    iterations_this_attempt = args.iterations - start_iteration + 1
    steady_seconds = total_seconds - (first_iteration_seconds or 0.0)
    steady_iterations = max(iterations_this_attempt - 1, 1)
    # environments that were respawned for a non-finite physics state (the MJX families
    # count them; other families have no such field)
    nan_count = getattr(state.env_state, "nan_count", None)
    nan_episodes = int(np.asarray(nan_count).sum()) if nan_count is not None else None
    final_pool = checked_pool()
    emit({
        "record": "unit_complete",
        "unit_id": args.unit_id,
        "nan_respawned_episodes": nan_episodes,
        # the Warp AntMaze's largest use of its shared contact pool over the whole unit
        **({"physics_pool": final_pool} if final_pool is not None else {}),
        "iterations": args.iterations,
        "windows": window_index,
        "env_steps": args.iterations * steps_per_iteration,
        "env_steps_per_copy": args.iterations * steps_per_copy_per_iteration,
        "episodes_per_copy": (args.iterations * episodes_per_copy_per_iteration
                              if fixed_clock else None),
        "copies": config.n_copies,
        "seconds_total": total_seconds,
        "seconds_build_and_prime": start - build_start,
        "seconds_first_iteration_with_compile": first_iteration_seconds,
        "seconds_per_iteration_steady": steady_seconds / steady_iterations,
        "attempt_finished": datetime.now().astimezone().isoformat(),
        # a resumed attempt's seconds cover only the iterations it ran itself
        **({"resumed_from_iteration": start_iteration - 1,
            "iterations_this_attempt": iterations_this_attempt} if saved is not None else {}),
    })
    nan_note = (f", {nan_episodes} episodes respawned for a non-finite physics state"
                if nan_episodes else "")
    say(f"unit {args.unit_id} complete: {args.iterations} iterations in "
        f"{total_seconds:.1f}s{nan_note}")
    out.close()
    # a complete unit needs no checkpoint: its shard says it is finished
    if checkpointing and checkpoint_path.exists():
        checkpoint_path.unlink()

    if args.no_aggregate:
        log.close()
        return

    # one shard is still a sweep of one: the run-level files are always the aggregate
    from aggregate_run import aggregate
    summary = aggregate(run_dir)
    say(f"aggregated: {summary['records']} episode-window records, steady rate "
        f"{summary['throughput']['total_env_steps_per_second_steady']:.3e} env steps per second in "
        f"total and {summary['throughput']['env_steps_per_second_per_copy_steady']:.0f} per copy "
        f"({summary['throughput']['hours_per_million_steps_per_copy_steady']:.2f} hours per million "
        f"steps per copy)")
    log.close()


if __name__ == "__main__":
    main()
