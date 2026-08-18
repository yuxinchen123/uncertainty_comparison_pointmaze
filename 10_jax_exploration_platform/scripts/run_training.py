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

No model state is saved: checkpoints are off by default on this platform. Resuming is therefore per
UNIT: a unit whose shard already ends in a `unit_complete` record is skipped, so re-running the same
command continues a job at the first unfinished unit and never repeats finished work.

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

import numpy as np

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

SPECS = {
    "agent": "ppo_full_batch@1",
}


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
    commit = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain"],
                            capture_output=True, text=True, check=True).stdout.strip()
    describe = subprocess.run(["git", "-C", str(repo_root), "describe", "--tags", "--always"],
                              capture_output=True, text=True).stdout.strip()
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


def parse_args() -> argparse.Namespace:
    """Every knob this run exposes; the defaults are the trainer's own."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="the run folder; created if missing")
    # required, and deliberately without a default: a job that loses its arguments must fail rather
    # than write a shard full of the runner's defaults under a plausible-looking name
    parser.add_argument("--unit-id", required=True, help="names this unit's data shard")
    parser.add_argument("--description", default="", help="the human sentence for the manifest")
    parser.add_argument("--copies", type=int, default=128,
                        help="copies, when neither knob is swept; a sweep derives the count")
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
    return parser.parse_args()


def specs_of(bonus: str, env_name: str) -> dict:
    """The component identities of this run, with the environment, bonus and schedule filled in."""
    # an older name for a family records the family's own name, so two runs written against
    # different vocabularies are still comparable by their manifests
    from exploration_platform.bonuses.registry import ALIASES
    name = ALIASES.get(bonus, bonus)
    if name not in UPDATE_SCHEDULES:
        raise ValueError(f"no update schedule recorded for bonus {bonus!r}; add it to "
                         f"UPDATE_SCHEDULES so the run's manifest names one")
    return {"env": ENVIRONMENTS[env_name]["spec"], **SPECS,
            "bonus": f"{name}@1", "update_schedule": UPDATE_SCHEDULES[name]}


def build_config(args: argparse.Namespace):
    """The trainer configuration this unit runs, sweeping the knobs whose lists are non-empty."""
    from exploration_platform.agents.ppo.config import PPOConfig
    from exploration_platform.training.sweep import sweep_config
    rates, weights = number_list(args.learning_rates), number_list(args.intrinsic_weights)
    common = dict(n_envs=args.envs_per_copy, num_steps=args.rollout_steps,
                  base_seed=args.base_seed, track_coverage=args.track_coverage,
                  copy_seed_offset=args.copy_seed_offset)
    # a sweep derives its copy count from the cells; a single configuration takes --copies
    if rates or weights:
        if args.copies_per_cell <= 0:
            raise ValueError("a sweep needs --copies-per-cell")
        return sweep_config(learning_rates=rates, betas=weights,
                            copies_per_group=args.copies_per_cell, style=args.update_style,
                            learning_rate=args.learning_rate, int_coef=args.intrinsic_coefficient,
                            **common)
    return PPOConfig(n_copies=args.copies, update_style=args.update_style,
                     learning_rate=args.learning_rate, int_coef=args.intrinsic_coefficient,
                     **common)


def write_launch_files(run_dir: Path, args: argparse.Namespace, config) -> None:
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
                "environment": {"specification": ENVIRONMENTS[args.env]["spec"],
                                "max_episode_steps": args.episode_steps,
                                **ENVIRONMENTS[args.env]["resolved"]},
                "runtime": {"interpreter": sys.executable, "unit_id": args.unit_id,
                            "iterations": args.iterations,
                            "window_iterations": args.window_iterations,
                            "run_seed": args.run_seed}}
    (run_dir / "config_resolved.yaml").write_text(as_yaml(resolved) + "\n")

    # machine-readable identity of the run: which code, which component versions, which shape
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "description": args.description or "one training unit on the JAX exploration platform",
        "git": git_state(PLATFORM_ROOT.parent),
        "specs": specs_of(args.bonus, args.env),
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


def main() -> None:
    """Create the run folder, train, write one record per phase, then aggregate."""
    args = parse_args()
    if args.episode_steps == 0:
        args.episode_steps = ENVIRONMENTS[args.env]["episode_steps"]
    if args.episode_steps != ENVIRONMENTS[args.env]["episode_steps"]:
        raise ValueError(f"--episode-steps {args.episode_steps} does not match the "
                         f"{args.env} cap {ENVIRONMENTS[args.env]['episode_steps']}; the episode "
                         f"clock would be computed against a cap the environment does not use")
    run_dir = Path(args.run_dir).resolve()
    for sub in ("data", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    from exploration_platform.training.runner import Runner
    import jax
    import jax.numpy as jnp

    config = build_config(args)
    if not (run_dir / "manifest.yaml").exists():
        write_launch_files(run_dir, args, config)

    shard = run_dir / "data" / f"{args.unit_id}.jsonl"
    log = open(run_dir / "logs" / f"{args.unit_id}.log", "a", buffering=1)

    def say(message: str) -> None:
        """Print to the terminal and to the unit's log, both immediately."""
        line = f"[{pacific_now()}] {message}"
        print(line, flush=True)
        log.write(line + "\n")

    # a shard whose last line says the unit finished is left alone: re-running is then a no-op, and
    # that is what makes a job's command resumable — it restarts at the first unfinished unit
    if shard.exists():
        existing = [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]
        if any(record.get("record") == "unit_complete" for record in existing):
            say(f"resume: {args.unit_id} is already complete in {shard.name} "
                f"({len(existing)} records on disk); skipping it")
            return
        say(f"resume: {args.unit_id} has a partial shard with {len(existing)} records; no model "
            f"state is saved, so this is a fresh attempt appended to the same file")

    # one line per record, opened for appending and flushed immediately, so the file is the live
    # progress signal and a re-run never truncates what is already there
    out = open(shard, "a", buffering=1)

    def emit(record: dict) -> None:
        """Append one record to the shard and push it to disk before returning."""
        record["written_at"] = datetime.now().astimezone().isoformat()
        out.write(json.dumps(record) + "\n")
        out.flush()
        os.fsync(out.fileno())

    hardware = hardware_record()
    hardware.update(unit_id=args.unit_id, attempt_started=datetime.now().astimezone().isoformat())
    emit(hardware)
    say(f"unit {args.unit_id} on {hardware['host']} / {hardware['device_kind']}, "
        f"jax {hardware['jax']}")

    # build the trainer, then time the compilation of the first iteration separately from the rest,
    # because the first iteration pays for compiling the whole program
    build_start = time.time()
    trainer = Runner(config, bonus=args.bonus, env_cfg=env_config_of(args.env))
    sweep = trainer.sweep
    steps_per_iteration = config.num_steps * config.n_copies * config.n_envs
    steps_per_copy_per_iteration = config.num_steps * config.n_envs
    episodes_per_copy_per_iteration = steps_per_copy_per_iteration / args.episode_steps
    # the episode clock: how many iterations it takes for the 128-step window to return to the
    # same place inside the 400-step episode. A window spanning a whole number of these carries
    # every part of the episode exactly once, so its reward sum is free of the clock.
    clock_iterations = int(args.episode_steps // np.gcd(config.num_steps, args.episode_steps))

    def episode_phase(iteration: int) -> int:
        """Where inside the episode this iteration's rollout window begins, in environment steps.

        before: iteration 19400, 128-step rollouts, 400-step episodes
        after:  272 — that iteration covered episode steps 272 to 400 and then 0 to 128 of the next
        """
        return ((iteration - 1) * config.num_steps) % args.episode_steps

    # the unit's own identity card: which cell each copy belongs to, so the aggregator can group
    # the per-copy arrays of every later record without re-deriving the sweep
    emit({"record": "unit_start", "unit_id": args.unit_id, "bonus": args.bonus,
          "copies": config.n_copies, "envs_per_copy": config.n_envs,
          "rollout_steps": config.num_steps, "iterations": args.iterations,
          "window_iterations": args.window_iterations, "episode_steps": args.episode_steps,
          "episode_clock_iterations": int(clock_iterations),
          "copies_per_cell": config.copies_per_group,
          "cell_settings": [list(setting) for setting in sweep.group_settings],
          "copy_cell": sweep.copy_group.tolist(),
          "copy_seed_index": list(sweep.copy_seed_index),
          # the slice of a chunked run this process holds; 0 and the whole count for an unchunked
          # run, so an aggregator can tell chunks apart and check that they cover the run exactly
          "copy_seed_offset": config.copy_seed_offset,
          "copy_seed_index_first": int(min(sweep.copy_seed_index)),
          "copy_seed_index_last": int(max(sweep.copy_seed_index)),
          "sweep_seed_mode": config.sweep_seed_mode if sweep.is_sweep else "distinct",
          "episodes_per_copy_per_iteration": episodes_per_copy_per_iteration,
          "steps_per_iteration": steps_per_iteration})
    say(f"{config.n_copies} copies x {config.n_envs} environments x {config.num_steps} steps, "
        f"{args.iterations} iterations, update style {config.update_style}, "
        f"{len(sweep.group_settings)} cells of {config.copies_per_group or config.n_copies} copies")

    state = trainer.init_state(run_seed=args.run_seed)
    prime_start = time.time()
    state = trainer.prime(state)
    jax.block_until_ready(state.obs)
    say(f"building and priming done in {time.time() - build_start:.1f}s "
        f"(priming alone {time.time() - prime_start:.1f}s)")

    # the window accumulators live on the device: adding to them is dispatched without waiting, so
    # the loop never synchronises except on the iterations that close a window
    reward_in_window = jnp.zeros((config.n_copies,))
    rint_in_window = jnp.zeros((config.n_copies,))
    start = time.time()
    first_iteration_seconds = None
    window_index = 0
    window_first_iteration = 1
    for iteration in range(1, args.iterations + 1):
        iteration_start = time.time()
        state, metrics = trainer.iterate(state,
                                         trainer.lr_argument(iteration, args.iterations))
        reward_in_window = reward_in_window + metrics["reward_ext_sum"]
        rint_in_window = rint_in_window + metrics["rint_mean"]
        if iteration == 1:
            jax.block_until_ready(metrics["loss"])
            first_iteration_seconds = time.time() - iteration_start
            say(f"first iteration (includes compiling the program) "
                f"{first_iteration_seconds:.1f}s")

        # one record per window, and always one at the last iteration; those are the only
        # iterations that wait for the device, so recording does not serialise the loop
        if iteration % args.window_iterations == 0 or iteration == args.iterations:
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
                "episode_clock_cycles_in_window": iterations_in_window / clock_iterations,
                # a window covering a whole number of clock cycles sees every part of the episode
                # the same number of times, so its reward sum carries no episode-clock artifact;
                # a window that does not is excluded from every score
                "phase_blocked": iterations_in_window % clock_iterations == 0,
                "episodes_per_copy_in_window":
                    iterations_in_window * episodes_per_copy_per_iteration,
                "episodes_per_copy_cumulative": iteration * episodes_per_copy_per_iteration,
                "seconds_since_first_iteration": elapsed,
                "env_steps": iteration * steps_per_iteration,
                "env_steps_per_copy": iteration * steps_per_copy_per_iteration,
                "loss": float(np.asarray(metrics["loss"])),
                "reward_ext_sum_per_copy": reward.tolist(),
                "rint_mean_per_copy":
                    (np.asarray(rint_in_window) / iterations_in_window).tolist(),
            }
            if config.track_coverage:
                record["coverage_per_copy"] = trainer.coverage(state).tolist()
            emit(record)
            coverage_note = (f", maze coverage mean {np.mean(record['coverage_per_copy']):.3f}"
                             if config.track_coverage else "")
            say(f"window {window_index} (iterations {window_first_iteration}-{iteration} of "
                f"{args.iterations}, phase blocked {record['phase_blocked']}) "
                f"loss {record['loss']:.4f}, extrinsic reward per copy summed over the window "
                f"mean {reward.mean():.3f}{coverage_note}, elapsed {elapsed:.1f}s")
            reward_in_window = jnp.zeros((config.n_copies,))
            rint_in_window = jnp.zeros((config.n_copies,))
            window_index += 1
            window_first_iteration = iteration + 1

    jax.block_until_ready(state.agent_params)
    total_seconds = time.time() - start
    # the steady-state rate excludes the first iteration, which paid for compiling the program
    steady_seconds = total_seconds - (first_iteration_seconds or 0.0)
    steady_iterations = max(args.iterations - 1, 1)
    # environments that were respawned for a non-finite physics state (the MJX families
    # count them; other families have no such field)
    nan_count = getattr(state.env_state, "nan_count", None)
    nan_episodes = int(np.asarray(nan_count).sum()) if nan_count is not None else None
    emit({
        "record": "unit_complete",
        "unit_id": args.unit_id,
        "nan_respawned_episodes": nan_episodes,
        "iterations": args.iterations,
        "windows": window_index,
        "env_steps": args.iterations * steps_per_iteration,
        "env_steps_per_copy": args.iterations * steps_per_copy_per_iteration,
        "episodes_per_copy": args.iterations * episodes_per_copy_per_iteration,
        "copies": config.n_copies,
        "seconds_total": total_seconds,
        "seconds_build_and_prime": start - build_start,
        "seconds_first_iteration_with_compile": first_iteration_seconds,
        "seconds_per_iteration_steady": steady_seconds / steady_iterations,
        "attempt_finished": datetime.now().astimezone().isoformat(),
    })
    nan_note = (f", {nan_episodes} episodes respawned for a non-finite physics state"
                if nan_episodes else "")
    say(f"unit {args.unit_id} complete: {args.iterations} iterations in "
        f"{total_seconds:.1f}s{nan_note}")
    out.close()

    # one shard is still a sweep of one: the run-level files are always the aggregate
    sys.path.insert(0, str(Path(__file__).resolve().parent))
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
