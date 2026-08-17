"""Run one training configuration and write the platform's standard run folder.

One call = one work unit. A run folder may hold many units (that is what a sweep is); this script
writes one shard, `data/<unit_id>.jsonl`, one line per record, flushed as it goes, and then calls
the aggregator so the run-level `metrics.jsonl` and `summary.json` exist even for a single unit.

The launch-time files (`manifest.yaml`, `config_resolved.yaml`, `command.txt`) are written before
the first iteration, per the `experiment-background` skill. `experiment_background.md` is written by
hand at the same moment and is not this script's job.

No model state is saved: checkpoints are off by default on this platform. A unit that dies part way
therefore cannot resume mid-run — re-running it starts a new attempt, and the aggregator keeps the
last complete attempt.

Example:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python run_training.py \
    --run-dir <runs/...> --unit-id unit_0000 --copies 128 --iterations 200 \
    --update-style full_batch --record-every 10 --track-coverage
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
SPECS = {
    "env": "pointmaze_large_cont400_nonoise@1",
    "agent": "ppo_full_batch@1",
}

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
        "nvidia_smi": (smi.stdout.strip() if smi and smi.returncode == 0
                       else "no nvidia-smi on this host"),
    }


def parse_args() -> argparse.Namespace:
    """Every knob this run exposes; the defaults are the trainer's own."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="the run folder; created if missing")
    parser.add_argument("--unit-id", default="unit_0000", help="names this unit's data shard")
    parser.add_argument("--description", default="", help="the human sentence for the manifest")
    parser.add_argument("--copies", type=int, default=128)
    parser.add_argument("--envs-per-copy", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--update-style", default="full_batch",
                        choices=["full_batch", "epoch_minibatch"])
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--intrinsic-coefficient", type=float, default=1.0)
    parser.add_argument("--bonus", default="rnd_next_state",
                        help="which intrinsic-reward family; see bonuses/registry.py")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--record-every", type=int, default=10,
                        help="write a record every this many iterations")
    parser.add_argument("--track-coverage", action="store_true",
                        help="keep a per-copy visited-cell map on the device")
    return parser.parse_args()


def specs_of(bonus: str) -> dict:
    """The component identities of this run, with the bonus and its update schedule filled in."""
    # an older name for a family records the family's own name, so two runs written against
    # different vocabularies are still comparable by their manifests
    from exploration_platform.bonuses.registry import ALIASES
    name = ALIASES.get(bonus, bonus)
    if name not in UPDATE_SCHEDULES:
        raise ValueError(f"no update schedule recorded for bonus {bonus!r}; add it to "
                         f"UPDATE_SCHEDULES so the run's manifest names one")
    return {**SPECS, "bonus": f"{name}@1", "update_schedule": UPDATE_SCHEDULES[name]}


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
                "environment": {"specification": SPECS["env"],
                                "map_name": "large", "start_cell": "(7, 1)", "goal_cell": "(1, 10)",
                                "position_noise": 0.0, "max_episode_steps": 400,
                                "continuing_task": True, "goal_radius": 0.45,
                                "reward_shift": 0.0},
                "runtime": {"interpreter": sys.executable, "unit_id": args.unit_id,
                            "record_every": args.record_every, "run_seed": args.run_seed}}
    (run_dir / "config_resolved.yaml").write_text(as_yaml(resolved) + "\n")

    # machine-readable identity of the run: which code, which component versions, which shape
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "description": args.description or "one training unit on the JAX exploration platform",
        "git": git_state(PLATFORM_ROOT.parent),
        "specs": specs_of(args.bonus),
        "compile_signature": {
            "copies": config.n_copies, "envs_per_copy": config.n_envs,
            "rollout_steps": config.num_steps, "update_style": config.update_style,
            "epochs": config.update_epochs, "minibatches": config.num_minibatches,
            "dtype": "float32", "stats_dtype": "float64",
        },
        "seeding": {"base_seed": config.base_seed, "run_seed": args.run_seed,
                    "mode": "distinct"},
        "artifacts": {"metrics": "metrics.jsonl", "summary": "summary.json",
                      "shards": "data/*.jsonl"},
    }
    (run_dir / "manifest.yaml").write_text(as_yaml(manifest) + "\n")


def main() -> None:
    """Create the run folder, train, write one shard line per record, then aggregate."""
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    for sub in ("data", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    from exploration_platform.agents.ppo.config import PPOConfig
    from exploration_platform.training.runner import Runner
    import jax

    config = PPOConfig(n_copies=args.copies, n_envs=args.envs_per_copy,
                       num_steps=args.rollout_steps, update_style=args.update_style,
                       learning_rate=args.learning_rate, int_coef=args.intrinsic_coefficient,
                       base_seed=args.base_seed, track_coverage=args.track_coverage)
    write_launch_files(run_dir, args, config)

    shard = run_dir / "data" / f"{args.unit_id}.jsonl"
    log = open(run_dir / "logs" / f"{args.unit_id}.log", "a", buffering=1)

    def say(message: str) -> None:
        """Print to the terminal and to the unit's log, both immediately."""
        line = f"[{pacific_now()}] {message}"
        print(line, flush=True)
        log.write(line + "\n")

    # a shard whose last line says the unit finished is left alone: re-running is then a no-op
    if shard.exists():
        existing = [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]
        if any(record.get("record") == "unit_complete" for record in existing):
            say(f"{args.unit_id} is already complete in {shard}; nothing to do")
            return
        say(f"{args.unit_id} has a partial shard; this is a new attempt appended to it "
            f"(no model state is saved, so the earlier attempt cannot be continued)")

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
    say(f"{config.n_copies} copies x {config.n_envs} environments x {config.num_steps} steps, "
        f"{args.iterations} iterations, update style {config.update_style}")

    # build the trainer, then time the compilation of the first iteration separately from the rest,
    # because the first iteration pays for compiling the whole program
    trainer = Runner(config, bonus=args.bonus)
    state = trainer.init_state(run_seed=args.run_seed)
    prime_start = time.time()
    state = trainer.prime(state)
    jax.block_until_ready(state.obs)
    say(f"priming done in {time.time() - prime_start:.1f}s")

    steps_per_iteration = config.num_steps * config.n_copies * config.n_envs
    start = time.time()
    first_iteration_seconds = None
    for iteration in range(1, args.iterations + 1):
        iteration_start = time.time()
        state, metrics = trainer.iterate(state,
                                         trainer.lr_argument(iteration, args.iterations))
        if iteration == 1:
            jax.block_until_ready(metrics["loss"])
            first_iteration_seconds = time.time() - iteration_start
            say(f"first iteration (includes compiling the program) "
                f"{first_iteration_seconds:.1f}s")

        # a record every `record_every` iterations and always at the end; those are the only
        # iterations that wait for the device, so recording does not serialise the loop
        if iteration % args.record_every == 0 or iteration == args.iterations:
            jax.block_until_ready(metrics["loss"])
            elapsed = time.time() - start
            reward = np.asarray(metrics["reward_ext_sum"])
            record = {
                "record": "iteration",
                "unit_id": args.unit_id,
                "iteration": iteration,
                "seconds_since_first_iteration": elapsed,
                "env_steps": iteration * steps_per_iteration,
                "env_steps_per_copy": iteration * config.num_steps * config.n_envs,
                "loss": float(np.asarray(metrics["loss"])),
                "reward_ext_sum_per_copy": reward.tolist(),
                "rint_mean_per_copy": np.asarray(metrics["rint_mean"]).tolist(),
            }
            if config.track_coverage:
                record["coverage_per_copy"] = trainer.coverage(state).tolist()
            emit(record)
            coverage_note = (f", maze coverage mean {np.mean(record['coverage_per_copy']):.3f}"
                             if config.track_coverage else "")
            say(f"iteration {iteration}/{args.iterations} loss {record['loss']:.4f}, "
                f"extrinsic reward per copy mean {reward.mean():.3f}{coverage_note}, "
                f"elapsed {elapsed:.1f}s")

    jax.block_until_ready(state.agent_params)
    total_seconds = time.time() - start
    # the steady-state rate excludes the first iteration, which paid for compiling the program
    steady_seconds = total_seconds - (first_iteration_seconds or 0.0)
    steady_iterations = max(args.iterations - 1, 1)
    emit({
        "record": "unit_complete",
        "unit_id": args.unit_id,
        "iterations": args.iterations,
        "env_steps": args.iterations * steps_per_iteration,
        "env_steps_per_copy": args.iterations * config.num_steps * config.n_envs,
        "copies": config.n_copies,
        "seconds_total": total_seconds,
        "seconds_first_iteration_with_compile": first_iteration_seconds,
        "seconds_per_iteration_steady": steady_seconds / steady_iterations,
        "attempt_finished": datetime.now().astimezone().isoformat(),
    })
    say(f"unit {args.unit_id} complete: {args.iterations} iterations in {total_seconds:.1f}s")
    out.close()

    # one shard is still a sweep of one: the run-level files are always the aggregate
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from aggregate_run import aggregate
    summary = aggregate(run_dir)
    say(f"aggregated: {summary['records']} metric records, steady rate "
        f"{summary['throughput']['total_env_steps_per_second_steady']:.3e} env steps per second in "
        f"total and {summary['throughput']['env_steps_per_second_per_copy_steady']:.0f} per copy "
        f"({summary['throughput']['hours_per_million_steps_per_copy_steady']:.2f} hours per million "
        f"steps per copy)")
    log.close()


if __name__ == "__main__":
    main()
