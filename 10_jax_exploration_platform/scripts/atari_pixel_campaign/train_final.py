"""Train the pixel Montezuma learner for a target number of updates, resumable at any point.

One run folder holds:
  run.json        configuration, seeds, environment settings, source commit (written once)
  checkpoint.pkl  the complete latest state (networks, optimizer, statistics, game and random
                  state, counters), replaced atomically every --checkpoint-seconds, at the target,
                  at the time limit and on SIGTERM / SIGINT
  metrics.jsonl   one line per completed update with each copy's metrics, flushed per line
  progress.json   current update, rate and last checkpoint time
  game_first_times.json, game_windows.jsonl
                  first-time steps of game events and per-window sums, folded from the metric
                  lines by game_records.py (its docstring gives both formats)

Running the same command again resumes from checkpoint.pkl and drops metric lines written after
it, so the record matches an uninterrupted run; the two game record files are rebuilt from the
kept lines. Requires the same deterministic runtime as the screens (XLA_FLAGS containing
--xla_gpu_exclude_nondeterministic_ops=true).

A run folder may be continued on another server: the game is compared by content (version, asset
record, source files), not by where it is stored. A non-finite loss ends the run with an error and
leaves the last checkpoint as it was.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
PLATFORM_ROOT = SCRIPT_DIRECTORY.parents[1]
sys.path.insert(0, str(PLATFORM_ROOT / "src"))

import game_records  # noqa: E402  (standard library only; found through SCRIPT_DIRECTORY)

ACCEPTED_ENVIRONMENT = {"cache_room_backgrounds": True, "render_final_two_only": True}
REQUIRED_XLA_FLAG = "--xla_gpu_exclude_nondeterministic_ops=true"
STOP = {"requested": False}
# The fields of run.json's game_source that decide whether two runs play the same game.
GAME_CONTENT_FIELDS = ("installed_version", "asset_record_sha256", "game_source_sha256")


def write_json(path, value):
    """Replace a small record atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def learner_settings(text):
    """JSON lists become the tuples the frozen configuration expects."""
    values = json.loads(text)
    for name in ("layer_convolution_algorithms",):
        if name in values:
            values[name] = tuple(values[name])
    if "iteration_compiler_options" in values:
        values["iteration_compiler_options"] = tuple(tuple(pair) for pair in values["iteration_compiler_options"])
    return values


def source_files_sha256(package):
    """One hash over every Python file of a package: each file's relative path and its bytes."""
    # before: package/__init__.py, package/games/a.py ; after: sha256 of
    # "__init__.py" 0 <bytes> 0 "games/a.py" 0 <bytes> 0, in sorted path order
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def game_source():
    """The imported JAXtari game by content, with where it is stored as information."""
    from importlib.metadata import version

    import jaxtari
    from jaxtari.paths import get_storage_dir

    package = Path(jaxtari.__file__).resolve().parent
    # The sprite and collision-map pack decides the room geometry, so it is part of the game.
    assets = Path(get_storage_dir()).resolve()
    record = assets / "asset_source.json"
    return {"installed_version": version("JAXtari"),
            "asset_record_sha256": hashlib.sha256(record.read_bytes()).hexdigest() if record.exists() else None,
            "game_source_sha256": source_files_sha256(package),
            "path": str(package), "asset_root": str(assets)}


def source_commit():
    """The commit of the code that runs: a frozen copy's SOURCE_COMMIT file, else the repository's HEAD."""
    # Inside a frozen copy, git would report the HEAD of the repository that encloses the copy.
    marker = PLATFORM_ROOT / "SOURCE_COMMIT"
    if marker.exists():
        return marker.read_text().strip()
    return subprocess.run(["git", "-C", str(PLATFORM_ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def copy_seeds(copies, seed_start, listed):
    """One seed per copy: the listed seeds ("13,13,14,14"), else consecutive seeds from seed_start."""
    if listed is None:
        return tuple(range(seed_start, seed_start + copies))
    seeds = tuple(int(value) for value in listed.split(","))
    if len(seeds) != copies:
        raise SystemExit(f"--copy-seeds needs one seed per copy: {copies} copies, {len(seeds)} seeds given")
    return seeds


def seeds_with_non_finite_loss(losses, seeds):
    """The seeds of the copies whose loss of one update is not a finite number."""
    # before: losses [0.31, nan, 0.28, inf], seeds (13, 14, 15, 16) ; after: [14, 16]
    return [int(seed) for seed, loss in zip(seeds, losses) if not math.isfinite(loss)]


def mismatched_identity(recorded, identity):
    """Names of identity fields that differ from the run folder's record (empty when it may resume).

    The game is compared by its content fields (GAME_CONTENT_FIELDS). Where the package and the
    asset pack are stored may differ, so that a run folder can continue on another server.
    """
    differing = [key for key in identity if key != "game_source" and recorded.get(key) != identity[key]]
    old, new = recorded.get("game_source") or {}, identity["game_source"]
    if any(old.get(field) != new[field] for field in GAME_CONTENT_FIELDS):
        differing.append("game_source")
    return differing


def keep_metrics_through(path, iteration):
    """Drop metric lines after the checkpoint so a resumed record has no duplicates."""
    if not path.exists():
        return 0
    kept = [line for line in path.read_text().splitlines()
            if line.strip() and json.loads(line)["update"] <= iteration]
    path.write_text("".join(line + "\n" for line in kept))
    return len(kept)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--copies", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--copy-seeds", default=None,
                        help="one seed per copy, comma separated (replaces --seed-start), e.g. 13,13,13,13")
    parser.add_argument("--learner", default="{}", help="JSON learner settings")
    parser.add_argument("--environment", default="{}", help="JSON settings added to the accepted environment")
    parser.add_argument("--target-updates", type=int, required=True)
    parser.add_argument("--observation-warmup-iterations", type=int, default=50)
    parser.add_argument("--checkpoint-seconds", type=float, default=1200)
    parser.add_argument("--time-limit-seconds", type=float, default=0, help="stop and checkpoint after this long")
    parser.add_argument("--game-window-updates", type=int, default=game_records.WINDOW_UPDATES,
                        help="updates per line of game_windows.jsonl")
    args = parser.parse_args()
    started = time.monotonic()
    if REQUIRED_XLA_FLAG not in os.environ.get("XLA_FLAGS", "").split():
        raise SystemExit(f"launch with XLA_FLAGS containing {REQUIRED_XLA_FLAG}")
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import numpy as np
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    from exploration_platform.atari_pixels.checkpointing import load_checkpoint
    from exploration_platform.atari_pixels.config import PixelPPOConfig
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment
    from exploration_platform.atari_pixels.training import PixelPPORND
    from run_unit import save_checkpoint

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda *_: STOP.update(requested=True))
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    config = PixelPPOConfig(n_copies=args.copies, observation_warmup_iterations=args.observation_warmup_iterations,
                            **learner_settings(args.learner))
    environment = {**ACCEPTED_ENVIRONMENT, **json.loads(args.environment)}
    seeds = copy_seeds(args.copies, args.seed_start, args.copy_seeds)
    commit = source_commit()
    # The target may be raised by a later invocation; everything else must match the folder.
    identity = {"learner": asdict(config), "environment": environment, "copy_seeds": list(seeds),
                "game_source": game_source()}
    run_record = run_dir / "run.json"
    if run_record.exists():
        recorded = json.loads(run_record.read_text())
        differing = mismatched_identity(recorded, json.loads(json.dumps(identity, default=str)))
        if differing:
            raise SystemExit(f"this run folder belongs to a different configuration: {differing} differ")
    else:
        write_json(run_record, {**identity, "source_commit": commit, "host": os.uname().nodename,
                                "device_kind": jax.devices()[0].device_kind,
                                "xla_flags": os.environ.get("XLA_FLAGS"), "created_unix": time.time()})
    trainer = PixelPPORND(config, AtariPixelEnvironment(num_envs=config.n_envs, **environment))
    checkpoint = run_dir / "checkpoint.pkl"
    metrics_path = run_dir / "metrics.jsonl"
    events = run_dir / "events.jsonl"

    def event(kind, **values):
        with events.open("a") as stream:
            stream.write(json.dumps({"event": kind, "unix": time.time(), "host": os.uname().nodename,
                                     "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                     "source_commit": commit, **values}) + "\n")

    if checkpoint.exists():
        state = load_checkpoint(checkpoint, config, list(seeds), environment)
        kept = keep_metrics_through(metrics_path, int(state.iteration))
        event("resumed", update=int(state.iteration), metric_lines_kept=kept)
    else:
        state = trainer.warmup(trainer.initialize(seeds))
        keep_metrics_through(metrics_path, 0)
        event("started", update=0)
    # Both game record files are rebuilt from the kept metric lines, so they always agree with them.
    records = game_records.rebuild(metrics_path, run_dir, environments_per_policy=config.n_envs,
                                   rollout_steps=config.num_steps, window_updates=args.game_window_updates)
    if records.updates_folded != int(state.iteration):
        raise SystemExit(f"metrics.jsonl holds {records.updates_folded} updates but the state is at "
                         f"update {int(state.iteration)}; the run folder is not consistent")
    first_times_path, windows_path = run_dir / game_records.FIRST_TIMES_FILE, run_dir / game_records.WINDOWS_FILE
    last_checkpoint = time.monotonic()

    def save(reason):
        nonlocal last_checkpoint
        jax.block_until_ready(state)
        save_checkpoint(checkpoint, state, config, list(seeds), environment)
        game_records.write_first_times(first_times_path, records)
        last_checkpoint = time.monotonic()
        event("checkpoint", update=int(state.iteration), reason=reason)

    window_start, window_updates = time.monotonic(), 0
    with metrics_path.open("a") as stream:
        while int(state.iteration) < args.target_updates:
            state, metrics = trainer.train_iteration(state)
            values = jax.device_get(metrics)
            update = int(state.iteration)
            line = {"update": update, "unix": time.time(),
                    **{name: np.asarray(value).tolist() for name, value in values.items()}}
            # A non-finite loss ends the run here: nothing of this update is written, and the
            # checkpoint on disk stays the last one saved before it.
            broken = seeds_with_non_finite_loss(line["loss"], seeds)
            if broken:
                event("non-finite loss", update=update, copy_seeds_with_non_finite_loss=broken)
                raise FloatingPointError(f"update {update}: the loss of the copies with seeds {broken} is not finite")
            stream.write(json.dumps(line) + "\n")
            stream.flush()
            # Fold the line into the game records; a window's line is written when it completes.
            window = records.fold(line)
            if window is not None:
                game_records.append_window(windows_path, window)
            if records.changed:
                game_records.write_first_times(first_times_path, records)
            window_updates += 1
            now = time.monotonic()
            if now - window_start >= 60:
                write_json(run_dir / "progress.json", {
                    "update": update, "target_updates": args.target_updates,
                    "seconds_per_update": (now - window_start) / window_updates,
                    "seconds_since_checkpoint": now - last_checkpoint, "unix": time.time()})
                window_start, window_updates = now, 0
            if STOP["requested"]:
                save("stop signal")
                return
            if args.time_limit_seconds and now - started >= args.time_limit_seconds:
                save("time limit")
                return
            if now - last_checkpoint >= args.checkpoint_seconds:
                save("interval")
    save("target reached")
    write_json(run_dir / "progress.json", {"update": int(state.iteration), "target_updates": args.target_updates,
                                           "completed": True, "unix": time.time()})
    event("completed", update=int(state.iteration))


if __name__ == "__main__":
    main()
