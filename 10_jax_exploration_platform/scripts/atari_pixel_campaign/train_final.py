"""Train the pixel Montezuma learner for a target number of updates, resumable at any point.

One run folder holds:
  run.json        configuration, seeds, environment settings, source commit (written once)
  checkpoint.pkl  the complete latest state (networks, optimizer, statistics, game and random
                  state, counters), replaced atomically every --checkpoint-seconds, at the target,
                  at the time limit and on SIGTERM / SIGINT
  metrics.jsonl   one line per completed update with each copy's metrics, flushed per line
  progress.json   current update, rate and last checkpoint time

Running the same command again resumes from checkpoint.pkl and drops metric lines written after
it, so the record matches an uninterrupted run. Requires the same deterministic runtime as the
screens (XLA_FLAGS containing --xla_gpu_exclude_nondeterministic_ops=true).
"""

import argparse
from dataclasses import asdict
import json
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

ACCEPTED_ENVIRONMENT = {"cache_room_backgrounds": True, "render_final_two_only": True}
REQUIRED_XLA_FLAG = "--xla_gpu_exclude_nondeterministic_ops=true"
STOP = {"requested": False}


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


def game_source():
    """Where the imported JAXtari game comes from: a source tree's commit, or the installed version."""
    import hashlib

    import jaxtari
    from jaxtari.paths import get_storage_dir

    package = Path(jaxtari.__file__).resolve().parent
    # The sprite and collision-map pack decides the room geometry, so it is part of the game.
    assets = Path(get_storage_dir()).resolve()
    record = assets / "asset_source.json"
    asset_identity = {"asset_root": str(assets), "asset_record_sha256":
                      hashlib.sha256(record.read_bytes()).hexdigest() if record.exists() else None}
    # A commit describes the game only when git tracks the package's own files. An installed package
    # inside an ignored environment folder of another repository would report that repository's HEAD.
    tracked = subprocess.run(["git", "-C", str(package), "ls-files", "--error-unmatch", "__init__.py"],
                             capture_output=True, text=True)
    commit = subprocess.run(["git", "-C", str(package), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    if tracked.returncode == 0 and commit.returncode == 0:
        dirty = subprocess.run(["git", "-C", str(package), "status", "--porcelain", "--", "."],
                               capture_output=True, text=True).stdout.strip()
        return {"path": str(package), "commit": commit.stdout.strip(), "uncommitted_changes": bool(dirty),
                **asset_identity}
    from importlib.metadata import version
    return {"path": str(package), "installed_version": version("JAXtari"), **asset_identity}


def mismatched_identity(recorded, identity):
    """Names of identity fields that differ from the run folder's record (empty when it may resume).

    Run folders written before 2026-11-07 recorded, for an installed game package, the HEAD of the
    repository enclosing the environment folder as the game's "commit"; that HEAD moves with
    unrelated commits. Such a record matches when the package path and the asset pack are the same
    and no file of the installed package changed after the run folder was created.
    """
    differing = [key for key in identity if key != "game_source" and recorded.get(key) != identity[key]]
    old, new = recorded.get("game_source") or {}, identity["game_source"]
    if old != new:
        same_place = all(old.get(key) == new.get(key) for key in ("path", "asset_root", "asset_record_sha256"))
        enclosing_commit = "commit" in old and "installed_version" in new
        created = recorded.get("created_unix")
        unchanged = created is not None and max(
            (item.stat().st_mtime for item in Path(new["path"]).rglob("*") if item.is_file()), default=0) <= created
        if not (same_place and enclosing_commit and unchanged):
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
    parser.add_argument("--learner", default="{}", help="JSON learner settings")
    parser.add_argument("--environment", default="{}", help="JSON settings added to the accepted environment")
    parser.add_argument("--target-updates", type=int, required=True)
    parser.add_argument("--observation-warmup-iterations", type=int, default=50)
    parser.add_argument("--checkpoint-seconds", type=float, default=1200)
    parser.add_argument("--time-limit-seconds", type=float, default=0, help="stop and checkpoint after this long")
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
    seeds = tuple(range(args.seed_start, args.seed_start + args.copies))
    commit = subprocess.run(["git", "-C", str(PLATFORM_ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
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
    last_checkpoint = time.monotonic()

    def save(reason):
        nonlocal last_checkpoint
        jax.block_until_ready(state)
        save_checkpoint(checkpoint, state, config, list(seeds), environment)
        last_checkpoint = time.monotonic()
        event("checkpoint", update=int(state.iteration), reason=reason)

    window_start, window_updates = time.monotonic(), 0
    with metrics_path.open("a") as stream:
        while int(state.iteration) < args.target_updates:
            state, metrics = trainer.train_iteration(state)
            values = jax.device_get(metrics)
            update = int(state.iteration)
            stream.write(json.dumps({"update": update, "unix": time.time(),
                                     **{name: np.asarray(value).tolist() for name, value in values.items()}}) + "\n")
            stream.flush()
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
