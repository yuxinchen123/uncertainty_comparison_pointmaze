"""Run one reproducible pixel Atari check, measurement, or checkpointed training unit."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

PLATFORM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLATFORM_ROOT / "src"))


def atomic_json(path, value):
    """Replace a small status record only after its complete contents reach disk."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_record(path, value):
    """Append one independently readable measurement or progress record."""
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


def hardware_record():
    """Record the actual visible GPU and driver rather than relying on a scheduler label."""
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version",
         "--format=csv,noheader"], capture_output=True, text=True, check=True)
    return {"hostname": os.uname().nodename, "gpu_query": query.stdout.strip(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""), "pid": os.getpid()}


def save_checkpoint(path, state, config, copy_seeds):
    """Save all host copies of learner and game state with its exact configuration."""
    import jax
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    payload = {"configuration": asdict(config), "copy_seeds": copy_seeds,
               "state": jax.device_get(state)}
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def parse_arguments():
    """Expose fixed-shape training parameters and separate short-check overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--mode", choices=["smoke", "benchmark", "train"], required=True)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--envs-per-copy", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup-rollouts", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--checkpoint-seconds", type=int, default=1200)
    parser.add_argument("--maximum-seconds", type=float, default=0)
    parser.add_argument("--variant", default="reference")
    return parser.parse_args()


def run(args):
    """Initialize or resume one unit and write progress after every complete update."""
    # Configure precision before importing game or learner modules.
    os.environ["JAX_ENABLE_X64"] = "true"
    import jax
    import numpy as np
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    from exploration_platform.atari_pixels.config import PixelPPOConfig
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment
    from exploration_platform.atari_pixels.training import PixelPPORND

    # Identity files exist before any model or game initialization.
    directory = args.run_dir / "data" / args.unit_id
    directory.mkdir(parents=True, exist_ok=True)
    arguments = {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()}
    atomic_json(directory / "arguments.json", arguments)
    atomic_json(directory / "hardware.json", hardware_record())
    overrides = json.loads(args.configuration.read_text()) if args.configuration else {}
    configuration = {
        "n_copies": args.copies, "n_envs": args.envs_per_copy,
        "num_steps": args.rollout_steps, "update_epochs": args.epochs,
        "num_minibatches": args.minibatches,
        "observation_warmup_iterations": args.warmup_rollouts, **overrides}
    # Before: JSON arrays; after: immutable tuples required by the configuration.
    for name in ("per_copy_learning_rates", "per_copy_intrinsic_coefficients"):
        if name in configuration:
            configuration[name] = tuple(configuration[name])
    config = PixelPPOConfig(**configuration)
    atomic_json(directory / "config_resolved.json", asdict(config))
    seeds = tuple(range(args.seed_start, args.seed_start + args.copies))
    environment = AtariPixelEnvironment(num_envs=args.envs_per_copy)
    trainer = PixelPPORND(config, environment)
    checkpoint = directory / "checkpoints" / "latest.pkl"
    started = time.monotonic()
    stop = {"requested": False}

    def request_stop(signum, frame):
        """Finish the current compiled update before saving an interrupt checkpoint."""
        stop["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)
    progress_path = directory / "progress.json"
    history_path = directory / "metrics.jsonl"
    atomic_json(progress_path, {"state": "initializing", "pid": os.getpid(),
                               "time": datetime.now(timezone.utc).isoformat()})

    # Resume only checkpoints produced with this exact learner configuration.
    if args.mode == "train" and checkpoint.exists():
        with checkpoint.open("rb") as stream:
            saved = pickle.load(stream)
        if saved["configuration"] != asdict(config) or tuple(saved["copy_seeds"]) != seeds:
            raise ValueError("checkpoint configuration or policy seeds do not match this unit")
        state = jax.tree.map(jax.device_put, saved["state"])
    else:
        state = trainer.initialize(seeds)
        jax.block_until_ready(state)
        atomic_json(progress_path, {"state": "normalization_warmup", "pid": os.getpid(),
                                   "time": datetime.now(timezone.utc).isoformat()})
        state = trainer.warmup(state)
        jax.block_until_ready(state)
    initialization_seconds = time.monotonic() - started
    append_record(history_path, {"record": "initialization", "seconds": initialization_seconds,
                                 "variant": args.variant})

    def record_progress(metrics, elapsed, update, target):
        """Store scalar and per-copy diagnostics without retaining rollout images."""
        # Before: device scalars/vectors; after: JSON numbers/lists, one entry per copy.
        host = jax.device_get(metrics)
        values = {name: np.asarray(value).tolist() for name, value in host.items()}
        record = {"record": "update", "update": update, "target_updates": target,
                  "steps_per_copy": update * config.batch_size, "seconds": elapsed,
                  "variant": args.variant, "metrics": values}
        append_record(history_path, record)
        atomic_json(progress_path, {**record, "state": "running", "pid": os.getpid(),
                                   "time": datetime.now(timezone.utc).isoformat()})

    # Benchmark warmup advances real policies and includes both donation compilation passes.
    if args.mode != "train":
        compile_started = time.monotonic()
        for _ in range(2):
            state, metrics = trainer.train_iteration(state)
            jax.block_until_ready(state)
        compilation_seconds = time.monotonic() - compile_started
        append_record(history_path, {"record": "compilation", "seconds": compilation_seconds})
        rounds = 1 if args.mode == "smoke" else args.rounds
        measured = []
        for round_index in range(rounds):
            round_started = time.monotonic()
            for _ in range(args.iterations):
                state, metrics = trainer.train_iteration(state)
            jax.block_until_ready(state)
            seconds = time.monotonic() - round_started
            transitions = args.iterations * config.batch_size * args.copies
            result = {"record": "benchmark", "round": round_index,
                      "seconds": seconds, "transitions": transitions,
                      "transitions_per_second": transitions / seconds,
                      "per_copy_transitions_per_second": transitions / seconds / args.copies,
                      "variant": args.variant, "copies": args.copies,
                      "device_memory": jax.devices()[0].memory_stats()}
            measured.append(result)
            append_record(history_path, result)
            atomic_json(progress_path, {**result, "state": "running", "pid": os.getpid(),
                                       "time": datetime.now(timezone.utc).isoformat()})
        result = {"completed": True, "mode": args.mode, "variant": args.variant,
                  "copies": args.copies, "rounds": measured,
                  "initialization_seconds": initialization_seconds,
                  "compilation_seconds": compilation_seconds}
        atomic_json(directory / "result.json", result)
        atomic_json(progress_path, {"state": "completed", "completed": True,
                                   "rounds": len(measured), "pid": os.getpid()})
        print(json.dumps(result), flush=True)
        return

    # Full runs retain all state and save at interruption or the requested cadence.
    update = int(jax.device_get(state.iteration))
    target = config.total_iterations
    last_checkpoint = time.monotonic()
    while update < target and not stop["requested"]:
        update_started = time.monotonic()
        state, metrics = trainer.train_iteration(state)
        jax.block_until_ready(state)
        update += 1
        record_progress(metrics, time.monotonic() - update_started, update, target)
        if time.monotonic() - last_checkpoint >= args.checkpoint_seconds:
            save_checkpoint(checkpoint, state, config, seeds)
            last_checkpoint = time.monotonic()
        if args.maximum_seconds and time.monotonic() - started >= args.maximum_seconds:
            stop["requested"] = True
    save_checkpoint(checkpoint, state, config, seeds)
    completed = update == target
    atomic_json(directory / "result.json", {
        "completed": completed, "updates": update, "target_updates": target,
        "steps_per_copy": update * config.batch_size, "copies": args.copies,
        "elapsed_seconds": time.monotonic() - started})
    atomic_json(progress_path, {"state": "completed" if completed else "checkpointed",
                               "completed": completed, "update": update,
                               "target_updates": target, "pid": os.getpid()})


if __name__ == "__main__":
    arguments = parse_arguments()
    try:
        run(arguments)
    except Exception as error:
        # Preserve a failure record and propagate the real exception to the scheduler.
        destination = arguments.run_dir / "data" / arguments.unit_id
        destination.mkdir(parents=True, exist_ok=True)
        atomic_json(destination / "failure.json", {"error": repr(error),
                    "time": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()})
        raise
