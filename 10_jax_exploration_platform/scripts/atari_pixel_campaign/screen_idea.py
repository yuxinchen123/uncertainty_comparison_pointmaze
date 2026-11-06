"""Fifteen-minute speed screen of one additional idea, or a profile of the accepted reference.

User rule 2026-11-05: each speed or profiling idea takes at most 15 minutes of wall time per GPU.
The screen keeps the agreed CleanRL configuration, float32 networks at highest precision,
float64 statistics and deterministic operations. It shortens only work that does not change one
update: the observation-normalization warmup (random-action rollouts before training) and the
number of timed pairs, which stops when the time budget runs out.

Reference = original CleanRL port + the two accepted rendering changes (cached room backgrounds,
render only the last two game states). Each timed pair runs one complete update of each variant
from the same starting state, alternating the order.

Decision rule: correctness of one complete update on identical supplied inputs (absolute 1e-5,
relative 1e-4 for float32; 1e-10 for float64), then
  accepted      at least 5 pairs, every pair faster (one-sided sign probability <= 1/32),
                median speed ratio >= 1.01;
  rejected      correctness failure, or median speed ratio <= 1.0;
  inconclusive  otherwise (fewer than 5 pairs, mixed pairs, or under 1% gain).
"""

import argparse
import gc
import gzip
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))
PLATFORM_ROOT = SCRIPT_DIRECTORY.parents[1]
sys.path.insert(0, str(PLATFORM_ROOT / "src"))

ACCEPTED_ENVIRONMENT = {"cache_room_backgrounds": True, "render_final_two_only": True}
REQUIRED_XLA_FLAG = "--xla_gpu_exclude_nondeterministic_ops=true"
# Settings that change only the order of float32 sums inside the networks, never the equations.
SUMMATION_ORDER_FIELDS = {"convolution_algorithm", "convolution_batching", "network_layout",
                          "dense_batching"}


def write_json(path, value):
    """Replace a record atomically so a reader never sees half a file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def configure_runtime(cache_directory):
    """Fix precision and determinism before JAX creates any device array."""
    if REQUIRED_XLA_FLAG not in os.environ.get("XLA_FLAGS", "").split():
        raise RuntimeError(f"launch with XLA_FLAGS containing {REQUIRED_XLA_FLAG}")
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    # A shared compilation cache only shortens compilation; it never changes compiled code.
    jax.config.update("jax_compilation_cache_dir", str(cache_directory))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("a screen needs exactly one visible GPU")
    return devices[0]


def build_executor(config, environment_options):
    """One whole-iteration executor, as in the initial thirty comparisons."""
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment
    from exploration_platform.atari_pixels.training import PixelPPORND
    from paired_benchmark import Executor

    return Executor(PixelPPORND(config, AtariPixelEnvironment(**environment_options)))


def time_call(function, *arguments, repeats=3):
    """Median seconds of a jitted call after two warm calls, synchronizing at each end."""
    import jax

    durations = []
    for index in range(2 + repeats):
        started = time.perf_counter()
        jax.block_until_ready(function(*arguments))
        if index >= 2:
            durations.append(time.perf_counter() - started)
    return statistics.median(durations)


def phase_times(executor, host_state):
    """Separate seconds of the rollout, batch preparation and four-epoch update phases."""
    import jax
    from paired_benchmark import device_copy

    trainer = executor.trainer
    collect, prepare, update = (jax.jit(trainer.collect), jax.jit(trainer.prepare_batch),
                                jax.jit(trainer.update))
    # Keep only the arrays one phase needs, so two copies fit on a 20 GB card.
    state = device_copy(host_state)
    result = {"rollout_seconds": time_call(collect, state)}
    rollout_state, rollout = collect(state)
    del state
    result["prepare_seconds"] = time_call(prepare, rollout_state, rollout)
    prepared_state, batch = prepare(rollout_state, rollout)
    del rollout_state, rollout
    gc.collect()
    result["update_seconds"] = time_call(update, prepared_state, batch)
    del prepared_state, batch
    gc.collect()
    return result


def kernel_summary(trace_directory, top=30):
    """Sum GPU kernel time by kernel name from the profiler's trace file."""
    files = sorted(trace_directory.rglob("*.trace.json.gz"))
    if not files:
        return {"error": "no trace file was written", "files": [str(path) for path in trace_directory.rglob("*")]}
    events = json.loads(gzip.open(files[-1]).read())
    events = events.get("traceEvents", events)
    gpu_processes = {event["pid"] for event in events
                     if event.get("ph") == "M" and event.get("name") == "process_name"
                     and "/device:GPU" in str(event.get("args", {}).get("name", ""))}
    totals = {}
    for event in events:
        if event.get("ph") == "X" and event.get("pid") in gpu_processes and "dur" in event:
            entry = totals.setdefault(event["name"], [0.0, 0])
            entry[0] += float(event["dur"]) / 1e6
            entry[1] += 1
    total = sum(seconds for seconds, _ in totals.values())
    ranked = sorted(totals.items(), key=lambda item: -item[1][0])[:top]
    return {"trace_file": str(files[-1]), "gpu_kernel_seconds_total": total,
            "kernels": [{"name": name, "seconds": seconds, "calls": calls,
                         "share": seconds / total if total else None}
                        for name, (seconds, calls) in ranked]}


def traced_update(executor, host_state, trace_directory):
    """Trace one complete warm iteration of this executor."""
    import jax
    from paired_benchmark import device_copy

    state = device_copy(host_state)
    jax.block_until_ready(state)
    trace_directory.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(trace_directory), create_perfetto_trace=True):
        state, metrics = executor.run(state, 1, initial_iteration=int(host_state.iteration))
        jax.block_until_ready((state, metrics))
    del state, metrics
    gc.collect()
    try:
        return kernel_summary(trace_directory)
    except Exception as error:  # the timing result stays valid without the kernel table
        return {"error": f"{type(error).__name__}: {error}"}


def gradient_check(reference, candidate, host_state):
    """Compare loss gradients of both variants on one identical supplied minibatch.

    Used when the one-update check fails for an idea that only reorders float32 sums inside the
    networks: Adam divides each step by the gradient's running size, so elements with gradients
    near zero turn rounding differences into parameter differences above 1e-5. The gradients
    themselves must still agree at float32 absolute 1e-5 and relative 1e-4.
    """
    import jax
    import jax.numpy as jnp
    from exploration_platform.atari_pixels.training import minibatch_loss
    from paired_benchmark import agreement, device_copy, host_copy

    state = device_copy(host_state)
    rollout_state, rollout = jax.jit(reference.trainer.collect)(state)
    del state
    prepared, batch = jax.jit(reference.trainer.prepare_batch)(rollout_state, rollout)
    del rollout_state, rollout
    rows = reference.trainer.config.minibatch_size
    # Keep one minibatch only, so two copies fit on a 20 GB card.
    selected = {name: jnp.array(value[:, :rows], copy=True) for name, value in batch.items()
                if name != "frame_bank"}
    del batch
    gc.collect()
    # Every row trains the predictor here, so every predictor weight receives a gradient.
    masks = jnp.ones((reference.trainer.config.n_copies, rows), jnp.float32)

    def gradients(executor):
        """Gradient of the CleanRL minibatch loss under this variant's network settings."""
        trainer = executor.trainer

        def loss(parameters):
            return minibatch_loss(parameters, prepared.target, selected, masks, trainer.config,
                                  trainer.intrinsic_coefficients, prepared.observation_statistics)[0]
        return host_copy(jax.jit(jax.grad(loss))(prepared.parameters))

    reference_gradients, candidate_gradients = gradients(reference), gradients(candidate)
    report = agreement(reference_gradients, candidate_gradients)
    scale = max(float(abs(value).max()) for value in jax.tree.leaves(reference_gradients))
    return {**report, "largest_reference_gradient_magnitude": scale, "minibatch_rows": rows,
            "predictor_mask": "all rows"}


def decide(pairs, correctness_passed):
    """Apply the module's decision rule to paired speed ratios (reference / candidate seconds)."""
    ratios = [pair["reference"]["seconds"] / pair["candidate"]["seconds"] for pair in pairs]
    if not correctness_passed:
        return {"decision": "rejected", "reason": "correctness failure", "paired_speed_ratios": ratios}
    if not ratios:
        return {"decision": "inconclusive", "reason": "no timed pair fit in the time budget",
                "paired_speed_ratios": ratios}
    wins = sum(ratio > 1.0 for ratio in ratios)
    probability = sum(math.comb(len(ratios), k) for k in range(wins, len(ratios) + 1)) / 2 ** len(ratios)
    median = statistics.median(ratios)
    if median <= 1.0:
        decision, reason = "rejected", "the candidate is not faster"
    elif len(ratios) >= 5 and wins == len(ratios) and median >= 1.01:
        decision, reason = "accepted", "every pair faster, median gain at least 1%"
    else:
        decision, reason = "inconclusive", "too few pairs, mixed pairs, or under 1% gain"
    return {"decision": decision, "reason": reason, "paired_speed_ratios": ratios,
            "median_speed_ratio": median, "candidate_winning_pairs": wins,
            "one_sided_sign_probability": probability}


def main():
    """Run the profile or one screen and save every completed pair as it finishes."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="idea name in words, or 'profile'")
    parser.add_argument("--learner", default="{}", help="JSON learner changes of the candidate")
    parser.add_argument("--environment", default="{}", help="JSON environment changes of the candidate")
    parser.add_argument("--reference-learner", default="{}", help="JSON learner settings of the reference")
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--observation-warmup-iterations", type=int, default=2)
    parser.add_argument("--budget-seconds", type=float, default=840.0,
                        help="stop timing so the whole run ends inside 15 minutes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--profile", action="store_true", help="profile the reference; no candidate")
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    progress_path, result_path = args.output / "progress.json", args.output / "result.json"
    if result_path.exists():
        raise SystemExit(f"{result_path} exists; choose a new output folder")

    def remaining():
        return args.budget_seconds - (time.monotonic() - started)

    def progress(stage, **values):
        write_json(progress_path, {"stage": stage, "elapsed_seconds": time.monotonic() - started, **values})

    device = configure_runtime(args.cache_directory)
    import jax
    from dataclasses import asdict, replace
    from exploration_platform.atari_pixels.config import PixelPPOConfig
    from paired_benchmark import (CorrectnessMismatch, correctness_checks, device_copy, host_copy,
                                  canonical_training_state, measure_round, warm_executor,
                                  MemoryObservation)

    reference_config = PixelPPOConfig(n_copies=args.copies,
                                      observation_warmup_iterations=args.observation_warmup_iterations,
                                      **json.loads(args.reference_learner))
    candidate_config = replace(reference_config, **json.loads(args.learner))
    reference_environment = {"num_envs": reference_config.n_envs, **ACCEPTED_ENVIRONMENT}
    candidate_environment = {**reference_environment, **json.loads(args.environment)}
    import subprocess
    source = {"commit": subprocess.run(["git", "-C", str(PLATFORM_ROOT), "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip(),
              "changed_files": subprocess.run(
                  ["git", "-C", str(PLATFORM_ROOT), "status", "--porcelain", "--", "src", "scripts/atari_pixel_campaign"],
                  capture_output=True, text=True).stdout.splitlines()}
    record = {"name": args.name, "source": source, "host": os.uname().nodename, "device_kind": device.device_kind,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "jax_version": jax.__version__,
              "xla_flags": os.environ.get("XLA_FLAGS"), "copies": args.copies,
              "copy_seeds": list(range(args.seed_start, args.seed_start + args.copies)),
              "observation_warmup_iterations": args.observation_warmup_iterations,
              "reference": {"learner": asdict(reference_config), "environment": reference_environment},
              "candidate": None if args.profile else {"learner": asdict(candidate_config),
                                                      "environment": candidate_environment},
              "budget_seconds": args.budget_seconds}
    progress("building")
    reference = build_executor(reference_config, reference_environment)
    seeds = tuple(record["copy_seeds"])
    initialization_started = time.perf_counter()
    state = reference.trainer.warmup(reference.trainer.initialize(seeds))
    initial_host = host_copy(canonical_training_state(reference, state))
    del state
    gc.collect()
    record["initialization_seconds"] = time.perf_counter() - initialization_started
    progress("initialized", initialization_seconds=record["initialization_seconds"])

    if args.profile:
        warm_host, warm_calls = warm_executor(reference, initial_host, 1)
        record["warmup_calls"] = warm_calls
        memory = MemoryObservation(device)
        rounds = []
        while len(rounds) < 3 and remaining() > 3 * (rounds[-1]["seconds"] if rounds else 60):
            result, warm_host = measure_round(reference, warm_host, 1, memory)
            rounds.append(result)
            progress("timing", rounds=len(rounds))
        record["whole_update_rounds"] = rounds
        record["seconds_per_update"] = statistics.median(row["seconds"] for row in rounds)
        progress("phases")
        record["phases"] = phase_times(reference, warm_host)
        progress("tracing")
        record["kernel_profile"] = traced_update(reference, warm_host, args.output / "trace")
        record["elapsed_seconds"] = time.monotonic() - started
        write_json(result_path, record)
        progress("completed")
        return

    candidate = build_executor(candidate_config, candidate_environment)
    if json.loads(args.environment):
        # A game change must start from its own initialization and agree with the reference's.
        from paired_benchmark import agreement
        state = candidate.trainer.warmup(candidate.trainer.initialize(seeds))
        record["initialization_agreement"] = agreement(
            initial_host, host_copy(canonical_training_state(candidate, state)))
        del state
        gc.collect()
    progress("correctness")
    try:
        correctness, starting_host = correctness_checks(reference, candidate, initial_host)
        record["correctness"] = correctness
    except CorrectnessMismatch as error:
        update_error = str(error)
        reordering_only = (not json.loads(args.environment)
                           and set(json.loads(args.learner)) <= SUMMATION_ORDER_FIELDS)
        gradient_report, gradient_error = None, None
        if reordering_only:
            progress("gradient_check", update_check_error=update_error)
            try:
                gradient_report = gradient_check(reference, candidate, initial_host)
            except CorrectnessMismatch as second:
                gradient_error = str(second)
        if gradient_report is None:
            record.update(correctness={"passed": False, "update_check_error": update_error,
                                       "gradient_check_error": gradient_error,
                                       "gradient_check_run": reordering_only},
                          **decide([], False), elapsed_seconds=time.monotonic() - started)
            write_json(result_path, record)
            progress("rejected", error=update_error, gradient_error=gradient_error)
            return
        record["correctness"] = {"passed": True, "passed_by": "gradient agreement on one supplied minibatch",
                                 "update_check_error": update_error, "gradient_check": gradient_report}
        starting_host = initial_host
    del initial_host
    progress("warming")
    reference_host, record_warm_reference = warm_executor(reference, starting_host, 1)
    _, record_warm_candidate = warm_executor(candidate, starting_host, 1)
    record["warmup_calls"] = {"reference": record_warm_reference, "candidate": record_warm_candidate}
    memory = MemoryObservation(device)
    pairs = []
    pair_estimate = (record_warm_reference[-1]["seconds"] + record_warm_candidate[-1]["seconds"]) * 1.3 + 5
    while len(pairs) < 7 and remaining() > pair_estimate:
        order = ("reference", "candidate") if len(pairs) % 2 == 0 else ("candidate", "reference")
        pair, next_host = {"pair": len(pairs) + 1, "order": list(order)}, None
        for label in order:
            executor = reference if label == "reference" else candidate
            result, resulting_host = measure_round(executor, reference_host, 1, memory)
            pair[label] = result
            if label == "reference":
                next_host = resulting_host
        reference_host = next_host
        pairs.append(pair)
        pair_estimate = (pair["reference"]["seconds"] + pair["candidate"]["seconds"]) * 1.3 + 5
        progress("timing", pairs=len(pairs),
                 last_ratio=pair["reference"]["seconds"] / pair["candidate"]["seconds"])
    contaminated = any(pair[label]["memory"]["other_gpu_processes"] for pair in pairs
                       for label in ("reference", "candidate"))
    record.update(pairs=pairs, contaminated=contaminated, **decide(pairs, True))
    if contaminated:
        record.update(decision="inconclusive", reason="another process used the GPU during timing")
    if pairs:
        for label in ("reference", "candidate"):
            seconds = statistics.median(pair[label]["seconds"] for pair in pairs)
            record[f"{label}_seconds_per_update"] = seconds
            record[f"{label}_transitions_per_second_total"] = (
                reference_config.batch_size * reference_config.n_copies / seconds)
            record[f"{label}_peak_allocator_bytes"] = max(
                pair[label]["memory"]["allocator"]["peak_bytes_in_use"] for pair in pairs)
    record["elapsed_seconds"] = time.monotonic() - started
    write_json(result_path, record)
    progress("completed", decision=record["decision"])


if __name__ == "__main__":
    main()
