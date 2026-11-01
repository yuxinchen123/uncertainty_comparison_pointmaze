"""Measure bounded grid PPO execution choices after a fixed-data correctness comparison."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import socket
import statistics
import time

from ideas import resolve_idea


class BenchmarkDeadline(TimeoutError):
    """Stop a measurement before its outer ten-minute process limit."""


def atomic_json(path, document):
    """Publish complete JSON documents without exposing a partially written result."""
    # A sibling rename keeps readers from observing truncated measurement records.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".temporary")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require_compute_node():
    """Reject login-node numerical work before importing JAX or NumPy."""
    # Slurm allocation alone is insufficient if this command is still on a portal host.
    import os
    hostname = socket.gethostname().split(".")[0]
    if hostname.startswith(("portal", "login")) or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("grid numerical benchmarks require an allocated Slurm compute node")


def _alarm(signum, frame):
    """Raise at the internal limit so the outer supervisor can retain the partial record."""
    raise BenchmarkDeadline("the bounded infrastructure measurement reached its execution limit")


def timing_summary(blocks, copies, rollout_steps):
    """Convert synchronized timing blocks into total and per-copy throughput."""
    # Before: three {seconds, iterations} blocks; after: comparable rate and latency scalars.
    if len(blocks) != 3 or min(copies, rollout_steps) < 1:
        raise ValueError("timing requires three blocks, positive copies, and positive rollout length")
    if any(block["seconds"] <= 0 or block["iterations"] < 1 for block in blocks):
        raise ValueError("every timing block must complete a positive number of iterations")
    seconds_per_iteration = statistics.median(block["seconds"] / block["iterations"] for block in blocks)
    total = copies * rollout_steps / seconds_per_iteration
    per_copy = total / copies
    return {"seconds_per_iteration": seconds_per_iteration,
            "steps_per_second_total": total, "steps_per_second_per_copy": per_copy,
            "hours_per_million_steps_per_copy": 1_000_000 / (3600 * per_copy),
            "timing_blocks": blocks}


def _compare_trees(reference, candidate, *, atol=2e-5, rtol=2e-5):
    """Require identical discrete values and bounded floating-point differences."""
    import jax
    import numpy as np
    # Compare complete matching trees, rather than only a final reward scalar.
    left, left_structure = jax.tree.flatten(reference)
    right, right_structure = jax.tree.flatten(candidate)
    if left_structure != right_structure:
        raise AssertionError("reference and candidate state trees have different structures")
    maximum = 0.0
    for index, (first, second) in enumerate(zip(left, right)):
        first, second = np.asarray(first), np.asarray(second)
        if first.shape != second.shape:
            raise AssertionError(f"state leaf {index} has different shapes")
        if first.dtype.kind in "biu":
            if not np.array_equal(first, second):
                raise AssertionError(f"discrete state leaf {index} differs")
        else:
            np.testing.assert_allclose(second, first, atol=atol, rtol=rtol,
                                       err_msg=f"floating state leaf {index}")
            maximum = max(maximum, float(np.max(np.abs(second - first), initial=0)))
    return maximum


def correctness_check(configuration, world, changes):
    """Compare initial state, collected data, fixed targets, and one fixed-data PPO update."""
    import jax
    import numpy as np
    from exploration_platform.grid_initialization.config import GridPPOConfig
    from exploration_platform.grid_initialization.training import (
        GridTrainer, prepare_batch, update_policy)
    # A small shared fixture retains the complete optimizer and rollout semantics.
    reference_config = GridPPOConfig(policy_kind=configuration.policy_kind, n_copies=4,
        hidden_width=configuration.hidden_width, num_steps=16, total_timesteps=1024,
        episode_length=configuration.episode_length)
    candidate_changes = {name: value for name, value in changes.items() if name != "n_copies"}
    candidate_config = replace(reference_config, **candidate_changes)
    fields = np.ones((4, world.state_count), dtype=np.float32)
    seeds = np.arange(4, dtype=np.uint32)
    reference = GridTrainer(reference_config, world.transition_table, world.coordinates, fields)
    candidate = GridTrainer(candidate_config, world.transition_table, world.coordinates, fields)
    reference_state, candidate_state = reference.initialize(seeds), candidate.initialize(seeds)
    maximum = _compare_trees(reference_state, candidate_state)

    # A common rollout isolates data collection before comparing one update on fixed data.
    reference_after, reference_rollout = jax.jit(reference.capture)(reference_state)
    candidate_after, candidate_rollout = jax.jit(candidate.capture)(candidate_state)
    jax.block_until_ready(candidate_rollout)
    maximum = max(maximum, _compare_trees((reference_after, reference_rollout),
                                        (candidate_after, candidate_rollout)))
    prepare_reference = jax.jit(lambda state, rollout: prepare_batch(
        state, rollout, reference.coordinates, reference_config))
    prepare_candidate = jax.jit(lambda state, rollout: prepare_batch(
        state, rollout, candidate.coordinates, candidate_config))
    reference_after, fixed_batch = prepare_reference(reference_after, reference_rollout)
    candidate_after, candidate_batch = prepare_candidate(candidate_after, candidate_rollout)
    maximum = max(maximum, _compare_trees(fixed_batch, candidate_batch))

    # Both optimizer executions consume exactly the reference data and initial parameters.
    update_reference = jax.jit(lambda state, batch: update_policy(
        state, batch, reference.coordinates, reference_config))
    update_candidate = jax.jit(lambda state, batch: update_policy(
        state, batch, candidate.coordinates, candidate_config))
    updated_reference = update_reference(reference_after, fixed_batch)
    updated_candidate = update_candidate(candidate_after, fixed_batch)
    jax.block_until_ready(updated_candidate)
    maximum = max(maximum, _compare_trees(updated_reference, updated_candidate))

    # Copy-count choices additionally prove that changing neighbors preserves the first seeds.
    if "n_copies" in changes:
        larger_config = replace(reference_config, n_copies=8)
        larger = GridTrainer(larger_config, world.transition_table, world.coordinates,
                             np.ones((8, world.state_count), np.float32))
        small_state, small_metrics = reference.train_iteration(reference.initialize(seeds))
        large_state, large_metrics = larger.train_iteration(larger.initialize(np.arange(8, dtype=np.uint32)))
        prefix = lambda tree: jax.tree.map(
            lambda array: array[:4] if array.ndim and array.shape[0] == 8 else array, tree)
        maximum = max(maximum, _compare_trees((small_state, small_metrics),
                                              (prefix(large_state), prefix(large_metrics))))
    return {"passed": True, "maximum_absolute_difference": maximum,
            "absolute_tolerance": 2e-5, "relative_tolerance": 2e-5,
            "fixed_data_optimizer_update": True, "rollout_compared": True,
            "copy_count_isolation_compared": "n_copies" in changes}


def benchmark_case(idea, policy_kind, geometry, *, copies=128, width=32, turns=4,
                   block_seconds=2.0, grouped_updates=1):
    """Compile one case and record three synchronized steady execution blocks."""
    import jax
    import numpy as np
    from exploration_platform.grid_initialization.config import GridPPOConfig
    from exploration_platform.grid_initialization.environments import make_geometry
    from exploration_platform.grid_initialization.training import GridTrainer
    # Build an identical science configuration; only the registered execution choices differ.
    started = time.monotonic()
    world = make_geometry(geometry, width, turns=turns if geometry == "snake" else 0)
    changes = dict(idea["changes"])
    grouping = changes.pop("grouped_updates", grouped_updates)
    configuration = GridPPOConfig(n_copies=copies, policy_kind=policy_kind,
        goal_state=world.goal_state, start_state=world.start_state,
        total_timesteps=2 ** 30, episode_length=2 ** 12)
    configuration = replace(configuration, **changes)
    correctness_started = time.monotonic()
    correctness = correctness_check(configuration, world, changes)
    correctness_seconds = time.monotonic() - correctness_started
    jax.clear_caches()

    # Compilation is measured separately and included in the case's total execution time.
    setup_started = time.monotonic()
    trainer = GridTrainer(configuration, world.transition_table, world.coordinates,
        np.ones((configuration.n_copies, world.state_count), dtype=np.float32))
    state = trainer.initialize(np.arange(configuration.n_copies, dtype=np.uint32))
    state, metrics = trainer.train_iteration(state)
    jax.block_until_ready((state, metrics))
    build_compile_seconds = time.monotonic() - setup_started

    # Grouping is a host dispatch choice and never changes rollout or optimizer order.
    blocks = []
    for _ in range(3):
        block_started, iterations = time.monotonic(), 0
        while time.monotonic() - block_started < block_seconds:
            for _ in range(grouping):
                state, metrics = trainer.train_iteration(state)
            jax.block_until_ready((state, metrics))
            iterations += grouping
        blocks.append({"seconds": time.monotonic() - block_started, "iterations": iterations})

    # Device peaks cover this process and are labeled cumulative across prior cases.
    device = jax.devices()[0]
    memory = device.memory_stats()
    if memory is None:
        raise RuntimeError("the GPU backend did not expose memory statistics")
    peak = int(memory["peak_bytes_in_use"])
    state_bytes = sum(array.size * array.dtype.itemsize for array in jax.tree.leaves(state))
    record = {"completed": True, "idea": idea["name"], "policy_kind": policy_kind,
        "geometry": geometry, "width": width, "turns": turns if geometry == "snake" else 0,
        "state_count": world.state_count, "copies": configuration.n_copies,
        "configuration": asdict(configuration), "changes": idea["changes"],
        "correctness": correctness, "correctness_seconds": correctness_seconds,
        "build_compile_seconds": build_compile_seconds,
        "peak_device_memory_mb": peak / 2 ** 20,
        "memory_measurement": "process cumulative peak; includes correctness and compiled executables",
        "training_state_memory_mb": state_bytes / 2 ** 20,
        "wall_seconds": time.monotonic() - started,
        **timing_summary(blocks, configuration.n_copies, configuration.num_steps)}
    return record


def run_benchmark(run_dir, idea, *, policy_kind="all", geometry="all", output=None,
                  copies=128, width=32, turns=4, block_seconds=2.0, wall_seconds=570):
    """Measure requested policy/geometry cases while preserving partial bounded results."""
    # Validate scalar controls before creating records or importing a numerical runtime.
    started = time.monotonic()
    if isinstance(idea, str):
        idea = resolve_idea(idea)
    if not 0 < wall_seconds < 590 or block_seconds <= 0 or copies < 1:
        raise ValueError("use positive controls and an internal execution limit below 590 seconds")
    if policy_kind not in {"tabular", "neural", "all"} or geometry not in {"straight", "snake", "all"}:
        raise ValueError("unknown policy or geometry")
    require_compute_node()
    import jax
    if any(device.platform != "gpu" for device in jax.devices()):
        raise RuntimeError("GPU infrastructure measurements require the GPU backend")
    output = Path(output) if output else Path(run_dir) / "data" / "benchmarks" / (
        idea["name"] + "__" + policy_kind + "__" + geometry + ".json")
    record = {"schema_version": 1, "kind": "grid_infrastructure_benchmark",
        "idea": idea["name"], "changes": idea["changes"], "rationale": idea["rationale"],
        "started_at": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
        "device_kind": jax.devices()[0].device_kind, "jax_version": jax.__version__,
        "execution_limit_seconds": wall_seconds, "completed": False, "cases": []}
    atomic_json(output, record)

    # A process supervisor enforces the outer limit even if native compilation blocks Python signals.
    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, max(0.01, wall_seconds - (time.monotonic() - started)))
    try:
        policies = ("tabular", "neural") if policy_kind == "all" else (policy_kind,)
        geometries = ("straight", "snake") if geometry == "all" else (geometry,)
        for selected_policy in policies:
            for selected_geometry in geometries:
                case = benchmark_case(idea, selected_policy, selected_geometry, copies=copies,
                    width=width, turns=turns, block_seconds=block_seconds)
                record["cases"].append(case)
                record["wall_seconds"] = time.monotonic() - started
                atomic_json(output, record)
                jax.clear_caches()
        record["completed"] = True
    except BenchmarkDeadline as error:
        record["failure"] = str(error)
    except AssertionError as error:
        record["failure"] = "correctness comparison failed: " + str(error)
        record["correctness"] = {"passed": False}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        record["wall_seconds"] = time.monotonic() - started
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output, record)
    return record


def main():
    """Read one registered or explicitly supplied adaptive benchmark specification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--idea", required=True)
    parser.add_argument("--idea-spec", type=Path)
    parser.add_argument("--policy-kind", choices=("tabular", "neural", "all"), default="all")
    parser.add_argument("--geometry", choices=("straight", "snake", "all"), default="all")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--copies", type=int, default=128)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--block-seconds", type=float, default=2.0)
    parser.add_argument("--wall-seconds", type=float, default=570)
    args = parser.parse_args()
    specification = json.loads(args.idea_spec.read_text()) if args.idea_spec else resolve_idea(args.idea)
    if specification["name"] != args.idea:
        raise ValueError("the supplied idea name differs from its specification")
    result = run_benchmark(args.run_dir, specification, policy_kind=args.policy_kind,
        geometry=args.geometry, output=args.output, copies=args.copies, width=args.width,
        turns=args.turns, block_seconds=args.block_seconds, wall_seconds=args.wall_seconds)
    print(json.dumps({"idea": args.idea, "completed": result["completed"],
                      "cases": len(result["cases"]), "wall_seconds": result["wall_seconds"]}), flush=True)
    if not result["completed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
