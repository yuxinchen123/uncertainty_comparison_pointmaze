"""Compare one distinct pixel-Atari implementation with matched reference work.

Initialization and compilation happen outside timing. Every timed pair starts
from the same supplied learner/game state, alternates execution order, and runs
seven rounds of ten complete updates by default. Only one variant's state is
kept on the GPU at a time. This process measures its actual memory use; an
isolated capacity test must still measure a variant in its own fresh process.
"""

import argparse
import csv
from dataclasses import asdict, replace
import gc
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ideas import (IDEAS_BY_NAME, canonical_signature, implementation_status,
                   registry_idea, summarize_pairs, trial_changes)


PLATFORM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLATFORM_ROOT / "src"))
DEFAULT_EXECUTION = {"compilation_mode": "whole_iteration", "iterations_per_call": 1,
                     "report_readout": "none", "report_every_iterations": 10}
PERFORMANCE_FIELDS = {name for entry in IDEAS_BY_NAME.values() for name in entry.learner} | {
    "hoist_critic", "hoist_rollout_log_probabilities", "associative_reward_filter"}
ENVIRONMENT_PERFORMANCE_FIELDS = {name for entry in IDEAS_BY_NAME.values()
                                  for name in entry.environment} | {"reset_only_when_needed"}


class CorrectnessMismatch(ValueError):
    """An execution change disagrees with its supplied reference inputs."""

    def __init__(self, message):
        """Preserve measurement identity when a numerical gate rejects the candidate."""
        super().__init__(message)
        self.context = {}


def atomic_json(path, value):
    """Write complete status/result records without exposing partially written JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def agreement(reference, candidate, *, exact_environment=True,
              float32_derived_reward_moments=False):
    """Compare canonical states exactly for game data and strictly for floating learner data."""
    import jax
    import numpy as np

    reference_leaves, reference_tree = jax.tree_util.tree_flatten_with_path(reference)
    candidate_leaves, candidate_tree = jax.tree_util.tree_flatten_with_path(candidate)
    if reference_tree != candidate_tree:
        raise CorrectnessMismatch("reference and candidate output structures differ")
    report = {"leaves": 0, "largest_float32_absolute_error": 0.0,
              "largest_float64_absolute_error": 0.0,
              "float32_derived_reward_moment_fields": []}
    for (path, expected), (_, actual) in zip(reference_leaves, candidate_leaves):
        name = jax.tree_util.keystr(path)
        expected, actual = np.asarray(expected), np.asarray(actual)
        if expected.shape != actual.shape:
            raise CorrectnessMismatch(f"{name}: shapes differ {expected.shape} / {actual.shape}")
        exact = (expected.dtype.kind in "biu" and actual.dtype.kind in "biu"
                 or exact_environment and ".environment" in name)
        if exact:
            if not np.array_equal(expected, actual):
                raise CorrectnessMismatch(f"{name}: game, pixel, action or integer values differ")
        else:
            if not np.isfinite(expected).all() or not np.isfinite(actual).all():
                raise CorrectnessMismatch(f"{name}: a floating result is not finite")
            is_double = expected.dtype == np.float64 or actual.dtype == np.float64
            derived = (float32_derived_reward_moments and is_double
                       and any(f".reward_statistics.{field}" in name for field in ("mean", "variance")))
            absolute, relative = ((1e-10, 1e-10) if is_double and not derived else (1e-5, 1e-4))
            if derived:
                report["float32_derived_reward_moment_fields"].append(name)
            difference = float(np.max(np.abs(expected - actual))) if expected.size else 0.0
            field = "largest_float64_absolute_error" if is_double else "largest_float32_absolute_error"
            report[field] = max(report[field], difference)
            if not np.allclose(expected, actual, atol=absolute, rtol=relative, equal_nan=False):
                raise CorrectnessMismatch(
                    f"{name}: absolute error {difference:g} exceeds atol={absolute:g}, rtol={relative:g}")
        report["leaves"] += 1
    return report


class Executor:
    """Keep compilation boundaries and iteration grouping distinct from algorithm settings."""

    def __init__(self, trainer, options=None):
        """Build either one iteration, separate phases, or a group of whole iterations."""
        import jax
        import jax.numpy as jnp

        self.trainer = trainer
        self.options = {**DEFAULT_EXECUTION, **(options or {})}
        if set(self.options) != set(DEFAULT_EXECUTION):
            raise ValueError("unknown execution option")
        self.group_size = int(self.options["iterations_per_call"])
        mode = self.options["compilation_mode"]
        if self.group_size < 1 or mode not in ("whole_iteration", "separate_phases"):
            raise ValueError("invalid compilation mode or iteration group size")
        if mode == "separate_phases" and self.group_size != 1:
            raise ValueError("grouping separately compiled phases would fuse those boundaries again")
        if self.options["report_readout"] not in ("none", "per_update", "scheduled"):
            raise ValueError("report_readout must be none, per_update or scheduled")
        if self.options["report_every_iterations"] < 1:
            raise ValueError("report intervals must be positive")
        if self.options["report_readout"] != "none" and self.group_size != 1:
            raise ValueError("the reporting-overhead comparison requires ungrouped updates")
        self.last_reports = []
        self.compiled_functions = []
        if mode == "separate_phases":
            self.collect = jax.jit(trainer.collect)
            self.prepare = jax.jit(trainer.prepare_batch)
            self.update = jax.jit(trainer.update)
            self.compiled_functions = [self.collect, self.prepare, self.update]
            self.one = self._separate_iteration
        else:
            self.one = trainer.train_iteration
            self.compiled_functions = [trainer._compiled_iteration]
        self.group = None
        if self.group_size > 1:
            def run_group(state):
                """Keep every original iteration and schedule position inside the group."""
                def body(current, unused):
                    """Advance one original complete iteration."""
                    return trainer.iteration(current)

                state, metrics = jax.lax.scan(body, state, None, length=self.group_size)
                return state, jax.tree.map(lambda values: values[-1], metrics)

            self.group = jax.jit(run_group, donate_argnums=(0,) if trainer.config.donate_state else ())
            self.compiled_functions = [self.group]
        self.array_copy = lambda values: jnp.array(values, copy=True)

    def _separate_iteration(self, state):
        """Preserve the learner's final metric and counter calculations across separate phases."""
        import jax.numpy as jnp

        iteration = state.iteration
        state, rollout = self.collect(state)
        state, batch = self.prepare(state, rollout)
        state, metrics = self.update(state, batch)
        count = rollout.episode_finished.astype(jnp.int32).sum(axis=(0, 2))
        returns = jnp.where(rollout.episode_finished, rollout.episode_returns, 0.0).sum(axis=(0, 2))
        metrics = {**metrics, "reward_ext_sum": rollout.rewards.sum(axis=(0, 2)),
                   "intrinsic_reward_mean": rollout.intrinsic_rewards.mean(axis=(0, 2)),
                   "episode_count": count, "episode_return_sum": returns,
                   "steps_per_policy": (iteration + 1) * self.trainer.config.batch_size}
        return state._replace(iteration=iteration + 1,
                              observations=self.array_copy(state.observations)), metrics

    def run(self, state, iterations, initial_iteration=0):
        """Execute exactly the requested number of updates, without adding padded iterations."""
        if iterations < 1 or iterations % self.group_size:
            raise ValueError("the timed update count must be positive and divide by iterations_per_call")
        import jax
        import jax.numpy as jnp
        import numpy as np

        readout = self.options["report_readout"]
        interval = self.options["report_every_iterations"]
        accumulated = None
        self.last_reports = []
        for index in range(iterations // self.group_size):
            if self.group is None:
                state, metrics = self.one(state)
            else:
                state, metrics = self.group(state)
                state = state._replace(observations=self.array_copy(state.observations))
            if readout != "none":
                # Before: one metric vector per update; after: exactly one published row
                # per complete ten-update window in either driver implementation.
                values = jax.device_get(metrics) if readout == "per_update" else metrics
                array_library = np if readout == "per_update" else jnp
                accumulated = accumulate_report(accumulated, values, array_library)
                count = index + 1
                if count % interval == 0:
                    window = jax.device_get(accumulated) if readout == "scheduled" else accumulated
                    self.last_reports.append({"iteration": initial_iteration + count,
                                              "window_updates": interval,
                                              "metrics": finish_report(window, interval)})
                    accumulated = None
        return state, metrics

    def cache_sizes(self):
        """Check that the second and subsequent warmups do not compile another signature."""
        return tuple(function._cache_size() for function in self.compiled_functions)


SUM_REPORT_FIELDS = {"reward_ext_sum", "episode_count", "episode_return_sum"}
LAST_REPORT_FIELDS = {"steps_per_policy", "learning_rate"}


def accumulate_report(previous, metrics, array_library):
    """Accumulate the same chronological sums on host or device, retaining counter end values."""
    if previous is None:
        return {name: array_library.array(value, copy=True) for name, value in metrics.items()}
    return {name: value if name in LAST_REPORT_FIELDS else array_library.add(previous[name], value)
            for name, value in metrics.items()}


def finish_report(window, count):
    """Publish sums for episodes/rewards, final counters, and means for loss diagnostics."""
    return {name: value if name in SUM_REPORT_FIELDS | LAST_REPORT_FIELDS else value / count
            for name, value in window.items()}


def device_copy(host_state):
    """Allocate separate buffers for every input leaf, including donation-sensitive observations."""
    import jax
    import jax.numpy as jnp

    return jax.tree.map(lambda value: jnp.array(value, copy=True), host_state)


def host_copy(state):
    """Finish device work and retain only independently owned host arrays between variants."""
    import jax
    import numpy as np

    jax.block_until_ready(state)
    return jax.tree.map(lambda value: np.array(value, copy=True), jax.device_get(state))


def canonical_training_state(executor, state):
    """Compare logical game stacks rather than an optimization's physical circular storage."""
    environment = executor.trainer.environment.canonical_state(state.environment)
    return state._replace(environment=environment)


def canonical_batch(state, batch):
    """Reconstruct the same supplied image batch when only its storage representation differs."""
    from exploration_platform.atari_pixels.state import normalize_frames
    from exploration_platform.atari_pixels.training import gather_indexed_frames

    result = {name: value for name, value in batch.items()
              if name not in {"target_features", "raw_frames", "frame_bank", "frame_lanes", "frame_indices"}}
    if "frame_bank" in batch:
        result["observations"] = gather_indexed_frames(
            batch["frame_bank"], batch["frame_lanes"], batch["frame_indices"])
    if "raw_frames" in batch:
        result["normalized_frames"] = normalize_frames(batch["raw_frames"], state.observation_statistics)
    return result


def check_fixed_inputs(reference, candidate, host_state):
    """Compare preparation and updates using the same supplied rollout, permutations and masks."""
    import jax

    common = device_copy(host_state)
    rollout_state, rollout = jax.jit(reference.trainer.collect)(common)
    rollout_state_host, rollout_host = host_copy(rollout_state), host_copy(rollout)
    del common, rollout_state, rollout
    gc.collect()
    prepared = []
    batches = []
    updates = []
    for executor in (reference, candidate):
        state, batch = jax.jit(executor.trainer.prepare_batch)(
            device_copy(rollout_state_host), device_copy(rollout_host))
        prepared.append(host_copy(state))
        batches.append(host_copy(canonical_batch(state, batch)))
        updated, metrics = jax.jit(executor.trainer.update)(state, batch)
        updates.append(host_copy((updated, metrics)))
        del state, batch, updated, metrics
        gc.collect()
    preparation_report = agreement(prepared[0], prepared[1])
    # Cached target features are an intentional extra field, not a different supplied minibatch.
    common_fields = sorted(set(batches[0]) & set(batches[1]))
    extras = (set(batches[0]) ^ set(batches[1])) - {"target_features"}
    if extras:
        raise CorrectnessMismatch(f"unexpected prepared-batch differences: {sorted(extras)}")
    batch_report = agreement({name: batches[0][name] for name in common_fields},
                             {name: batches[1][name] for name in common_fields},
                             exact_environment=False)
    update_report = agreement(updates[0], updates[1])
    return {"preparation": preparation_report, "fixed_batch": batch_report,
            "fixed_batch_update": update_report}


def correctness_checks(reference, candidate, initial_host):
    """Require fixed-input update agreement and a complete real-game iteration comparison."""
    common_count = max(reference.group_size, candidate.group_size)
    for executor in (reference, candidate):
        if executor.options["report_readout"] != "none":
            common_count = max(common_count, executor.options["report_every_iterations"])
    if common_count % min(reference.group_size, candidate.group_size):
        raise ValueError("reference and candidate execution groups do not share a check length")
    supplied = check_fixed_inputs(reference, candidate, initial_host)
    outputs = []
    for executor in (reference, candidate):
        state, metrics = executor.run(device_copy(initial_host), common_count,
                                       initial_iteration=int(initial_host.iteration))
        outputs.append(host_copy((canonical_training_state(executor, state),
                                  metrics, executor.last_reports)))
        del state, metrics
        gc.collect()
    # Full rollouts may batch the float32 RND network differently. Their reward moments
    # inherit that input roundoff; the identical-rollout checks above still require strict
    # float64 agreement, including for these same mean/variance calculations.
    complete = agreement(outputs[0], outputs[1], float32_derived_reward_moments=True)
    return {"passed": True, "supplied_inputs": supplied, "complete_iterations": common_count,
            "complete_iteration": complete}, outputs[0][0]


def query_gpu_processes():
    """Read actual GPU-process memory and UUIDs for contamination and memory records."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=10)
    rows = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 3:
            raise RuntimeError(f"unexpected nvidia-smi process row: {row!r}")
        pid, uuid, memory = (value.strip() for value in row)
        if not memory.isdigit():
            raise RuntimeError(f"GPU memory is unavailable for process {pid}: {memory}")
        rows.append({"pid": int(pid), "uuid": uuid, "bytes": int(memory) * 2 ** 20})
    return rows


class MemoryObservation:
    """Sample process allocations while also recording JAX's exact allocator high-water mark."""

    def __init__(self, device):
        """Identify this initialized process's single selected GPU before starting measurements."""
        self.device = device
        current = [row for row in query_gpu_processes() if row["pid"] == os.getpid()]
        uuids = {row["uuid"] for row in current}
        if len(uuids) != 1:
            raise RuntimeError("paired timing requires exactly one initialized GPU per process")
        self.uuid = uuids.pop()
        self.stop = threading.Event()
        self.thread = None
        self.samples = []
        self.errors = []

    def _sample(self):
        """Observe memory and unrelated processes without interfering with their execution."""
        while not self.stop.is_set():
            try:
                self.samples.append((time.monotonic(), [row for row in query_gpu_processes()
                                                        if row["uuid"] == self.uuid]))
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                self.errors.append(str(error))
                return
            self.stop.wait(0.2)

    def begin(self):
        """Start measured sampling outside the timed training window."""
        self.samples = [(time.monotonic(), [row for row in query_gpu_processes()
                                           if row["uuid"] == self.uuid])]
        self.errors = []
        self.stop.clear()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def finish(self):
        """Return measured process samples and allocator statistics without estimating a peak."""
        self.stop.set()
        self.thread.join()
        self.samples.append((time.monotonic(), [row for row in query_gpu_processes()
                                                if row["uuid"] == self.uuid]))
        if self.errors:
            raise RuntimeError("GPU memory sampling failed: " + "; ".join(self.errors))
        own = [row["bytes"] for _, rows in self.samples for row in rows if row["pid"] == os.getpid()]
        if not own:
            raise RuntimeError("nvidia-smi did not observe this process's GPU memory during the round")
        others = sorted({row["pid"] for _, rows in self.samples for row in rows
                         if row["pid"] != os.getpid()})
        allocator = self.device.memory_stats()
        if allocator is None or "peak_bytes_in_use" not in allocator:
            raise RuntimeError("the GPU backend did not expose its actual allocator peak")
        return {"gpu_uuid": self.uuid, "sampled_process_peak_bytes": max(own),
                "process_memory_samples": len(self.samples), "sampling_interval_seconds": 0.2,
                "allocator": {name: int(value) for name, value in allocator.items()},
                "other_gpu_processes": others,
                "peak_scope": "process allocations and lifetime allocator high-water mark; "
                              "compiled executables and allocator caches may remain from both variants"}


def warm_executor(executor, initial_host, updates):
    """Warm at least twice, continuing until the compiled-signature count is unchanged."""
    import jax

    state = device_copy(initial_host)
    observations = []
    previous_sizes = None
    initial_iteration = int(initial_host.iteration)
    for index in range(6):
        started = time.perf_counter()
        state, metrics = executor.run(state, updates, initial_iteration=initial_iteration)
        initial_iteration += updates
        jax.block_until_ready((state, metrics))
        sizes = executor.cache_sizes()
        observations.append({"call": index + 1, "seconds": time.perf_counter() - started,
                             "compiled_signatures": list(sizes), "updates": updates})
        if index >= 1 and sizes == previous_sizes:
            host = host_copy(state)
            del state, metrics
            gc.collect()
            return host, observations
        previous_sizes = sizes
    raise RuntimeError("compilation signatures did not stabilize after six warmup calls")


def measure_round(executor, starting_host, iterations, memory):
    """Time complete updates, synchronize only at boundaries, and report actual measured work."""
    import jax

    initial_iteration = int(starting_host.iteration)
    state = device_copy(starting_host)
    jax.block_until_ready(state)
    memory.begin()
    try:
        started = time.perf_counter()
        state, metrics = executor.run(state, iterations, initial_iteration=initial_iteration)
        jax.block_until_ready((state, metrics))
        seconds = time.perf_counter() - started
    finally:
        memory_record = memory.finish()
    config = executor.trainer.config
    transitions = iterations * config.batch_size * config.n_copies
    host = host_copy(state)
    require_finite((host, jax.device_get(metrics)), "timed round")
    result = {"seconds": seconds, "iterations": iterations, "copies": config.n_copies,
              "report_origin_iteration": initial_iteration,
              "environments_per_copy": config.n_envs, "rollout_steps": config.num_steps,
              "training_transitions": transitions,
              "transitions_per_second_total": transitions / seconds,
              "transitions_per_second_per_copy": transitions / seconds / config.n_copies,
              "seconds_per_update": seconds / iterations, "memory": memory_record,
              "published_reports": jax.tree.map(lambda value: value.tolist()
                                                  if hasattr(value, "tolist") else value,
                                                  executor.last_reports)}
    del state, metrics
    gc.collect()
    return result, host


def require_finite(tree, label):
    """Reject non-finite parameters, optimizer state or statistics outside the timed window."""
    import jax
    import numpy as np

    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        array = np.asarray(value)
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise CorrectnessMismatch(f"{label}: non-finite state at {jax.tree_util.keystr(path)}")


def synchronized_latency(executor, starting_host, calls=3):
    """Measure per-call synchronization separately, disclosing any grouped update count."""
    import jax

    state = device_copy(starting_host)
    jax.block_until_ready(state)
    durations = []
    initial_iteration = int(starting_host.iteration)
    for _ in range(calls):
        started = time.perf_counter()
        state, metrics = executor.run(state, executor.group_size, initial_iteration=initial_iteration)
        initial_iteration += executor.group_size
        jax.block_until_ready((state, metrics))
        durations.append(time.perf_counter() - started)
    del state, metrics
    gc.collect()
    return {"calls": calls, "updates_per_call": executor.group_size,
            "seconds_per_synchronized_call": durations,
            "median_seconds_per_synchronized_call": statistics.median(durations),
            "median_amortized_seconds_per_update": statistics.median(durations) / executor.group_size}


def run_paired_benchmark(config, learner_changes=None, environment_changes=None,
                         execution_changes=None, *, rounds=7, iterations=10, seed_start=1,
                         progress=None, reference_environment=None, reference_execution=None,
                         latency_calls=3):
    """Run correctness and alternating matched-work timing for one concrete distinct candidate."""
    import jax
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment
    from exploration_platform.atari_pixels.training import PixelPPORND

    if rounds < 7 or iterations < 1:
        raise ValueError("a comparison needs at least seven rounds and a positive update count")
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("paired measurements require one visible compute GPU")
    device_kind = jax.devices()[0].device_kind
    hardware_class = ("h100" if "H100" in device_kind else
                      "jaguar03_rtx_a4500" if "A4500" in device_kind and os.uname().nodename == "jaguar03"
                      else "unsupported_hardware")
    if hardware_class == "unsupported_hardware":
        raise RuntimeError(f"hardware {os.uname().nodename} / {device_kind} is outside this study")
    if jax.config.jax_default_matmul_precision != "highest" or not jax.config.x64_enabled:
        raise RuntimeError("paired measurements require highest float32 matmul precision and float64 statistics")
    source_commit = subprocess.run(["git", "-C", str(PLATFORM_ROOT.parent), "rev-parse", "HEAD"],
                                   capture_output=True, text=True, check=True).stdout.strip()
    learner_changes, environment_changes = dict(learner_changes or {}), dict(environment_changes or {})
    execution_changes = dict(execution_changes or {})
    if set(learner_changes) - PERFORMANCE_FIELDS:
        raise ValueError("a speed candidate cannot change the CleanRL algorithm or its copy count")
    if set(environment_changes) - ENVIRONMENT_PERFORMANCE_FIELDS:
        raise ValueError("a speed candidate cannot change the game protocol or preprocessing equations")
    if latency_calls < 1:
        raise ValueError("synchronized latency needs at least one separate call")
    environment_options = {"num_envs": config.n_envs, **(reference_environment or {})}
    defaults = {name: parameter.default for name, parameter in
                inspect.signature(AtariPixelEnvironment).parameters.items()
                if parameter.default is not inspect.Parameter.empty}
    reference_environment_options = {**defaults, **environment_options}
    candidate_environment_options = {**reference_environment_options, **environment_changes}
    reference_execution_options = {**DEFAULT_EXECUTION, **(reference_execution or {})}
    candidate_execution_options = {**reference_execution_options, **execution_changes}
    candidate_config = replace(config, **learner_changes)
    before = canonical_signature(asdict(config), reference_environment_options, reference_execution_options)
    after = canonical_signature(asdict(candidate_config), candidate_environment_options, candidate_execution_options)
    if before == after:
        raise ValueError("candidate equals reference; an unchanged configuration is not another experiment")
    seeds = tuple(range(seed_start, seed_start + config.n_copies))
    build_started = time.perf_counter()
    reference = Executor(PixelPPORND(config, AtariPixelEnvironment(**reference_environment_options)),
                         reference_execution_options)
    candidate = Executor(PixelPPORND(candidate_config, AtariPixelEnvironment(**candidate_environment_options)),
                         candidate_execution_options)
    for executor in (reference, candidate):
        if iterations % executor.group_size:
            raise ValueError("timed updates must divide exactly into each variant's iteration group")
    # Initialize each implementation independently, then use the same reference state for comparison.
    initial_states = []
    initialization = []
    for label, executor in (("reference", reference), ("candidate", candidate)):
        if progress:
            progress({"state": "initializing", "variant": label})
        started = time.perf_counter()
        state = executor.trainer.warmup(executor.trainer.initialize(seeds))
        initial_states.append(host_copy(canonical_training_state(executor, state)))
        initialization.append({"variant": label, "seconds": time.perf_counter() - started})
        del state
        gc.collect()
    try:
        initialization_agreement = agreement(initial_states[0], initial_states[1])
        correctness, starting_host = correctness_checks(reference, candidate, initial_states[0])
    except CorrectnessMismatch as error:
        error.context.update(
            hardware_class=hardware_class, hostname=os.uname().nodename, device_kind=device_kind,
            source_commit=source_commit,
            implementation_distinct=True, correctness_attempted=True, failure_stage="correctness",
            reference={"learner": asdict(config), "environment": reference_environment_options,
                       "execution": reference_execution_options},
            candidate={"learner": asdict(candidate_config), "environment": candidate_environment_options,
                       "execution": candidate_execution_options})
        raise
    del initial_states
    if progress:
        progress({"state": "correctness_passed", "correctness": correctness})
    memory = MemoryObservation(jax.devices()[0])
    warm_updates = max(reference.group_size, candidate.group_size)
    reference_host, reference_warm = warm_executor(reference, starting_host, warm_updates)
    candidate_host, candidate_warm = warm_executor(candidate, starting_host, warm_updates)
    # The number of warmup calls may differ because of donation compilation. Timed pairs use
    # the same reference state regardless, so schedules and supplied parameters still match.
    del candidate_host, starting_host
    paired = []
    for round_index in range(rounds):
        order = ("reference", "candidate") if round_index % 2 == 0 else ("candidate", "reference")
        pair = {"round": round_index + 1, "order": list(order)}
        next_reference_host = None
        for label in order:
            executor = reference if label == "reference" else candidate
            record, resulting_host = measure_round(executor, reference_host, iterations, memory)
            pair[label] = record
            if label == "reference":
                next_reference_host = resulting_host
            if progress:
                progress({"state": "running", "round": round_index + 1,
                          "total_rounds": rounds, "variant": label, **record})
        reference_host = next_reference_host
        paired.append(pair)
    contaminated = any(row[label]["memory"]["other_gpu_processes"]
                       for row in paired for label in ("reference", "candidate"))
    selection = summarize_pairs(paired, correctness_passed=True, contaminated=contaminated)
    selection.update(
        median_candidate_transitions_per_second_total=statistics.median(
            row["candidate"]["transitions_per_second_total"] for row in paired),
        median_reference_transitions_per_second_total=statistics.median(
            row["reference"]["transitions_per_second_total"] for row in paired))
    latency = {"reference": synchronized_latency(reference, reference_host, latency_calls),
               "candidate": synchronized_latency(candidate, reference_host, latency_calls)}
    return {"schema_version": 1, **selection, "implementation_distinct": True,
            "correctness_attempted": True,
            "source_commit": source_commit,
            "copies": config.n_copies,
            "hardware_class": hardware_class, "hostname": os.uname().nodename,
            "device_kind": device_kind, "jax_version": jax.__version__, "copy_seeds": list(seeds),
            "reference": {"learner": asdict(config), "environment": reference_environment_options,
                          "execution": reference_execution_options},
            "candidate": {"learner": asdict(candidate_config), "environment": candidate_environment_options,
                          "execution": candidate_execution_options},
            "correctness": {"initialization": initialization_agreement, **correctness},
            "initialization": initialization,
            "warmup": {"reference": reference_warm, "candidate": candidate_warm},
            "rounds": paired, "synchronized_latency": latency,
            "measurement_scope": ("training_with_report_readout" if any(
                executor.options["report_readout"] != "none" for executor in (reference, candidate))
                                  else "training_iteration_execution"),
            "build_to_completion_seconds": time.perf_counter() - build_started,
            "memory_note": "Actual process memory samples and allocator peaks are recorded. "
                           "Measure isolated capacity in a fresh single-variant process.",
            "game_frames_note": "Training transitions are measured; frame-skip conversions are nominal "
                                "and exclude variable NOOP/FIRE reset overhead."}


def main():
    """Run one catalog hypothesis and atomically preserve negative or successful results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idea", required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--trial-index", type=int)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text()) if args.registry else None
    entry = registry_idea(args.idea, registry)
    availability = implementation_status(entry)
    if availability["state"] != "ready_for_correctness_check":
        raise ValueError(f"{entry.name} is pending implementation: {availability}")
    if not args.output.parent.is_dir():
        raise ValueError("create and record the study output directory before starting the benchmark")
    if args.output.exists():
        raise ValueError("preserve earlier results; choose a new output file for this measurement")
    # Set precision before loading the game or creating any network/device arrays.
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    from exploration_platform.atari_pixels.config import PixelPPOConfig

    overrides = json.loads(args.configuration.read_text()) if args.configuration else {}
    for name in ("per_copy_learning_rates", "per_copy_intrinsic_coefficients"):
        if name in overrides:
            overrides[name] = tuple(overrides[name])
    config = PixelPPOConfig(**{**overrides, "n_copies": args.copies})
    changes = trial_changes(entry, args.trial_index)
    progress_path = args.output.with_suffix(".progress.json")
    try:
        result = run_paired_benchmark(
            config, changes["learner"], changes["environment"], changes["execution"],
            rounds=args.rounds, iterations=args.iterations, seed_start=args.seed_start,
            reference_execution=entry.reference_execution,
            progress=lambda value: atomic_json(progress_path, value))
    except CorrectnessMismatch as error:
        result = {**error.context, "idea": entry.name, "decision": "rejected", "measurement_valid": False,
                  "correctness_passed": False, "error": str(error), "copies": config.n_copies,
                  "parameter_trial_index": args.trial_index}
        atomic_json(args.output, result)
        atomic_json(progress_path, {"state": "rejected", **result})
        raise
    result.update(idea=entry.name, hypothesis=entry.hypothesis,
                  parameter_trial_index=args.trial_index,
                  additional_correctness_checks=list(entry.additional_checks))
    atomic_json(args.output, result)
    atomic_json(progress_path, {"state": "completed", "decision": result["decision"],
                                "copies": config.n_copies, "paired_rounds": len(result["rounds"])})
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
