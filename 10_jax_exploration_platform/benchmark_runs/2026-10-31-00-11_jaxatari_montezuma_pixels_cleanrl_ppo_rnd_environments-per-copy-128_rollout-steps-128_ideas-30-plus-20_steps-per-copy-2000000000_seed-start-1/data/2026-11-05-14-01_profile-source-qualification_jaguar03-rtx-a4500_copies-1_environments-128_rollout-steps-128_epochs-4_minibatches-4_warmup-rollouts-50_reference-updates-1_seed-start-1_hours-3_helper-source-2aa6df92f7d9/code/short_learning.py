"""Produce short independent learning trajectories without claiming learning equivalence.

Preparation and validators import only the standard library. Numerical imports
occur inside the admitted measurement process. This is a component check and
never selects ideas, submits jobs, or contributes final-training transitions.
"""

import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


MINIMUM_UPDATES = 128
CONTRACT = "finite_learning_independent_trajectories_strict_replay_reviewed_comparison"
COMPONENT_KIND = "tiny_pixel_learning_and_latency_component"
COMPONENT_CONTRACT = "tiny_pixel_learning_component_no_formal_evidence"
COMPONENT_RECIPE = {"n_copies": 1, "n_envs": 1, "num_steps": 2, "update_epochs": 1,
    "num_minibatches": 1, "observation_warmup_iterations": 0, "total_timesteps": 2_000_000_000,
    "donate_state": True, "copy_seeds": [11]}
METRICS = {"loss", "policy_loss", "value_loss", "predictor_loss", "entropy", "approximate_kl",
           "gradient_norm", "reward_ext_sum", "intrinsic_reward_mean", "episode_count",
           "episode_return_sum", "steps_per_policy"}
REPLAY_AGREEMENT = {"exact_environment": True, "float32_derived_reward_moments": False}
OBSERVER_AGREEMENT = {"exact_environment": True, "float32_derived_reward_moments": True}
DIAGNOSTIC_FIELDS = {"metrics", "iteration", "optimizer_step", "copy_seeds", "floating_state_nonfinite_count",
    "observation_variance_mean", "reward_variance", "observation_count", "reward_count",
    "boundary_action_histogram", "boundary_action_bins", "boundary_action_unknown_count",
    "boundary_room_histogram", "boundary_room_bins", "boundary_room_unknown_count", "boundary_inventory_sum",
    "environment_key", "frame_count", "step_count", "reset_count"}


def protocol(configurations):
    """Predeclare complete learning and report windows while retaining the full annealing horizon."""
    # Before: groups three and reports ten; after: 150 updates, with a thirty-update replay window.
    multiple = 1
    for configuration in configurations:
        options = configuration["execution"]
        multiple = math.lcm(multiple, options["iterations_per_call"])
        if options["report_readout"] != "none":
            multiple = math.lcm(multiple, options["report_every_iterations"])
    return {"kind": CONTRACT, "minimum_updates": MINIMUM_UPDATES,
            "updates": ((MINIMUM_UPDATES + multiple - 1) // multiple) * multiple,
            "complete_window_updates": multiple, "replay_window_updates": multiple,
            "full_training_horizon": 2_000_000_000, "normalization_warmup_iterations": 50,
            "curves": "every_original_complete_update", "free_trajectories": "independently_advancing",
            "learning_equivalence_established": False, "zero_reward_only_evidence": False,
            "automatic_tolerance_adjustment": False, "independent_comparative_analysis_required": True,
            "boundary_action_bins": list(range(18)), "boundary_room_bins": list(range(33)),
            "boundary_sample_scope": "one_end_of_update_sample_per_environment_lane"}


def latency_protocol(configuration):
    """Predeclare complete selected reporting windows with synchronization after each original update."""
    options = configuration["execution"]
    interval = options["report_every_iterations"] if options["report_readout"] != "none" else 1
    window = math.lcm(options["iterations_per_call"], interval)
    return {"kind": "selected_reporting_with_each_original_update_synchronized", "measurement_windows": 3,
        "complete_window_updates": window, "calls": 3 * window, "updates_per_call": 1,
        "selected_iterations_per_call": options["iterations_per_call"],
        "selected_report_readout": options["report_readout"],
        "selected_report_every_iterations": options["report_every_iterations"],
        "report_rows_per_window": window // interval if options["report_readout"] != "none" else 0,
        "agreement_policy": OBSERVER_AGREEMENT}


def component_protocol(configurations):
    """Describe only the explicitly approved tiny recipe, keeping the formal contract unchanged."""
    for configuration in configurations:
        learner, options = configuration['learner'], configuration['execution']
        if any(type(learner.get(name)) is not type(value) or learner[name] != value
                for name, value in COMPONENT_RECIPE.items() if name != 'copy_seeds'):
            raise ValueError('the component requires its exact approved tiny learner recipe')
        if type(configuration['environment'].get('num_envs')) is not int or configuration['environment']['num_envs'] != 1:
            raise ValueError('the component requires exactly one actual game environment')
        if (options['compilation_mode'] != 'whole_iteration'
                or type(options['iterations_per_call']) is not int or options['iterations_per_call'] not in {1, 3}
                or options['report_readout'] not in {'scheduled', 'per_update', 'none'}
                or options['report_every_iterations'] != 4
                or options['iterations_per_call'] > 1 and options['report_readout'] != 'none'):
            raise ValueError('the component requires valid ungrouped readout or grouped no-readout execution')
    declared = protocol(configurations)
    return {**declared, 'kind': COMPONENT_CONTRACT, 'normalization_warmup_iterations': 0,
        'component_only': True, 'formal_learning_credit': 0, 'case_credit': 0, 'idea_credit': 0,
        'packing_point_credit': 0, 'throughput_credit': 0, 'admission_granted': False}


def checked_formal_protocol(plan):
    """Reject tiny or component plans before a formal producer can import JAX or create a memory observer."""
    configurations = [plan['reference_configuration'], plan['configuration']]
    original = {'n_envs': 128, 'num_steps': 128, 'update_epochs': 4, 'num_minibatches': 4,
                'observation_warmup_iterations': 50, 'total_timesteps': 2_000_000_000}
    if (plan.get('kind') != 'selected_configuration_measurement' or plan.get('mode') != 'learning-check'
            or 'component_recipe' in plan
            or any(type(configuration['learner'].get(name)) is not int or configuration['learner'][name] != value
                   for configuration in configurations for name, value in original.items())
            or any(type(configuration['environment'].get('num_envs')) is not int
                   or configuration['environment']['num_envs'] != 128 for configuration in configurations)
            or plan.get('learning_protocol') != protocol(configurations)):
        raise ValueError('formal short learning requires the original full configuration and fifty normalization warmups')
    return plan['learning_protocol']


def checked_component_protocol(plan):
    """Allow the separate component entry only for its exact kind, recipe, seeds and complete protocol."""
    configurations = [plan['reference_configuration'], plan['configuration']]
    declared = component_protocol(configurations)
    recipe = plan.get('component_recipe')
    if (plan.get('kind') != COMPONENT_KIND or not isinstance(recipe, dict) or set(recipe) != set(COMPONENT_RECIPE)
            or any(type(recipe[name]) is not type(value) or recipe[name] != value for name, value in COMPONENT_RECIPE.items())
            or plan.get('copy_seeds') != [11] or any(type(seed) is not int for seed in plan['copy_seeds'])
            or plan.get('learning_protocol') != declared
            or any(name in plan for name in ('completed_study', 'final_confirmation_protocol', 'completed_registry_sha256'))):
        raise ValueError('the component entry cannot execute a formal or differently prepared plan')
    return declared


def _parameter_changes(before, after, copies):
    """Compare existing host snapshots one tensor and copy at a time without extra GPU allocations."""
    import jax
    import numpy as np

    result = np.zeros((copies,), np.float64)
    for old, new in zip(jax.tree.leaves(before), jax.tree.leaves(after)):
        for copy in range(copies):
            difference = np.asarray(new[copy], dtype=np.float64) - np.asarray(old[copy], dtype=np.float64)
            result[copy] += np.square(difference).sum()
    return result


def _diagnostics(state, metrics, copies, room_count):
    """Retain per-update metrics and explicitly labelled environment-boundary observations."""
    import jax
    import jax.numpy as jnp

    # These are the final lane actions/rooms of an update, not a complete native-frame histogram.
    environment = state.environment
    actions = environment.previous_action.reshape(copies, -1)
    rooms = environment.game.room_id.reshape(copies, -1)
    nonfinite = sum(jnp.logical_not(jnp.isfinite(value)).sum() for value in jax.tree.leaves(state)
                    if value.dtype.kind in "fc")
    return {"metrics": metrics, "iteration": state.iteration,
        "floating_state_nonfinite_count": nonfinite,
        "optimizer_step": state.optimizer.step, "copy_seeds": state.copy_seeds,
        "observation_variance_mean": state.observation_statistics.variance.reshape(copies, -1).mean(axis=1),
        "reward_variance": state.reward_statistics.variance,
        "observation_count": state.observation_statistics.count, "reward_count": state.reward_statistics.count,
        "boundary_action_histogram": jnp.stack([(actions == action).sum(axis=1) for action in range(18)], axis=1),
        "boundary_action_bins": jnp.arange(18, dtype=jnp.int32),
        "boundary_action_unknown_count": ((actions < 0) | (actions >= 18)).sum(axis=1),
        "boundary_room_histogram": jnp.stack([(rooms == room).sum(axis=1) for room in range(room_count)], axis=1),
        "boundary_room_bins": jnp.arange(room_count, dtype=jnp.int32),
        "boundary_room_unknown_count": ((rooms < 0) | (rooms >= room_count)).sum(axis=1),
        "boundary_inventory_sum": environment.game.inventory.reshape(copies, -1, 4).sum(axis=1),
        "environment_key": environment.key, "frame_count": environment.frame_count,
        "step_count": environment.step_count, "reset_count": environment.reset_count}


def _capture_function(executor):
    """Observe every original update while preserving the selected execution grouping and equations."""
    import jax

    # The grouped body remains the frozen trainer.iteration; only its returned diagnostics are retained.
    copies = executor.trainer.config.n_copies
    room_count = int(executor.trainer.environment.game.consts.MAX_ROOMS)
    if room_count != 33 or executor.trainer.config.action_count != 18:
        raise ValueError("the pinned boundary action/room namespace changed")
    if executor.group_size > 1:
        def group(state):
            """Retain all original per-update rows rather than discarding all but the last metric."""
            def body(previous, unused):
                """Execute one unchanged complete PPO and RND update, then observe its result."""
                state, metrics = executor.trainer.iteration(previous)
                return state, _diagnostics(state, metrics, copies, room_count)
            return jax.lax.scan(body, state, None, length=executor.group_size)
        compiled = jax.jit(group, donate_argnums=(0,) if executor.trainer.config.donate_state else ())
        return compiled, compiled

    def one(previous):
        """Keep the selected one-update or separate-phase entry point intact."""
        state, metrics = executor.one(previous)
        row = _diagnostics(state, metrics, copies, room_count)
        return state, jax.tree.map(lambda value: value[None], row)
    return one, None


def _advance(executor, capture, host, updates, paired):
    """Advance one arm from its own state and retain independent buffers for donation and replay."""
    import jax

    state, rows = paired.device_copy(host), []
    for unused in range(updates // executor.group_size):
        state, captured = capture(state)
        state = state._replace(observations=executor.array_copy(state.observations))
        jax.block_until_ready((state, captured))
        values = jax.device_get(captured)
        rows.extend(jax.tree.map(lambda value: value[index], values) for index in range(executor.group_size))
    result = paired.host_copy(state)
    del state, captured
    gc.collect()
    return result, rows


def _json_values(value):
    """Turn one observed array tree into literal JSON while rejecting nonfinite values at serialization."""
    if isinstance(value, dict):
        return {name: _json_values(item) for name, item in value.items()}
    if value is None or type(value) in (int, float, bool, str):
        return value
    return value.tolist()


def _reports(paired, rows, options, initial_iteration):
    """Rebuild the selected chronological reporting windows from every retained original metric."""
    import numpy as np

    if options["report_readout"] == "none":
        return []
    interval, sums, reports = options["report_every_iterations"], None, []
    for index, row in enumerate(rows, 1):
        sums = paired.accumulate_report(sums, row["metrics"], np)
        if index % interval == 0:
            reports.append({"iteration": initial_iteration + index, "window_updates": interval,
                            "metrics": paired.finish_report(sums, interval)})
            sums = None
    if sums is not None:
        raise ValueError("the instrumented learning window ended before a complete selected report")
    return reports


def _observer_agreement(executor, capture, state, updates, paired):
    """Compare the complete observer window against the actual selected Executor.run entry point."""
    import jax

    observed_state, rows = _advance(executor, capture, state, updates, paired)
    actual_state, actual_metrics = executor.run(paired.device_copy(state), updates, initial_iteration=int(state.iteration))
    jax.block_until_ready((actual_state, actual_metrics))
    # This is the existing complete-rollout contract; supplied-input and same-implementation replay stay strict.
    state_report = paired.agreement((observed_state, rows[-1]["metrics"]),
                                    (paired.host_copy(actual_state), paired.host_copy(actual_metrics)), **OBSERVER_AGREEMENT)
    observed_reports = _reports(paired, rows, executor.options, int(state.iteration))
    report_report = paired.agreement(observed_reports, executor.last_reports)
    del actual_state, actual_metrics, observed_state, rows
    gc.collect()
    return {"passed": True, "window_updates": updates,
        "scope": "instrumented_window_versus_actual_selected_Executor_run_same_full_state",
        "state_and_final_metrics": state_report, "selected_reports": report_report,
        "report_rows": len(observed_reports), "automatic_tolerance_adjustment": False,
        "agreement_policy": OBSERVER_AGREEMENT}


def _precision(state, paired):
    """Check actual network, optimizer and running-statistics dtypes and all floating state leaves."""
    import jax
    import numpy as np

    paired.require_finite(state, "short learning full state")
    network = jax.tree.leaves((state.parameters, state.target, state.optimizer.first, state.optimizer.second))
    statistics = jax.tree.leaves((state.observation_statistics, state.reward_statistics))
    if not network or not statistics or any(np.asarray(value).dtype != np.float32 for value in network):
        raise ValueError("short learning networks and optimizer must remain float32")
    if any(np.asarray(value).dtype != np.float64 for value in statistics):
        raise ValueError("short learning running statistics must remain float64")
    return {"all_state_finite": True, "network_dtype": "float32", "statistics_dtype": "float64",
            "network_leaves": len(network), "statistics_leaves": len(statistics),
            "full_state_leaves": len(jax.tree.leaves(state))}


def run(directory, plan, paired, config_type, trainer_type, environment_type, progress, memory_factory):
    """Run only original full-configuration formal learning after its normal producer admission."""
    declared = checked_formal_protocol(plan)
    return _run_with_memory(directory, plan, paired, config_type, trainer_type, environment_type,
                            progress, memory_factory, declared)


def run_component(directory, plan, paired, config_type, trainer_type, environment_type, progress, memory_factory):
    """Exercise the approved tiny diagnostic through an explicit separate entry that grants no formal credit."""
    declared = checked_component_protocol(plan)
    result = _run_with_memory(directory, plan, paired, config_type, trainer_type, environment_type,
                              progress, memory_factory, declared)
    return {**result, 'kind': COMPONENT_KIND, 'component_recipe': COMPONENT_RECIPE,
        'formal_learning_credit': 0, 'case_credit': 0, 'idea_credit': 0, 'packing_point_credit': 0,
        'throughput_credit': 0, 'admission_granted': False}


def _run_with_memory(directory, plan, paired, config_type, trainer_type, environment_type, progress, memory_factory, declared):
    """Retain whole-check physical-memory and competing-process observations even on failure."""
    import jax
    import train_optimized as training

    memory = memory_factory(jax.devices()[0], plan["gpu_uuid"])
    memory.begin()
    try:
        result = _run(directory, plan, paired, config_type, trainer_type, environment_type, progress, declared)
    finally:
        observed = memory.finish()
        training.atomic_json(Path(directory) / "learning_memory.json", observed)
    if observed["other_gpu_processes"]:
        raise ValueError("another process occupied the GPU during the short learning check")
    result["memory"] = observed
    return result


def _run(directory, plan, paired, config_type, trainer_type, environment_type, progress, declared):
    """Collect independent short trajectories, complete curves and strict same-implementation replay."""
    import jax
    import numpy as np
    import train_optimized as training

    configurations = {"reference": plan["reference_configuration"], "candidate": plan["configuration"]}
    if plan.get("learning_protocol") != declared:
        raise ValueError("the short-learning budget or health contract changed after preparation")
    copies, seeds = len(plan["copy_seeds"]), tuple(plan["copy_seeds"])
    executors, captures, states, origins, targets, precisions, warmups, observers, parameters = {}, {}, {}, {}, {}, {}, {}, {}, {}
    curve_path = Path(directory) / "learning_curves.jsonl"
    if curve_path.exists():
        raise ValueError("retain the previous short-learning curves and prepare a fresh unit")

    # Warm compilation on disposable clones; observed free trajectories still start at update zero.
    for arm, configuration in configurations.items():
        progress({"state": "short_learning_initialization", "variant": arm})
        trainer = trainer_type(config_type(**configuration["learner"]), environment_type(**configuration["environment"]))
        executor = paired.Executor(trainer, configuration["execution"])
        state = paired.host_copy(trainer.warmup(trainer.initialize(seeds)))
        precisions[arm] = {"initial": _precision(state, paired)}
        if int(state.iteration) != 0 or not np.array_equal(state.copy_seeds, np.asarray(seeds, dtype=np.uint32)):
            raise ValueError("short learning must start from the original zero-update schedule and exact policy seeds")
        capture, compiled = _capture_function(executor)
        warm = []
        previous_cache = None
        for index in range(10):
            started = time.perf_counter()
            unused, unused_rows = _advance(executor, capture, state, declared["complete_window_updates"], paired)
            size = compiled._cache_size() if compiled is not None else executor.cache_sizes()
            warm.append({"seconds": time.perf_counter() - started, "cache_size": size})
            del unused, unused_rows
            if index >= 1 and size == previous_cache:
                break
            previous_cache = size
        else:
            raise ValueError("short-learning compilation did not stabilize within ten disposable warmup windows")
        unused, normal_warmup = paired.warm_executor(executor, state, declared["complete_window_updates"])
        del unused
        observers[arm] = _observer_agreement(executor, capture, state, declared["complete_window_updates"], paired)
        observers[arm]["selected_executor_warmup"] = normal_warmup
        executors[arm], captures[arm], states[arm] = executor, capture, state
        targets[arm], origins[arm], warmups[arm] = paired.host_copy(state.target), int(state.iteration), warm
        parameters[arm] = state.parameters

    # Each arm keeps its own advancing host state; no candidate update is replaced by reference state.
    replays, replay_boundaries = {}, {}
    report_path = Path(directory) / "learning_reports.jsonl"
    if report_path.exists():
        raise ValueError("retain previous learning reports and prepare a fresh unit")
    window = declared["complete_window_updates"]
    for start in range(0, declared["updates"], window):
        for arm in ("reference", "candidate") if (start // window) % 2 == 0 else ("candidate", "reference"):
            progress({"state": "short_learning", "variant": arm, "completed_updates": start, "target_updates": declared["updates"]})
            if start == declared["updates"] - window:
                replay_boundaries[arm] = states[arm]
            states[arm], rows = _advance(executors[arm], captures[arm], states[arm], window, paired)
            precisions[arm]["latest"] = _precision(states[arm], paired)
            for index, row in enumerate(rows):
                training.append_record(curve_path, {"variant": arm, "update_in_check": start + index + 1,
                    **_json_values(row)})
            for report in _reports(paired, rows, configurations[arm]["execution"], start):
                training.append_record(report_path, {"variant": arm, **_json_values(report)})
            if start == declared["updates"] - window:
                repeated, repeated_rows = _advance(executors[arm], captures[arm], replay_boundaries[arm], window, paired)
                checks = paired.agreement((states[arm], rows), (repeated, repeated_rows), **REPLAY_AGREEMENT)
                replays[arm] = {"passed": True, "window_updates": window,
                    "from_iteration": start, "to_iteration": declared["updates"], "agreement": checks,
                    "scope": "same_implementation_same_full_state_inputs", "agreement_policy": REPLAY_AGREEMENT}
                del repeated, repeated_rows

    # Target parameters are frozen; trajectory comparison itself is left for an independent raw-data review.
    final = {}
    for arm, state in states.items():
        if any(not np.array_equal(old, new) for old, new in zip(jax.tree.leaves(targets[arm]), jax.tree.leaves(state.target))):
            raise ValueError("short learning changed the frozen RND target")
        final[arm] = {"completed_updates": int(state.iteration) - origins[arm],
            "optimizer_step": np.asarray(state.optimizer.step).tolist(), "frozen_target_unchanged": True,
            "precision": precisions[arm], "replay": replays[arm], "observer_agreement": observers[arm],
            "parameter_change_scope": "initial_versus_final_existing_host_snapshots",
            "agent_change_squared": _parameter_changes(parameters[arm]["agent"], state.parameters["agent"], copies).tolist(),
            "predictor_change_squared": _parameter_changes(parameters[arm]["predictor"], state.parameters["predictor"], copies).tolist()}
    return {"completed": True, "learning_protocol": declared, "reference": configurations["reference"],
        "candidate": configurations["candidate"], "copies": copies, "training_start_iteration": origins,
        "learning_equivalence_established": False, "zero_reward_only_evidence": False,
        "independent_comparative_analysis_required": True, "arms": final, "warmup": warmups,
        "curves": {"path": str(curve_path), "sha256": hashlib.sha256(curve_path.read_bytes()).hexdigest()},
        "reports": None if not report_path.exists() else {"path": str(report_path), "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()},
        "diagnostic_instrumentation": "all original per-update metrics retained; no timing or throughput claim",
        "hypothesis_count_contribution": 0, "comparison_count_contribution": 0, "final_training_credit": 0}


def synchronized_report_window(executor, state, updates, paired, *, device_get, synchronize,
                               device_arrays, host_arrays, clock):
    """Retain the selected accumulator across single-update calls, including each actual readout in timing."""
    options = executor.options
    interval, readout = options["report_every_iterations"], options["report_readout"]
    if executor.group_size != 1 or updates < 1 or readout != "none" and updates % interval:
        raise ValueError("single-update synchronization requires complete selected report windows")
    initial_iteration = int(state.iteration)
    synchronize(state)
    accumulated, reports, durations = None, [], []
    for index in range(updates):
        started = clock()
        state, metrics = executor.one(state)
        if readout != "none":
            values = device_get(metrics) if readout == "per_update" else metrics
            arrays = host_arrays if readout == "per_update" else device_arrays
            accumulated = paired.accumulate_report(accumulated, values, arrays)
            if (index + 1) % interval == 0:
                window = device_get(accumulated) if readout == "scheduled" else accumulated
                reports.append({"iteration": initial_iteration + index + 1, "window_updates": interval,
                                "metrics": paired.finish_report(window, interval)})
                accumulated = None
        # Scheduled accumulation may still be pending before its later host readout.
        synchronize((state, metrics, accumulated))
        durations.append(clock() - started)
    if accumulated is not None:
        raise ValueError("single-update latency omitted a selected scheduled report")
    return state, metrics, reports, durations


def single_update_latencies(paired, selected, reference, seeds, config_type, trainer_type, environment_type):
    """Measure complete selected reporting windows while synchronizing each original update."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    records = {}
    for arm, configuration in (("reference", reference), ("candidate", selected)):
        trainer = trainer_type(config_type(**configuration["learner"]), environment_type(**configuration["environment"]))
        options = {**configuration["execution"], "iterations_per_call": 1}
        executor = paired.Executor(trainer, options)
        actual = paired.Executor(trainer, configuration["execution"])
        declared = latency_protocol(configuration)
        window = declared["complete_window_updates"]
        host = paired.host_copy(trainer.warmup(trainer.initialize(seeds)))
        unused, warmup = paired.warm_executor(executor, host, window)
        del unused
        unused, selected_warmup = paired.warm_executor(actual, host, window)
        del unused
        durations, checks = [], []
        for unused in range(declared["measurement_windows"]):
            origin = int(host.iteration)
            state, metrics, reports, measured = synchronized_report_window(executor, paired.device_copy(host), window, paired,
                device_get=jax.device_get, synchronize=jax.block_until_ready, device_arrays=jnp, host_arrays=np,
                clock=time.perf_counter)
            observed, observed_metrics = paired.host_copy(state), paired.host_copy(metrics)
            del state, metrics
            expected, expected_metrics = actual.run(paired.device_copy(host), window, initial_iteration=origin)
            jax.block_until_ready((expected, expected_metrics))
            agreement = paired.agreement((observed, observed_metrics),
                (paired.host_copy(expected), paired.host_copy(expected_metrics)), **OBSERVER_AGREEMENT)
            report_agreement = paired.agreement(reports, actual.last_reports)
            checks.append({"passed": True, "from_iteration": origin, "to_iteration": origin + window,
                "state_and_final_metrics": agreement, "selected_reports": report_agreement,
                "report_rows": len(reports), "published_reports": [_json_values(row) for row in reports]})
            durations.extend(measured)
            host = observed
            del expected, expected_metrics, observed_metrics, reports
            gc.collect()
        records[arm] = {"calls": len(durations), "updates_per_call": 1, "protocol": declared,
            "seconds_per_synchronized_call": durations,
            "median_seconds_per_synchronized_call": statistics.median(durations),
            "mean_seconds_per_synchronized_call": sum(durations) / len(durations),
            "median_amortized_seconds_per_update": statistics.median(durations),
            "synchronization_boundary": "each_original_complete_update",
            "selected_iterations_per_call": configuration["execution"]["iterations_per_call"],
            "warmup": warmup, "selected_executor_warmup": selected_warmup, "window_agreement": checks}
        del executor, actual, trainer, host
        jax.clear_caches()
        gc.collect()
    return records
