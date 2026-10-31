"""Validate post-comparison evidence without selecting ideas, importing JAX or launching work.

The final-verification specification names a study_registry, a source_commit,
a runtime_environment, contexts for h100/jaguar03_rtx_a4500, and a scaling_record.
Each context supplies correctness_record, confirmation_record and packing_records.
Packing records are isolated, fresh-process measurements with full configuration,
hardware/runtime identity and seven rounds of ten complete updates.
"""

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import tempfile


CONTEXTS = ("h100", "jaguar03_rtx_a4500")
EXECUTION_KEYS = {"compilation_mode", "iterations_per_call", "report_readout", "report_every_iterations"}
RUNTIME_KEYS = {"JAX_ENABLE_X64", "JAX_DEFAULT_MATMUL_PRECISION", "XLA_FLAGS"}
ALGORITHM = {"n_envs": 128, "num_steps": 128, "action_count": 18,
    "update_epochs": 4, "num_minibatches": 4, "total_timesteps": 2_000_000_000,
    "learning_rate": 1e-4, "adam_epsilon": 1e-5, "anneal_learning_rate": True,
    "gamma_ext": .999, "gamma_int": .99, "gae_lambda": .95, "clip_coefficient": .1,
    "entropy_coefficient": .001, "value_coefficient": .5, "max_gradient_norm": .5,
    "intrinsic_coefficient": 1., "extrinsic_coefficient": 2., "predictor_update_proportion": .25,
    "observation_warmup_iterations": 50, "upstream_reward_filter_axis": True,
    "warmup_keeps_initial_observation": True}
ENVIRONMENT = {"num_envs": 128, "frame_skip": 4, "max_episode_steps": 108000,
               "noop_max": 30, "episodic_life": True, "sticky_action_probability": .25}
FINAL_FLAGS = ("final_combination_correctness_passed", "final_confirmation_passed",
               "h100_packing_complete", "a4500_packing_complete", "eight_gpu_scaling_complete")
PREGATHER_ASYNC_RETRY_MODULE = "pregather_async_retry.py"


def canonical_digest(value):
    """Identify the complete normalized document rather than a convenient run name."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def retry_validation_module():
    """Load the pinned adjacent retry dependency even when this validator was imported under an alias."""
    path = Path(__file__).resolve().with_name("infrastructure_retry.py")
    specification = importlib.util.spec_from_file_location("pixel_infrastructure_retry_validation", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def pregather_retry_validation_module():
    """Resolve the case-only fifth-attempt metadata validator beside this exact caller."""
    specification = importlib.util.spec_from_file_location("pixel_pregather_async_retry_validation",
        Path(__file__).resolve().with_name(PREGATHER_ASYNC_RETRY_MODULE))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def helper_requirements(required, verified_files):
    """Require the new metadata dependency only when the already verified pinned caller declares it."""
    selected = set(required)
    relative = "10_jax_exploration_platform/scripts/atari_pixel_campaign/" + PREGATHER_ASYNC_RETRY_MODULE
    declares = False
    for name in ("post_comparison_validation.py", "controller.py"):
        path = "10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name
        if path in verified_files:
            tree = ast.parse(verified_files[path])
            declares |= any(isinstance(node, ast.Constant) and node.value == PREGATHER_ASYNC_RETRY_MODULE for node in ast.walk(tree))
    if declares:
        selected.add(relative)
    else:
        selected.discard(relative)
    return selected


class Evidence:
    """Read original bytes once per identity and retain hashes for later admission checks."""

    def __init__(self, resolver=None):
        """Start an empty set of original artifact receipts."""
        if resolver is not None:
            from portable_evidence import ExactPathResolver
            if not isinstance(resolver, ExactPathResolver):
                raise TypeError("portable evidence requires the strict exact-path resolver")
        self.resolver = resolver
        self.receipts = {}
        self.source_defaults = {}
        self.runtime_provenance = {}

    def read(self, path):
        """Require an absolute existing JSON path and reject evidence changing during validation."""
        return json.loads(self.file(path))

    def path_identity(self, path, strict=False):
        """Keep original mapped path identities while preserving local path normalization."""
        if self.resolver is not None:
            from portable_evidence import original_path
            return original_path(path)
        return str(Path(path).resolve(strict=strict))

    def physical_file(self, path):
        """Locate an explicitly pinned file, or the original local file when no mapping is supplied."""
        if self.resolver is not None:
            return self.resolver.file_path(path)
        return Path(path)

    def git_root(self, path, commit):
        """Resolve historical Git locations without changing their recorded logical root."""
        return self.resolver.git_root(path, commit) if self.resolver is not None else Path(path)

    def file(self, path, record=True):
        """Read and hash an original artifact, including source or launch-script evidence."""
        if not Path(path).is_absolute():
            raise ValueError("evidence paths must be absolute: " + str(path))
        identity = self.path_identity(path, strict=True)
        contents = self.resolver.read_bytes(path) if self.resolver is not None else Path(identity).read_bytes()
        receipt = {"path": identity, "sha256": hashlib.sha256(contents).hexdigest()}
        if identity in self.receipts and self.receipts[identity] != receipt:
            raise ValueError("evidence changed while it was being validated: " + str(path))
        if record:
            self.receipts[identity] = receipt
        return contents

    def records(self):
        """Return receipts in a stable order for immutable validation records."""
        return [self.receipts[name] for name in sorted(self.receipts)]


def positive(value, name):
    """Reject missing, Boolean, non-finite and nonpositive measurements."""
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be a finite positive number")
    return value


def runtime_environment(value):
    """Bind full precision and deterministic operation requirements to the actual compiler flags."""
    if not isinstance(value, dict) or set(value) != RUNTIME_KEYS:
        raise ValueError("runtime_environment must contain the three explicit JAX/XLA settings")
    if value["JAX_ENABLE_X64"] != "true" or value["JAX_DEFAULT_MATMUL_PRECISION"] != "highest":
        raise ValueError("final work requires float64 statistics and highest float32 matrix precision")
    flags = shlex.split(value["XLA_FLAGS"])
    deterministic = [flag for flag in flags if flag.startswith("--xla_gpu_exclude_nondeterministic_ops=")]
    layout = [flag for flag in flags if flag.startswith(("--xla_gpu_force_conv_nhwc=", "--xla_gpu_force_conv_nchw="))]
    if (deterministic != ["--xla_gpu_exclude_nondeterministic_ops=true"] or len(layout) > 1
            or any(not flag.endswith("=true") for flag in layout)
            or any("cache" in flag or "autotune_results" in flag for flag in flags)):
        raise ValueError("final evidence must retain determinism and cannot import unvalidated cache/layout settings")
    return {**value, "XLA_FLAGS": shlex.join(flags)}


def configuration(value):
    """Keep every CleanRL and game setting fixed while admitting complete execution settings."""
    if set(value) != {"learner", "environment", "execution"}:
        raise ValueError("configuration requires learner, environment and execution")
    learner, environment, execution = (value[name] for name in ("learner", "environment", "execution"))
    if any(learner.get(name) != expected for name, expected in ALGORITHM.items()):
        raise ValueError("the selected learner differs from the fixed CleanRL algorithm")
    if type(learner.get("n_copies")) is not int or learner["n_copies"] < 1:
        raise ValueError("the selected configuration needs a positive integer copy count")
    if any(learner.get(name) for name in ("per_copy_learning_rates", "per_copy_intrinsic_coefficients", "per_copy_scalar_settings")):
        raise ValueError("this study does not select a hyperparameter search for final training")
    if any(environment.get(name) != expected for name, expected in ENVIRONMENT.items()):
        raise ValueError("the selected environment differs from the fixed pixel protocol")
    if (set(execution) != EXECUTION_KEYS or execution["compilation_mode"] not in {"whole_iteration", "separate_phases"}
            or execution["report_readout"] not in {"none", "per_update", "scheduled"}
            or any(type(execution[name]) is not int or execution[name] < 1
                   for name in ("iterations_per_call", "report_every_iterations"))):
        raise ValueError("the selected execution configuration is incomplete or invalid")
    if execution["iterations_per_call"] != 1 and (execution["compilation_mode"] == "separate_phases"
                                                   or execution["report_readout"] != "none"):
        raise ValueError("this Executor does not combine grouped updates with separate phases or report readout")
    return json.loads(json.dumps(value, allow_nan=False))


def configuration_without_copies(value):
    """Compare packing candidates while changing only the number of independent policies."""
    value = configuration(value)
    value["learner"].pop("n_copies")
    return value


def identity(record, context, source_commit, runtime):
    """Check actual source, GPU type and runtime before using a result in another phase."""
    if context not in CONTEXTS or record.get("hardware_class") != context or record.get("source_commit") != source_commit:
        raise ValueError("evidence source or hardware context differs from its specification")
    if runtime_environment(record.get("runtime_environment")) != runtime:
        raise ValueError("evidence runtime differs from the selected deterministic compiler settings")


def measured_round(row, copies, minimum_updates=10):
    """Check complete matched training work and an observed uncontaminated GPU-memory record."""
    iterations = row.get("iterations")
    if type(iterations) is not int or iterations < minimum_updates:
        raise ValueError("confirmation/capacity requires at least ten complete updates per round")
    if (row.get("copies") != copies or row.get("environments_per_copy") != 128 or row.get("rollout_steps") != 128
            or row.get("training_transitions") != copies * 128 * 128 * iterations):
        raise ValueError("measured work differs from the full-size independent-policy configuration")
    memory = row.get("memory", {})
    if (memory.get("other_gpu_processes") != [] or not isinstance(memory.get("gpu_uuid"), str)
            or not memory["gpu_uuid"].startswith("GPU-")):
        raise ValueError("a measured round lacks an uncontaminated physical-GPU record")
    positive(memory.get("sampled_process_peak_bytes"), "observed GPU memory")
    positive(memory.get("allocator", {}).get("peak_bytes_in_use"), "allocator peak")
    return copies * 128 * 128 * iterations / positive(row.get("seconds"), "measured seconds")


def correctness(record):
    """Require real fixed-input and complete-iteration reports instead of a bare success Boolean."""
    checks = record.get("correctness", {})
    supplied = checks.get("supplied_inputs", {})
    reports = [checks.get("complete_iteration", {})] + [supplied.get(name, {})
               for name in ("preparation", "fixed_batch", "fixed_batch_update")]
    if (record.get("correctness_passed") is not True or checks.get("passed") is not True
            or record.get("correctness_attempted") is not True
            or any(type(report.get("leaves")) is not int or report["leaves"] < 1 for report in reports)):
        raise ValueError("combination evidence lacks actual fixed-input and complete-iteration correctness reports")


def execution_identity(record):
    """Require a real process birth counter and a finite positive recorded execution interval."""
    value = record.get("execution_identity", {})
    if (not isinstance(value.get("hostname"), str) or type(value.get("pid")) is not int or value["pid"] < 1
            or not isinstance(value.get("process_start_time"), str) or not value["process_start_time"].isdigit()):
        raise ValueError("independent executions need recorded host, PID and process birth counter")
    start = positive(value.get("started_at_seconds"), "execution start")
    end = positive(value.get("finished_at_seconds"), "execution finish")
    if end <= start:
        raise ValueError("execution time bounds are reversed")
    return value["hostname"], value["pid"], value["process_start_time"]


def validate_combination(specification, evidence):
    """Require independent correctness and seven longer paired confirmation rounds for the selected combination."""
    first_path, second_path = specification["correctness_record"], specification["confirmation_record"]
    if evidence.path_identity(first_path) == evidence.path_identity(second_path):
        raise ValueError("combination correctness and confirmation require separate recorded executions")
    first, second = evidence.read(first_path), evidence.read(second_path)
    if execution_identity(first) == execution_identity(second):
        raise ValueError("combination correctness and confirmation reused the same process execution")
    context, source = specification["hardware_class"], specification["source_commit"]
    runtime = runtime_environment(specification["runtime_environment"])
    for record in (first, second):
        identity(record, context, source, runtime)
        correctness(record)
    selected = configuration(second["candidate"])
    reference = configuration(second["reference"])
    if (configuration(first["reference"]) != reference or canonical_digest(reference) == canonical_digest(selected)
            or reference["learner"]["n_copies"] != selected["learner"]["n_copies"]
            or any(record.get("implementation_distinct") is not True for record in (first, second))):
        raise ValueError("combination correctness and confirmation need the same distinct full reference")
    if configuration(first["candidate"]) != selected or first.get("copies") != second.get("copies"):
        raise ValueError("the independently checked and confirmed combinations differ")
    copies = selected["learner"]["n_copies"]
    if second.get("copies") != copies or second.get("measurement_valid") is not True or second.get("contaminated") is not False:
        raise ValueError("combination confirmation is not a valid matched measurement")
    rounds = second.get("rounds", [])
    if len(rounds) < 7:
        raise ValueError("combination confirmation requires seven paired rounds")
    ratios = []
    for index, pair in enumerate(rounds):
        expected_order = ["reference", "candidate"] if index % 2 == 0 else ["candidate", "reference"]
        if pair.get("order") != expected_order or pair["reference"].get("iterations") != pair["candidate"].get("iterations"):
            raise ValueError("confirmation order or work is not matched")
        for name in ("reference", "candidate"):
            measured_round(pair[name], copies)
        ratios.append(pair["reference"]["seconds"] / pair["candidate"]["seconds"])
    wins = sum(ratio > 1 for ratio in ratios)
    probability = sum(math.comb(len(ratios), count) for count in range(wins, len(ratios) + 1)) / 2 ** len(ratios)
    if statistics.median(ratios) <= 1 or probability > .05 or second.get("decision") != "accepted":
        raise ValueError("the selected combination has no repeatable longer-window improvement")
    return {"hardware_class": context, "source_commit": source, "configuration": selected,
            "runtime_environment": runtime, "copies": copies, "paired_rounds": len(rounds),
            "median_speedup": statistics.median(ratios), "one_sided_sign_probability": probability}


def packing_point(path, context, source, runtime, evidence):
    """Validate one isolated process, its fixed update window and measured memory headroom."""
    record = evidence.read(path)
    identity(record, context, source, runtime)
    current = configuration(record["configuration"])
    copies = current["learner"]["n_copies"]
    hardware = record.get("hardware", {})
    process = (hardware.get("hostname"), hardware.get("pid"), hardware.get("process_start_time"))
    if (record.get("completed") is not True or record.get("isolated_variant") is not True
            or record.get("normalization_warmup_iterations") != 50
            or not isinstance(process[0], str) or type(process[1]) is not int or not process[2]):
        raise ValueError("packing needs a complete isolated variant initialized in a recorded fresh process")
    rows = list(csv.reader(hardware.get("gpu_query", "").splitlines()))
    if len(rows) != 1 or len(rows[0]) < 3:
        raise ValueError("packing needs the selected GPU's original memory.total query")
    gpu, name, total = [field.strip() for field in rows[0][:3]]
    match = re.fullmatch(r"(\d+) MiB", total)
    if (not match or not gpu.startswith("GPU-") or hardware.get("gpu_uuid") != gpu
            or ("H100" not in name if context == "h100" else "A4500" not in name or process[0] != "jaguar03")):
        raise ValueError("packing hardware UUID, GPU type or actual total memory is inconsistent")
    total_bytes = int(match.group(1)) * 2 ** 20
    positive(total_bytes, "actual total GPU memory")
    rounds = record.get("rounds", [])
    if len(rounds) != 7 or len({row.get("iterations") for row in rounds}) != 1:
        raise ValueError("packing needs seven equal complete-update windows for every point")
    throughput = statistics.median(measured_round(row, copies) for row in rounds)
    if any(row["memory"]["gpu_uuid"] != gpu for row in rounds):
        raise ValueError("packing samples came from a different GPU than the hardware record")
    sampled_peak = max(row["memory"]["sampled_process_peak_bytes"] for row in rounds)
    device_samples = [sample for row in rounds for sample in row["memory"].get("device_memory_samples", [])]
    if any(len(row["memory"].get("device_memory_samples", [])) < 2 for row in rounds):
        raise ValueError("packing needs actual device used/free samples throughout every round")
    for sample in device_samples:
        if (sample.get("gpu_uuid") != gpu or sample.get("total_bytes") != total_bytes
                or type(sample.get("used_bytes")) is not int or type(sample.get("free_bytes")) is not int
                or min(sample["used_bytes"], sample["free_bytes"]) < 0
                or sample["used_bytes"] + sample["free_bytes"] > total_bytes):
            raise ValueError("packing device memory samples disagree with the selected physical GPU")
    free_fraction = min(sample["free_bytes"] / total_bytes for sample in device_samples)
    if free_fraction < .15:
        raise ValueError("packing lacks fifteen percent measured GPU memory headroom")
    seeds = record.get("copy_seeds", [])
    if len(seeds) != copies or len(set(seeds)) != copies or any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in seeds):
        raise ValueError("packing requires the explicit independent policy seeds")
    return {"copies": copies, "throughput": throughput, "configuration": current,
            "process": list(process), "gpu_uuid": gpu, "iterations": rounds[0]["iterations"],
            "copy_seeds": seeds, "sampled_process_peak_bytes": sampled_peak,
            "total_gpu_memory_bytes": total_bytes, "free_memory_fraction": free_fraction}


def validate_packing(specification, evidence):
    """Repeat the first below-ten-percent doubling independently, then choose its smaller count."""
    context, source = specification["hardware_class"], specification["source_commit"]
    runtime = runtime_environment(specification["runtime_environment"])
    points = [packing_point(path, context, source, runtime, evidence) for path in specification["packing_records"]]
    points.sort(key=lambda point: point["copies"])
    if len(points) < 2 or points[0]["copies"] != 1 or any(later["copies"] != 2 * earlier["copies"] for earlier, later in zip(points, points[1:])):
        raise ValueError("packing measurements must double from one without missing a count")
    baseline = configuration_without_copies(points[0]["configuration"])
    processes = set()
    for point in points:
        if (configuration_without_copies(point["configuration"]) != baseline
                or point["gpu_uuid"] != points[0]["gpu_uuid"] or point["iterations"] != points[0]["iterations"]
                or point["copy_seeds"] != list(range(points[0]["copy_seeds"][0], points[0]["copy_seeds"][0] + point["copies"]))):
            raise ValueError("packing changed implementation, GPU, seeds or complete-update window")
        process = tuple(point["process"])
        if process in processes:
            raise ValueError("packing points must use distinct fresh processes")
        processes.add(process)
    gains = [later["throughput"] / earlier["throughput"] - 1 for earlier, later in zip(points, points[1:])]
    below = [index for index, gain in enumerate(gains) if gain < .10]
    if not below:
        return {"complete": False, "hardware_class": context, "next_copies": points[-1]["copies"] * 2,
                "points": points, "marginal_gains": gains}
    if below[0] != len(gains) - 1:
        raise ValueError("packing continued beyond its first below-ten-percent doubling")
    boundary = [points[-2]["copies"], points[-1]["copies"]]
    paths = specification.get("packing_confirmation_records", [])
    if len(paths) != 2:
        return {"complete": False, "hardware_class": context, "next_action": "repeat_boundary_comparison",
                "required_confirmation_counts": boundary, "points": points, "marginal_gains": gains}
    repeated = sorted([packing_point(path, context, source, runtime, evidence) for path in paths], key=lambda point: point["copies"])
    if [point["copies"] for point in repeated] != boundary:
        raise ValueError("the confirmation must repeat both counts at the observed packing limit")
    for point in repeated:
        original = next(prior for prior in points if prior["copies"] == point["copies"])
        if (point["configuration"] != original["configuration"] or point["gpu_uuid"] != original["gpu_uuid"]
                or point["iterations"] != original["iterations"] or point["copy_seeds"] != original["copy_seeds"]
                or tuple(point["process"]) in processes):
            raise ValueError("packing confirmation changed identity/window or reused an earlier process")
        processes.add(tuple(point["process"]))
    repeated_gain = repeated[1]["throughput"] / repeated[0]["throughput"] - 1
    if repeated_gain >= .10:
        return {"complete": False, "hardware_class": context, "next_action": "resolve_disagreeing_packing_measurements",
                "points": points, "marginal_gains": gains, "confirmation": repeated,
                "confirmed_marginal_gain": repeated_gain}
    if type(specification.get("selected_copies")) is not int or specification["selected_copies"] != boundary[0]:
        raise ValueError("the confirmed packing rule selects the smaller copy count")
    return {"complete": True, "hardware_class": context, "source_commit": source,
            "selected_copies": boundary[0], "configuration": points[-2]["configuration"],
            "points": points, "marginal_gains": gains, "confirmation": repeated,
            "confirmed_marginal_gain": repeated_gain, "stop_reason": "confirmed_doubling_gain_below_ten_percent"}


def validate_scaling(specification, evidence):
    """Compare synchronized one-versus-eight A4500 groups with identical work per physical card."""
    single = evidence.read(specification["single_gpu_record"])
    eight = evidence.read(specification["eight_gpu_record"])
    runtime = runtime_environment(specification["runtime_environment"])
    for record in (single, eight):
        identity(record, "jaguar03_rtx_a4500", specification["source_commit"], runtime)
    selected = configuration(single["configuration"])
    if configuration(eight["configuration"]) != selected or single.get("gpu_count") != 1 or eight.get("gpu_count") != 8:
        raise ValueError("scaling changed the selected per-card configuration or GPU counts")
    copies = selected["learner"]["n_copies"]
    first, second = single.get("rounds", []), eight.get("rounds", [])
    if len(first) < 7 or len(first) != len(second):
        raise ValueError("scaling needs at least seven matched wall-time rounds")
    speedups = []
    for one, many in zip(first, second):
        speeds = []
        for row, count in ((one, 1), (many, 8)):
            if row.get("measurement_scope") != "synchronized_gpu_group_wall_time":
                raise ValueError("sum of individual worker rates is not an eight-GPU wall-time measurement")
            duration = positive(row.get("seconds"), "group wall time")
            start, end = row.get("start_time_seconds"), row.get("end_time_seconds")
            if (type(start) not in (int, float) or type(end) not in (int, float) or not math.isfinite(start)
                    or not math.isfinite(end) or abs(end - start - duration) > max(1e-6, duration * 1e-6)):
                raise ValueError("group timing bounds disagree with the measured wall duration")
            workers = row.get("workers", [])
            if len(workers) != count or len({worker.get("gpu_uuid") for worker in workers}) != count:
                raise ValueError("scaling workers must occupy distinct recorded physical GPUs")
            seeds = []
            transitions = 0
            for worker in workers:
                if worker.get("copies") != copies or worker.get("iterations") != one["workers"][0].get("iterations"):
                    raise ValueError("scaling changed the work per GPU")
                measured_round(worker, copies)
                if worker.get("gpu_uuid") != worker["memory"]["gpu_uuid"]:
                    raise ValueError("a scaling worker's physical GPU differs from its memory evidence")
                worker_start = worker.get("start_time_seconds")
                worker_end = worker.get("end_time_seconds")
                if (type(worker_start) not in (int, float) or type(worker_end) not in (int, float)
                        or not math.isfinite(worker_start) or not math.isfinite(worker_end)
                        or worker_start < start or worker_end > end or worker_end <= worker_start
                        or abs(worker_end - worker_start - worker["seconds"]) > max(1e-6, worker["seconds"] * 1e-6)):
                    raise ValueError("the group wall window does not contain every GPU worker")
                current_seeds = worker.get("copy_seeds", [])
                if len(current_seeds) != copies or any(type(seed) is not int or seed < 0 for seed in current_seeds):
                    raise ValueError("scaling workers need explicit independent policy identities")
                seeds.extend(current_seeds)
                transitions += worker["training_transitions"]
            if len(seeds) != len(set(seeds)):
                raise ValueError("the scaling group repeated a policy seed")
            if max(worker["start_time_seconds"] for worker in workers) >= min(worker["end_time_seconds"] for worker in workers):
                raise ValueError("the GPU workers did not execute concurrently within their common wall window")
            speeds.append(transitions / duration)
        if one["workers"][0]["copy_seeds"] != many["workers"][0]["copy_seeds"]:
            raise ValueError("the single-card seed set must also appear in the eight-card group")
        speedups.append(speeds[1] / speeds[0])
    return {"complete": True, "configuration": selected, "rounds": len(first),
            "median_aggregate_speedup": statistics.median(speedups),
            "median_parallel_efficiency": statistics.median(speedups) / 8}


def case_source_defaults(case, evidence):
    """Read defaults from the exact recorded source and keep source-byte receipts without importing it."""
    root, commit = Path(case["source_root"]), case["source_commit"]
    key = (str(root), commit)
    if key in evidence.source_defaults:
        return evidence.source_defaults[key]
    physical_root = evidence.git_root(case["source_root"], commit)
    actual = subprocess.check_output(["git", "-C", str(physical_root), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise ValueError("a historical case source checkout no longer matches its recorded commit")
    files = {}
    for relative in ("src/exploration_platform/atari_pixels/environment.py",
                     "scripts/atari_pixel_campaign/paired_benchmark.py",
                     "scripts/atari_pixel_campaign/worker_environment.sh"):
        path = root / "10_jax_exploration_platform" / relative
        contents = evidence.file(path)
        original = subprocess.check_output(["git", "-C", str(physical_root), "show", commit + ":10_jax_exploration_platform/" + relative])
        if contents != original:
            raise ValueError("a historical case source file changed: " + str(path))
        files[relative] = contents.decode()
    tree = ast.parse(files["src/exploration_platform/atari_pixels/environment.py"])
    constructor = next(node for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "AtariPixelEnvironment"
                       for node in item.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    environment = {argument.arg: ast.literal_eval(value) for argument, value in
                   zip(constructor.args.args[-len(constructor.args.defaults):], constructor.args.defaults)}
    tree = ast.parse(files["scripts/atari_pixel_campaign/paired_benchmark.py"])
    execution = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "DEFAULT_EXECUTION" for target in node.targets))
    answer = environment, execution, files["scripts/atari_pixel_campaign/worker_environment.sh"]
    evidence.source_defaults[key] = answer
    return answer


def cohort_runtime_evidence(case, record, hardware, worker_environment, evidence):
    """Keep actual new runtime records or explicitly infer a legacy protocol from immutable launch/gate evidence."""
    if "runtime_environment" in record:
        return {"scope": "actual_runtime", "runtime_environment": runtime_environment(record["runtime_environment"])}
    run = Path(case["campaign_run_dir"])
    context, commit = case["hardware_class"], case["source_commit"]
    gate = evidence.read(run / "data/source_checks" / commit / (context + ".json"))
    if gate.get("passed") is not True or gate.get("source_commit") != commit or gate.get("hardware_class") != context:
        raise ValueError("legacy case has no matching frozen-source admission gates")
    runtimes = []
    for kind in ("learner", "renderer"):
        receipt = gate["checks"][kind]
        original = evidence.read(receipt["path"])
        if (evidence.receipts[evidence.path_identity(receipt["path"])]["sha256"] != receipt["sha256"]
                or original.get("passed") is not True or original.get("exit_code") != 0
                or original.get("source_commit") != commit):
            raise ValueError("legacy admission check was changed or did not pass at the case source")
        command = original["command"]
        if kind == "learner":
            if len(command) != 6 or command[1:4] != ["-m", "pytest", "-q"]:
                raise ValueError("legacy admission did not run the full numerical test suite")
        elif len(command) != 4 or command[2:] != ["--candidate", "all"]:
            raise ValueError("legacy admission did not check every renderer")
        runtime = original["hardware"]["runtime_environment"]
        query = original["hardware"].get("gpu_query", "")
        if ("H100" not in query if context == "h100" else "A4500" not in query
                or original["hardware"].get("hostname") != "jaguar03"):
            raise ValueError("legacy gate ran on another hardware context")
        runtimes.append(runtime_environment({name: runtime.get(name) for name in RUNTIME_KEYS}))
    if runtimes[0] != runtimes[1] or any(flag not in worker_environment for flag in shlex.split(runtimes[0]["XLA_FLAGS"])):
        raise ValueError("legacy gates and frozen worker runtime declarations differ")
    if 'JAX_ENABLE_X64=true' not in worker_environment or 'JAX_DEFAULT_MATMUL_PRECISION=highest' not in worker_environment:
        raise ValueError("legacy worker source does not retain the admitted precision protocol")
    job = hardware.get("slurm_job_id")
    if job:
        ledger = run / "slurm/submissions.tsv"
        rows = [line.split("\t") for line in evidence.file(ledger, record=False).decode().splitlines()]
        selected_rows = [row for row in rows if len(row) == 4 and row[0] == job]
        matches = [(row[2], row[3]) for row in selected_rows]
    else:
        ledger = run / "processes.json"
        selected_rows = [row for row in json.loads(evidence.file(ledger, record=False)) if row.get("pid") == hardware.get("worker_pid")
                         and row.get("start_time") == hardware.get("worker_start_time")]
        matches = [(row["source_commit"], row["script"]) for row in selected_rows]
    if len(matches) != 1 or matches[0][0] != commit:
        raise ValueError("legacy runtime cannot be linked to exactly one recorded parent launch")
    script = evidence.file(matches[0][1]).decode()
    expected = str(Path(case["source_root"]) / "10_jax_exploration_platform/scripts/atari_pixel_campaign/worker_environment.sh")
    sources = [shlex.split(line) for line in script.splitlines() if line.startswith("source ")]
    if not any(parts == ["source", expected] for parts in sources) or "XLA_FLAGS" in script:
        raise ValueError("legacy launch does not preserve the admitted worker environment")
    return {"scope": "inferred_source_protocol", "runtime_environment": runtimes[0],
            "limitation": "historical process environment was not captured; JAX cache variables are unknown",
            "source_commit": commit, "launch_script": matches[0][1],
            "parent_launch_ledger": str(ledger), "parent_launch_record": selected_rows[0],
            "parent_launch_record_digest": canonical_digest(selected_rows[0]),
            "ledger_scope": "the original selected launch row; later append-only submissions are permitted"}


def validate_cohort_case(path, name, context, copies, source, evidence):
    """Validate a completed immutable case and its actual per-case complete-update timing window."""
    record = evidence.read(path)
    case = evidence.read(Path(path).parent / "case.json")
    hardware = evidence.read(Path(path).parent / "hardware.json")
    case_digest = canonical_digest({key: value for key, value in case.items() if key != "specification_digest"})
    expected = {"idea": name, "hardware_class": context, "copies": copies, "source_commit": source,
                "unit_id": case["unit_id"], "specification_digest": case_digest}
    if (record.get("case_completed") is not True or case.get("specification_digest") != case_digest
            or any(record.get(key) != value for key, value in expected.items())
            or (case["hypothesis"], case["hardware_class"], case["configuration"]["n_copies"], case["source_commit"]) != (name, context, copies, source)):
        raise ValueError("a cohort result does not match its immutable completed case")
    if any(hardware.get(key) != case.get(key) for key in ("source_root", "source_commit", "hardware_class")):
        raise ValueError("the recorded case process ran another source or hardware context")
    if type(hardware.get("pid")) is not int or not hardware.get("process_start_time"):
        raise ValueError("a completed cohort result lacks its actual process identity")
    environment, execution, worker = case_source_defaults(case, evidence)
    reference = {"learner": case["configuration"], "environment": {**environment, **case.get("reference_environment", {})},
                 "execution": {**execution, **(case.get("reference_execution") or {})}}
    candidate = {scope: {**reference[scope], **case["candidate"][scope]} for scope in ("learner", "environment", "execution")}
    if (configuration(record["reference"]) != configuration(reference)
            or configuration(record["candidate"]) != configuration(candidate)
            or canonical_digest(reference) == canonical_digest(candidate) or record.get("implementation_distinct") is not True):
        raise ValueError("a cohort result changed its supplied configuration or has no distinct implementation")
    # An explicit attempt beyond the unchanged original limit requires the separate actual-runtime receipt.
    # Ordinary historical cases retain their original source-protocol reconstruction.
    if record.get("attempt") == 5:
        runtime = pregather_retry_validation_module().validate_execution(path, case, evidence=evidence)
    elif (record.get("attempt", 1) > case.get("maximum_attempts", 3)
            or "infrastructure_retry_approval" in hardware):
        runtime = retry_validation_module().validate_execution(path, case, evidence=evidence)
    else:
        runtime = cohort_runtime_evidence(case, record, hardware, worker, evidence)
    if record.get("failure_stage") == "correctness":
        if (record.get("decision") != "rejected" or record.get("correctness_attempted") is not True
                or record.get("correctness_passed") is not False or not record.get("error")):
            raise ValueError("a numerical rejection has no actual recorded correctness failure")
        return runtime
    correctness(record)
    rounds = record.get("rounds", [])
    if record.get("measurement_valid") is not True or record.get("contaminated") is not False or len(rounds) != case["rounds"] or case["rounds"] < 7:
        raise ValueError("a cohort success has no complete valid measurement")
    for index, pair in enumerate(rounds):
        if pair.get("order") != (["reference", "candidate"] if index % 2 == 0 else ["candidate", "reference"]):
            raise ValueError("a cohort pair did not alternate its execution order")
        for arm in ("reference", "candidate"):
            if pair.get(arm, {}).get("iterations") != case["iterations"]:
                raise ValueError("a cohort measurement changed its planned complete-update window")
            measured_round(pair[arm], copies, minimum_updates=case["iterations"])
    return runtime

def require_fifty_investigated(registry, evidence):
    """Require every recorded matched copy count, including valid numerical rejections, for all fifty mechanisms."""
    initial = registry["initial_ideas"]
    batches = registry["adaptive_batches"]
    if len(initial) != 30 or len(batches) != 4 or any(len(batch["hypotheses"]) != 5 for batch in batches):
        raise ValueError("the complete study requires thirty initial and four batches of five additional mechanisms")
    entries = [(entry, registry.get("initial_matched_copy_counts"), registry["reference_commit"]) for entry in initial]
    entries += [(entry, batch.get("matched_copy_counts"), batch["reference_commit"])
                for batch in batches for entry in batch["hypotheses"]]
    names = [entry.get("name") for entry, _, _ in entries]
    if any(not name for name in names) or len(set(names)) != 50:
        raise ValueError("all fifty mechanisms must have distinct completed identities")
    for entry, counts, source in entries:
        if not counts or any(type(count) is not int or count < 1 for count in counts):
            raise ValueError("every cohort must preserve its matched policy-copy counts")
        for context in CONTEXTS:
            for copies in counts:
                records = [record for record in entry.get("results", {}).get(context, []) if record.get("copies") == copies]
                valid = False
                for summary in records:
                    if not summary.get("result_path"):
                        continue
                    proof = validate_cohort_case(summary["result_path"], entry["name"], context, copies, source, evidence)
                    evidence.runtime_provenance[summary["result_path"]] = proof
                    valid = True
                if not valid:
                    raise ValueError(f"unfinished mechanism/copy measurement: {entry['name']} / {context} / {copies}")


def validate_final_verification(specification, evidence=None, specification_path=None):
    """Derive all five final gates from source-matched raw evidence rather than supplied Boolean flags."""
    evidence = Evidence() if evidence is None else evidence
    if specification_path is not None and evidence.read(specification_path) != specification:
        raise ValueError("the final evidence specification changed")
    source = specification["source_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", source) or set(specification["contexts"]) != set(CONTEXTS):
        raise ValueError("final verification needs one full source commit and both hardware contexts")
    runtime = runtime_environment(specification["runtime_environment"])
    require_fifty_investigated(evidence.read(specification["study_registry"]), evidence)
    selected = {}
    for context in CONTEXTS:
        common = {**specification["contexts"][context], "source_commit": source,
                  "hardware_class": context, "runtime_environment": runtime}
        combination = validate_combination(common, evidence)
        packing = validate_packing(common, evidence)
        if not packing["complete"] or packing["configuration"] != combination["configuration"]:
            raise ValueError("the selected packing count needs its own combination correctness and confirmation")
        selected[context] = {"configuration": combination["configuration"], "copies": combination["copies"],
                             "combination": combination, "packing": packing}
    scaling = validate_scaling({**specification["scaling"], "source_commit": source, "runtime_environment": runtime}, evidence)
    if scaling["configuration"] != selected["jaguar03_rtx_a4500"]["configuration"]:
        raise ValueError("one-versus-eight scaling did not use the final selected A4500 configuration")
    return {"schema_version": 1, "kind": "final_verification", "source_commit": source,
            **{name: True for name in FINAL_FLAGS}, "runtime_environment": runtime,
            "specification": specification, "specification_digest": canonical_digest(specification),
            "specification_path": evidence.path_identity(specification_path) if specification_path is not None else None,
            "cohort_runtime_provenance": [{"result_path": path, **evidence.runtime_provenance[path]}
                                          for path in sorted(evidence.runtime_provenance)],
            "selected": selected, "scaling": scaling, "evidence": evidence.records()}


def validate_training_admission(path, resolver=None):
    """Revalidate original final evidence before admitting the exact source, configuration and policy seeds."""
    evidence = Evidence(resolver)
    specification = evidence.read(path)
    saved = evidence.read(specification["final_verification"])
    if saved.get("kind") != "final_verification" or saved.get("specification_digest") != canonical_digest(saved["specification"]):
        raise ValueError("the final verification receipt is missing its original evidence specification")
    rebuilt = validate_final_verification(saved["specification"], Evidence(resolver), specification_path=saved.get("specification_path"))
    if rebuilt != saved:
        raise ValueError("final verification evidence changed or no longer passes its recorded gates")
    context = specification["hardware_class"]
    if context not in CONTEXTS or specification["source_commit"] != saved["source_commit"]:
        raise ValueError("training source or context differs from final verification")
    selected = saved["selected"][context]
    actual = configuration(specification["configuration"])
    if actual != selected["configuration"] or runtime_environment(specification["runtime_environment"]) != saved["runtime_environment"]:
        raise ValueError("training configuration or runtime differs from the confirmed selected implementation")
    seeds = specification["copy_seeds"]
    if (len(seeds) != selected["copies"] or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in seeds)):
        raise ValueError("training needs one distinct nonnegative seed per selected policy copy")
    root = Path(specification["source_root"])
    if not root.is_absolute():
        raise ValueError("training source_root must be an existing absolute checkout")
    physical_root = evidence.git_root(specification["source_root"], specification["source_commit"])
    if not physical_root.is_dir():
        raise ValueError("training source_root must be an existing absolute checkout")
    commit = subprocess.check_output(["git", "-C", str(physical_root), "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output(["git", "-C", str(physical_root), "status", "--porcelain", "--untracked-files=all",
        "--", "10_jax_exploration_platform/src/exploration_platform/atari_pixels",
        "10_jax_exploration_platform/scripts/atari_pixel_campaign"], text=True).strip()
    if commit != specification["source_commit"] or changes:
        raise ValueError("training source checkout differs from its verified frozen commit")
    module = root / "10_jax_exploration_platform/src/exploration_platform/atari_pixels"
    source_files = {}
    for name in ("config.py", "environment.py"):
        path = module / name
        contents = evidence.file(path, record=False)
        if resolver is not None:
            relative = str(path.relative_to(root))
            original = subprocess.check_output(["git", "-C", str(physical_root), "show", commit + ":" + relative])
            if contents != original:
                raise ValueError("a staged training source file differs from its pinned Git blob: " + str(path))
        source_files[name] = contents
    tree = ast.parse(source_files["config.py"])
    fields = {node.target.id for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "PixelPPOConfig"
              for node in item.body if isinstance(node, ast.AnnAssign) and node.value is not None}
    if set(actual["learner"]) != fields:
        raise ValueError("training requires every frozen learner field explicitly resolved")
    tree = ast.parse(source_files["environment.py"])
    constructor = next(node for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "AtariPixelEnvironment"
                       for node in item.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    if set(actual["environment"]) != {argument.arg for argument in constructor.args.args if argument.arg != "self"}:
        raise ValueError("training requires every frozen environment field explicitly resolved")
    return {**specification, "configuration": actual, "runtime_environment": saved["runtime_environment"],
            "validation": {"kind": "training_admission", "source_commit": saved["source_commit"],
                "specification_digest": canonical_digest(specification), "configuration_digest": canonical_digest(actual),
                "runtime_environment": saved["runtime_environment"], "selected_copies": selected["copies"],
                "admission_records": evidence.records(), "evidence": saved["evidence"]}}


def validate_training_completion(specification_path, unit_directory, resolver=None):
    """Verify the actual 2B schedule and checkpoint bytes without importing or unpickling arrays."""
    admitted = validate_training_admission(specification_path, resolver=resolver)
    evidence = Evidence(resolver)
    directory = Path(evidence.path_identity(unit_directory, strict=True))
    result = evidence.read(directory / "result.json")
    identity_record = evidence.read(directory / "training_identity.json")
    checkpoint = evidence.read(directory / "checkpoints/latest.metadata.json")
    selected = admitted["configuration"]
    copies = selected["learner"]["n_copies"]
    target = 2_000_000_000 // (128 * 128)
    steps = target * 128 * 128
    expected = {"source_root": admitted["source_root"], "source_commit": admitted["source_commit"],
                "hardware_class": admitted["hardware_class"], "configuration": selected,
                "copy_seeds": admitted["copy_seeds"]}
    if any(identity_record.get(name) != value for name, value in expected.items()):
        raise ValueError("completed training identity differs from its validated final specification")
    actual_runtime = {name: identity_record["runtime_environment"].get(name) for name in RUNTIME_KEYS}
    if runtime_environment(actual_runtime) != admitted["runtime_environment"]:
        raise ValueError("completed training used a different compiler runtime")
    digest = canonical_digest(identity_record)
    if (result.get("completed") is not True or result.get("update") != target or result.get("target_updates") != target
            or result.get("requested_steps_per_copy") != 2_000_000_000 or result.get("steps_per_copy") != steps
            or result.get("copy_seeds") != admitted["copy_seeds"] or result.get("copies") != copies
            or result.get("identity_digest") != digest):
        raise ValueError("training has not completed the exact original per-policy update schedule")
    if (checkpoint.get("kind") != "optimized_training_checkpoint" or checkpoint.get("identity_digest") != digest
            or checkpoint.get("configuration_digest") != canonical_digest(selected)
            or checkpoint.get("source_commit") != admitted["source_commit"] or checkpoint.get("iteration") != target
            or checkpoint.get("target_updates") != target or checkpoint.get("steps_per_copy") != steps
            or checkpoint.get("copy_seeds") != admitted["copy_seeds"]):
        raise ValueError("the checkpoint metadata does not describe the completed learner state")
    path = directory / "checkpoints/latest.pkl"
    physical_path = evidence.physical_file(path)
    if checkpoint.get("checkpoint_path") != evidence.path_identity(path) or not physical_path.is_file():
        raise ValueError("the completed checkpoint is not at its recorded original path")
    digest = hashlib.sha256()
    with physical_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != checkpoint.get("checkpoint_sha256"):
        raise ValueError("the completed checkpoint bytes changed")
    return {"kind": "training_completion", "completed": True, "source_commit": admitted["source_commit"],
            "hardware_class": admitted["hardware_class"], "copy_seeds": admitted["copy_seeds"],
            "updates": target, "steps_per_copy": steps, "checkpoint_sha256": digest.hexdigest(),
            "configuration_digest": canonical_digest(selected), "evidence": evidence.records()}


def main():
    """Validate a complete final-verification specification or a final-training admission document."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, required=True)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--training-admission", action="store_true")
    operation.add_argument("--training-completion", type=Path, metavar="UNIT_DIRECTORY")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evidence-manifest", type=Path)
    parser.add_argument("--evidence-manifest-sha256")
    args = parser.parse_args()
    if args.evidence_manifest_sha256 and args.evidence_manifest is None:
        parser.error("an expected evidence manifest hash requires --evidence-manifest")
    resolver = None
    if args.evidence_manifest is not None:
        from portable_evidence import ExactPathResolver
        resolver = ExactPathResolver(args.evidence_manifest, args.evidence_manifest_sha256)
    if args.training_completion:
        result = validate_training_completion(args.specification, args.training_completion, resolver=resolver)
    elif args.training_admission:
        result = validate_training_admission(args.specification, resolver=resolver)
    else:
        evidence = Evidence(resolver)
        result = validate_final_verification(evidence.read(args.specification), evidence, args.specification)
    contents = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    if args.output:
        if args.output.exists():
            if args.output.read_bytes() != contents:
                raise ValueError("an existing validation receipt is immutable")
        else:
            descriptor, temporary = tempfile.mkstemp(prefix=".validation-", dir=args.output.parent)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(contents)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, args.output)
            finally:
                Path(temporary).unlink()
    print(contents.decode(), end="")


if __name__ == "__main__":
    main()
