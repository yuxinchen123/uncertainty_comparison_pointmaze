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
import socket
import statistics
import subprocess
import tempfile
import time
import types


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
FINAL_PROTOCOL = {"kind": "post_fifty_final_evidence", "paired_rounds": 11,
                  "minimum_updates_per_round": 20, "minimum_learning_updates": 128,
                  "fixed_total_scaling_where_both_fit": True}
SHORT_LEARNING_MODULE = "short_learning.py"
JAXTARI_COMMIT = "04ab9aec318c72162e2e53503a594a489d7e162b"
PREGATHER_ASYNC_RETRY_MODULE = "pregather_async_retry.py"
H100_NVL_SLURM_QUEUE_MODULE = "h100_nvl_slurm_queue.py"
H100_NVL_SLURM_DEPENDENCIES = ("h100_nvl_slurm_queue.py", "h100_nvl_slurm_case.py",
                             "h100_nvl_slurm_evidence.py", "h100_nvl_slurm_site_validation.py")
H100_NVL_SLURM_QUALIFICATION_DEPENDENCIES = ("h100_nvl_slurm_qualification.py", "h100_nvl_allocation_observation.py")


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


def h100_nvl_slurm_validation_module():
    """Load the exact adjacent Slurm execution validator even when this reader is imported under an alias."""
    specification = importlib.util.spec_from_file_location("pixel_h100_nvl_slurm_validation",
        Path(__file__).resolve().with_name(H100_NVL_SLURM_QUEUE_MODULE))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def reader_feature_dependencies(trees):
    """Keep shared reader files whenever either explicitly committed feature declaration needs them."""
    features = {"SHORT_LEARNING_MODULE": ("short_learning.py", {"short_learning.py", "portable_evidence.py"}),
        "DEVICE_MEMORY_LIMIT_MODULE": ("device_memory_limit.py", {"device_memory_limit.py", "validate_training_deployment.py", "portable_evidence.py"})}
    needed = set()
    for tree in trees:
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in features:
                        name, dependencies = features[target.id]
                        if not isinstance(node.value, ast.Constant) or node.value.value != name:
                            raise ValueError("a committed final reader changed its declared dependency filename")
                        needed.update(dependencies)
    return needed, set().union(*(value[1] for value in features.values()))


def helper_requirements(required, verified_files):
    """Require the new metadata dependency only when the already verified pinned caller declares it."""
    selected = set(required)
    prefix = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
    declarations = set()
    callers = []
    for name in ("post_comparison_validation.py", "controller.py"):
        path = prefix + name
        if path in verified_files:
            tree = ast.parse(verified_files[path])
            callers.append(tree)
            declarations.update(node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str))
    for declaration, dependencies in ((PREGATHER_ASYNC_RETRY_MODULE, (PREGATHER_ASYNC_RETRY_MODULE,)),
                                      (H100_NVL_SLURM_QUEUE_MODULE, H100_NVL_SLURM_DEPENDENCIES)):
        for name in dependencies:
            relative = prefix + name
            if declaration in declarations:
                selected.add(relative)
            else:
                selected.discard(relative)
    qualification_declared = False
    for name in ("h100_nvl_slurm_queue.py", "h100_nvl_slurm_evidence.py"):
        if H100_NVL_SLURM_QUEUE_MODULE in declarations and prefix + name in verified_files:
            tree = ast.parse(verified_files[prefix + name])
            qualification_declared |= any(isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "H100_NVL_SLURM_QUALIFICATION_MODULE" for target in node.targets)
                and isinstance(node.value, ast.Constant) and node.value.value == "h100_nvl_slurm_qualification.py" for node in tree.body)
    for name in H100_NVL_SLURM_QUALIFICATION_DEPENDENCIES:
        relative = prefix + name
        if qualification_declared:
            selected.add(relative)
        else:
            selected.discard(relative)
    needed, universe = reader_feature_dependencies(callers)
    for name in universe:
        if name in needed:
            selected.add(prefix + name)
        elif prefix + name not in verified_files:
            selected.discard(prefix + name)
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


def validate_combination(specification, evidence, *, final=False):
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
    if final and len(rounds) != 11:
        raise ValueError("final confirmation requires exactly eleven paired rounds")
    if len(rounds) < 7:
        raise ValueError("combination confirmation requires seven paired rounds")
    ratios = []
    for index, pair in enumerate(rounds):
        expected_order = ["reference", "candidate"] if index % 2 == 0 else ["candidate", "reference"]
        if pair.get("order") != expected_order or pair["reference"].get("iterations") != pair["candidate"].get("iterations"):
            raise ValueError("confirmation order or work is not matched")
        for name in ("reference", "candidate"):
            measured_round(pair[name], copies, minimum_updates=20 if final else 10)
        ratios.append(pair["reference"]["seconds"] / pair["candidate"]["seconds"])
    wins = sum(ratio > 1 for ratio in ratios)
    probability = sum(math.comb(len(ratios), count) for count in range(wins, len(ratios) + 1)) / 2 ** len(ratios)
    if statistics.median(ratios) <= 1 or probability > .05 or second.get("decision") != "accepted":
        raise ValueError("the selected combination has no repeatable longer-window improvement")
    return {"hardware_class": context, "source_commit": source, "configuration": selected,
            "runtime_environment": runtime, "copies": copies, "paired_rounds": len(rounds),
            "median_speedup": statistics.median(ratios), "one_sided_sign_probability": probability}


def packing_point(path, context, source, runtime, evidence, *, require_headroom=True):
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
    if require_headroom and free_fraction < .15:
        raise ValueError("packing lacks fifteen percent measured GPU memory headroom")
    seeds = record.get("copy_seeds", [])
    if len(seeds) != copies or len(set(seeds)) != copies or any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in seeds):
        raise ValueError("packing requires the explicit independent policy seeds")
    return {"copies": copies, "throughput": throughput, "configuration": current,
            "process": list(process), "gpu_uuid": gpu, "iterations": rounds[0]["iterations"],
            "copy_seeds": seeds, "sampled_process_peak_bytes": sampled_peak,
            "total_gpu_memory_bytes": total_bytes, "free_memory_fraction": free_fraction}


def memory_limited_packing(specification, points, gains, processes, evidence):
    """Use a completed low-headroom doubling and a fresh fitting repeat as a measured memory limit."""
    context, source = specification["hardware_class"], specification["source_commit"]
    runtime = runtime_environment(specification["runtime_environment"])
    # Before: fitting counts 1,2 and a completed low-headroom count 4; after: select only repeated count 2.
    if (any(point["free_memory_fraction"] < .15 for point in points[:-1])
            or points[-1]["free_memory_fraction"] >= .15):
        raise ValueError("a measured memory limit needs fitting points followed by one low-headroom doubling")
    if any(gain < .10 for gain in gains[:-1]):
        raise ValueError("packing continued beyond its first below-ten-percent doubling before its memory limit")
    original = points[-2]
    paths = specification.get("packing_confirmation_records", [])
    if len(paths) > 1:
        raise ValueError("a measured memory limit repeats only its largest fitting copy count")
    repeated = [packing_point(path, context, source, runtime, evidence) for path in paths]
    if repeated:
        point = repeated[0]
        if (point["configuration"] != original["configuration"] or point["gpu_uuid"] != original["gpu_uuid"]
                or point["iterations"] != original["iterations"] or point["copy_seeds"] != original["copy_seeds"]
                or tuple(point["process"]) in processes):
            raise ValueError("memory-limit confirmation changed identity/window or reused an earlier process")
    # Require the actual producer protocol and full physical group on every new-path record, including the boundary.
    group = None
    all_paths = specification["packing_records"] + paths
    for path in all_paths:
        record = evidence.read(path)
        hardware = record["hardware"]
        process = execution_identity(record)
        rows = list(csv.reader(hardware["gpu_query"].splitlines()))
        actual = hardware.get("runtime_environment", {})
        if (runtime_environment({key: actual.get(key) for key in RUNTIME_KEYS}) != runtime
                or actual.get("JAX_ENABLE_COMPILATION_CACHE") != "false"
                or actual.get("JAX_COMPILATION_CACHE_DIR") not in (None, "")
                or actual.get("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES") != ""):
            raise ValueError("memory-limit measurements need their actual full-precision uncached runtime")
        identity_fields = (hardware["hostname"], *[part.strip() for part in rows[0]],
                           canonical_digest({key: value for key, value in actual.items()
                                             if key not in {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"}}))
        updates = record["rounds"][0]["iterations"]
        options = record["configuration"]["execution"]
        multiple = options["iterations_per_call"]
        if options["report_readout"] != "none":
            multiple = math.lcm(multiple, options["report_every_iterations"])
        if updates != ((10 + multiple - 1) // multiple) * multiple:
            raise ValueError("memory-limit windows must contain the producer's complete update groups and report windows")
        if (len(rows[0]) != 4 or any(not part for part in identity_fields)
                or record.get("measurement_mode") != "packing-point"
                or record.get("contaminated", False) is not False or record.get("measurement_valid", True) is not True
                or record.get("protocol") != {"rounds": 7, "requested_minimum_updates_per_round": 10,
                                              "actual_updates_per_round": updates}
                or process != (hardware["hostname"], hardware["pid"], hardware["process_start_time"])):
            raise ValueError("memory-limit evidence needs its actual producer protocol and exact hardware process")
        total = int(re.fullmatch(r"(\d+) MiB", rows[0][2].strip()).group(1)) * 2 ** 20
        for row in record["rounds"]:
            observed = row["memory"]["sampled_process_peak_bytes"], row["memory"]["allocator"]["peak_bytes_in_use"]
            if any(type(value) is not int or not 0 < value <= total for value in observed):
                raise ValueError("memory-limit process and allocator peaks must be actual possible byte counts")
        if group is not None and identity_fields != group:
            raise ValueError("memory-limit measurements changed their physical GPU, host, model, memory or driver/runtime")
        group = identity_fields
    if not repeated:
        return {"complete": False, "hardware_class": context, "next_action": "repeat_largest_fitting_count",
                "required_confirmation_counts": [original["copies"]], "points": points, "marginal_gains": gains}
    if type(specification.get("selected_copies")) is not int or specification["selected_copies"] != original["copies"]:
        raise ValueError("the measured memory limit selects only the largest fitting copy count")
    return {"complete": True, "hardware_class": context, "source_commit": source,
            "selected_copies": original["copies"], "configuration": original["configuration"],
            "points": points, "marginal_gains": gains, "confirmation": repeated,
            "stop_reason": "completed_doubling_below_fifteen_percent_memory_headroom"}


def validate_packing(specification, evidence):
    """Confirm the throughput limit, or explicitly opt into a completed measured-memory boundary."""
    context, source = specification["hardware_class"], specification["source_commit"]
    runtime = runtime_environment(specification["runtime_environment"])
    rule = specification.get("packing_stop_rule")
    if "packing_stop_rule" in specification and rule != "measured_memory_headroom":
        raise ValueError("packing_stop_rule must explicitly name measured_memory_headroom")
    points = [packing_point(path, context, source, runtime, evidence, require_headroom=rule is None)
              for path in specification["packing_records"]]
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
    if rule == "measured_memory_headroom":
        return memory_limited_packing(specification, points, gains, processes, evidence)
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
    new_h100 = (context == "h100" and (hardware.get("slurm_job_id")
        or hardware.get("hostname", "").split(".")[0] in ("serval06", "serval07", "serval08", "serval09"))
        or record.get("execution_kind") == "h100_nvl_slurm_original_case"
        or hardware.get("execution_kind") == "h100_nvl_slurm_original_case")
    if new_h100:
        if Path(path).name != "result.json":
            raise ValueError("a Slurm H100 cohort must use its exact canonical attempt result path")
        actual, checked = h100_nvl_slurm_validation_module().validate_execution(
            Path(path).parent, case, require_completed=True, evidence=evidence)
        if checked != record:
            raise ValueError("the Slurm execution receipt differs from the supplied original cohort result")
        runtime = {"scope": "actual_h100_nvl_slurm_runtime",
            "runtime_environment": {key: actual["runtime_environment"][key] for key in RUNTIME_KEYS},
            "actual_runtime_environment": actual["runtime_environment"],
            "execution_record": {"path": evidence.path_identity(Path(path).parent / "slurm_execution.json"),
                                 "sha256": hashlib.sha256(evidence.file(Path(path).parent / "slurm_execution.json")).hexdigest()},
            "placement_policy": actual["placement_policy"], "allocation_validation": actual["allocation_validation"]}
    elif record.get("attempt") == 5:
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


def publish_evidence_bytes(path, raw):
    """Fsync one new metadata artifact without replacing any existing preparation or result."""
    path = Path(path)
    if not path.is_absolute() or path.parent.resolve(strict=True) != path.parent or path.exists() or path.is_symlink():
        raise ValueError("publish final evidence only at a fresh exact ordinary metadata path")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def produce_completed_study_proof(specification, output):
    """Freeze real fifty-idea recomputation on owned compute before any final-phase measurement."""
    import controller

    if set(specification) != {"source_commit", "runtime_environment", "study_registry", "helpers", "accepted_combinations"}:
        raise ValueError("completed-study preparation requires exact source/runtime/registry/helper inputs")
    registry_path = Path(specification["study_registry"])
    controller.final_validation_allocation(registry_path.parent)
    started = time.time()
    evidence = Evidence()
    runtime = runtime_environment(specification["runtime_environment"])
    registry_raw = evidence.file(registry_path)
    registry = json.loads(registry_raw)
    if not registry.get("adaptive_batches") or registry["adaptive_batches"][-1].get("reference_commit") != specification["source_commit"]:
        raise ValueError("completed-study proof must use the last completed batch's explicitly accepted source")
    committed_measurement_sources(specification["helpers"], evidence)
    prefix = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
    required = {prefix + name for name in ("post_comparison_validation.py", "controller.py", "ideas.py",
        "infrastructure_retry.py", "pregather_async_retry.py", *H100_NVL_SLURM_DEPENDENCIES,
        *H100_NVL_SLURM_QUALIFICATION_DEPENDENCIES)}
    verified = {relative: evidence.file(Path(specification["helpers"]["source_root"]) / relative)
                for relative in specification["helpers"]["files"]}
    if not helper_requirements(required, verified) <= set(verified):
        raise ValueError("completed-study proof omitted a committed metadata-reader dependency")
    for name, module in (("post_comparison_validation.py", None), ("controller.py", controller)):
        executed = Path(__file__) if module is None else Path(module.__file__)
        if hashlib.sha256(executed.read_bytes()).hexdigest() != specification["helpers"]["files"].get(prefix + name):
            raise ValueError("completed-study proof must execute its recorded committed metadata helpers")
    require_fifty_investigated(registry, evidence)
    last_results = [row for item in registry["adaptive_batches"][-1]["hypotheses"]
                    for context in CONTEXTS for row in item["results"][context]]
    last_finished = max(positive(evidence.read(row["result_path"])["execution_identity"]["finished_at_seconds"],
        "last adaptive case finish") for row in last_results)
    accepted = {}
    if set(specification["accepted_combinations"]) != set(CONTEXTS):
        raise ValueError("completed-study proof needs both actual accepted combination records")
    for context in CONTEXTS:
        common = {**specification["accepted_combinations"][context], "source_commit": specification["source_commit"],
            "hardware_class": context, "runtime_environment": runtime}
        combination = validate_combination(common, evidence)
        records = {}
        for mode, name in (("correctness", "correctness_record"), ("confirmation", "confirmation_record")):
            result, plan, process, assets = measurement_binding(common[name], mode, common, evidence)
            if result["execution_identity"]["started_at_seconds"] <= last_finished:
                raise ValueError("the accepted final source combination must be measured after the last adaptive batch")
            if result["execution_identity"]["finished_at_seconds"] >= started:
                raise ValueError("the completed-study proof must be produced after both accepted combinations finish")
            if mode == "confirmation":
                updates = complete_window([result["reference"], result["candidate"]], 10)
                if ("final_confirmation_protocol" in plan or plan.get("rounds") != 7 or plan.get("minimum_updates_per_round") != 10
                        or plan.get("iterations") != updates or len(result.get("rounds", [])) != 7
                        or any(pair[arm].get("iterations") != updates for pair in result["rounds"] for arm in ("reference", "candidate"))):
                    raise ValueError("the completed-study source proof uses intermediate seven-by-ten acceptance, before final eleven-by-twenty")
            records[name] = {"path": evidence.path_identity(common[name]), "sha256": hashlib.sha256(evidence.file(common[name])).hexdigest()}
        accepted[context] = {"source_commit": specification["source_commit"], "configuration": combination["configuration"], "records": records}
    snapshot_path = Path(output).parent / "completed_fifty_registry.json"
    publish_evidence_bytes(snapshot_path, registry_raw)
    owner_files = {}
    for name in ("submitted_jobids_uva.txt", "submissions.tsv"):
        path = registry_path.parent / "slurm" / name
        if path.stat().st_uid != os.getuid():
            raise ValueError("completed-study proof requires its own UID's exact UVA ledgers")
        raw = path.read_bytes()
        destination = Path(output).parent / ("completed_fifty_" + name)
        publish_evidence_bytes(destination, raw)
        owner_files[name] = {"path": str(destination), "sha256": hashlib.sha256(raw).hexdigest()}
    job = os.environ["SLURM_JOB_ID"]
    ids = Path(owner_files["submitted_jobids_uva.txt"]["path"]).read_text().splitlines()
    rows = [line.split("\t") for line in Path(owner_files["submissions.tsv"]["path"]).read_text().splitlines() if line.split("\t", 1)[0] == job]
    if (ids.count(job) != 1 or len(rows) != 1 or len(rows[0]) != 4 or not re.fullmatch(r"[0-9a-f]{40}", rows[0][2])
            or not Path(rows[0][-1]).is_absolute() or not Path(rows[0][-1]).is_relative_to(registry_path.parent)):
        raise ValueError("completed-study proof requires its exact owned UVA ID and submission row")
    script = {"path": rows[0][-1], "sha256": hashlib.sha256(Path(rows[0][-1]).read_bytes()).hexdigest()}
    receipts = [row for row in evidence.records() if row["path"] != evidence.path_identity(registry_path)]
    receipts.append({"path": str(snapshot_path), "sha256": hashlib.sha256(registry_raw).hexdigest()})
    receipts.sort(key=lambda row: row["path"])
    proof = {"schema_version": 1, "kind": "completed_fifty_idea_study_proof",
        "source_commit": specification["source_commit"], "runtime_environment": runtime,
        "registry": {"path": str(snapshot_path), "sha256": hashlib.sha256(registry_raw).hexdigest(),
                     "original_path": evidence.path_identity(registry_path)},
        "helpers": specification["helpers"], "investigated_hypotheses": 50,
        "accepted_combinations": accepted, "last_adaptive_case_finished_at_seconds": last_finished,
        "owner": {"cluster": "uva", "job_id": job, "submission_row": rows[0], "snapshots": owner_files,
                  "submission_script": script, "uid": os.getuid()},
        "scientific_recomputation_on_owned_compute": True, "evidence": receipts,
        "producer": {"hostname": socket.gethostname(), "pid": os.getpid(),
            "process_start_time": Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
            "slurm_job_id": os.environ["SLURM_JOB_ID"], "started_at_seconds": started, "finished_at_seconds": time.time()},
        "hypothesis_count_contribution": 0, "comparison_count_contribution": 0, "final_training_credit": 0}
    proof["verification_digest"] = canonical_digest(proof)
    return proof


def pinned_evidence_hash(path, expected, evidence):
    """Hash already reviewed raw files in bounded chunks without repeating scientific calculations."""
    physical = evidence.physical_file(path)
    before = physical.stat()
    digest = hashlib.sha256()
    with physical.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = physical.stat()
    if (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or digest.hexdigest() != expected
            or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
        raise ValueError("reviewed completed-study raw evidence changed its bytes")
    receipt = {"path": evidence.path_identity(path, strict=True), "sha256": expected}
    if receipt["path"] in evidence.receipts and evidence.receipts[receipt["path"]] != receipt:
        raise ValueError("completed-study evidence changed during validation")
    evidence.receipts[receipt["path"]] = receipt


def verify_completed_study(binding, source, runtime, evidence, *, before_seconds=None, registry_sha256=None,
                           hardware_class=None, selected_configuration=None):
    """Validate the reviewed completed-study snapshot and byte hashes, including native portable copies."""
    if not isinstance(binding, dict) or set(binding) != {"proof", "review"}:
        raise ValueError("final preparation requires a separately reviewed completed-fifty-idea proof")
    documents = {}
    for name in ("proof", "review"):
        item = binding[name]
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("completed-study input requires its exact original path and hash")
        raw = evidence.file(item["path"])
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("a completed-study proof or independent review changed")
        documents[name] = json.loads(raw)
    proof, review = documents["proof"], documents["review"]
    unsigned = {name: value for name, value in proof.items() if name != "verification_digest"}
    if (type(proof.get("schema_version")) is not int or proof["schema_version"] != 1
            or proof.get("kind") != "completed_fifty_idea_study_proof"
            or proof.get("verification_digest") != canonical_digest(unsigned)
            or type(proof.get("investigated_hypotheses")) is not int or proof["investigated_hypotheses"] != 50
            or proof.get("source_commit") != source or runtime_environment(proof["runtime_environment"]) != runtime
            or proof.get("scientific_recomputation_on_owned_compute") is not True
            or any(type(proof.get(name)) is not int or proof[name] != 0 for name in (
                "hypothesis_count_contribution", "comparison_count_contribution", "final_training_credit"))):
        raise ValueError("completed-study proof changed its source, verified fifty ideas or zero-credit boundary")
    execution_identity({"execution_identity": proof["producer"]})
    committed_measurement_sources(proof["helpers"], evidence)
    snapshot = proof["registry"]
    if hashlib.sha256(evidence.file(snapshot["path"])).hexdigest() != snapshot["sha256"] or registry_sha256 is not None and registry_sha256 != snapshot["sha256"]:
        raise ValueError("final measurement does not bind the actual completed fifty-idea registry snapshot")
    registry = evidence.read(snapshot["path"])
    if registry.get("adaptive_batches", [{}])[-1].get("reference_commit") != source:
        raise ValueError("completed-study proof changed the final adaptive source identity")
    accepted = proof.get("accepted_combinations", {})
    if set(accepted) != set(CONTEXTS):
        raise ValueError("completed-study proof omitted the actual final accepted combinations")
    for context in CONTEXTS:
        value = accepted[context]
        if value.get("source_commit") != source or set(value.get("records", {})) != {"correctness_record", "confirmation_record"}:
            raise ValueError("completed-study proof's final combination/source identities disagree")
        for receipt in value["records"].values():
            raw = evidence.file(receipt["path"])
            result = json.loads(raw)
            if (hashlib.sha256(raw).hexdigest() != receipt["sha256"] or result.get("source_commit") != source
                    or result.get("hardware_class") != context or result.get("candidate") != value["configuration"]
                    or result["execution_identity"]["started_at_seconds"] <= proof["last_adaptive_case_finished_at_seconds"]
                    or result["execution_identity"]["finished_at_seconds"] >= proof["producer"]["started_at_seconds"]):
                raise ValueError("completed-study proof changed the actual accepted combination's source, configuration or phase")
    if selected_configuration is not None and (hardware_class not in CONTEXTS
            or configuration_without_copies(selected_configuration) != configuration_without_copies(accepted[hardware_class]["configuration"])):
        raise ValueError("final measurement changed the post-study accepted implementation")
    entries = list(registry.get("initial_ideas", [])) + [item for batch in registry.get("adaptive_batches", []) for item in batch.get("hypotheses", [])]
    paths = {row.get("path") for row in proof.get("evidence", [])}
    if (len(entries) != 50 or len({item.get("name") for item in entries}) != 50
            or len(proof.get("evidence", [])) != len(paths)
            or any(row.get("result_path") not in paths for item in entries for context in CONTEXTS for row in item.get("results", {}).get(context, []))):
        raise ValueError("completed-study proof omitted its exact raw hypothesis coverage")
    cohorts = [(registry.get("initial_ideas", []), registry.get("initial_matched_copy_counts", []))]
    cohorts += [(batch.get("hypotheses", []), batch.get("matched_copy_counts", [])) for batch in registry.get("adaptive_batches", [])]
    if len(cohorts) != 5 or len(cohorts[0][0]) != 30 or any(len(items) != 5 for items, unused in cohorts[1:]):
        raise ValueError("completed-study proof requires the original thirty and four adaptive batches of five")
    for items, counts in cohorts:
        if not counts or len(set(counts)) != len(counts) or any(type(count) is not int or count < 1 for count in counts):
            raise ValueError("completed-study proof omitted an exact matched-copy cohort")
        for item in items:
            for context in CONTEXTS:
                results = item.get("results", {}).get(context, [])
                if len(results) != len(counts) or {row.get("copies") for row in results} != set(counts):
                    raise ValueError("completed-study proof omitted a required hardware or policy-copy measurement")
    last_rows = [row for item in cohorts[-1][0] for context in CONTEXTS for row in item["results"][context]]
    actual_last_finish = max(positive(evidence.read(row["result_path"])["execution_identity"]["finished_at_seconds"],
        "last adaptive case finish") for row in last_rows)
    if proof.get("last_adaptive_case_finished_at_seconds") != actual_last_finish:
        raise ValueError("completed-study proof changed the actual last adaptive completion time")
    owner = proof.get("owner", {})
    job = owner.get("job_id")
    if owner.get("cluster") != "uva" or job != proof["producer"].get("slurm_job_id") or not isinstance(job, str) or not re.fullmatch(r"[1-9][0-9]*", job):
        raise ValueError("completed-study proof changed its owned compute cluster or exact job")
    owner_raw = {}
    for name in ("submitted_jobids_uva.txt", "submissions.tsv"):
        receipt = owner.get("snapshots", {}).get(name, {})
        raw = evidence.file(receipt["path"])
        if hashlib.sha256(raw).hexdigest() != receipt.get("sha256"):
            raise ValueError("completed-study owned ledger snapshot changed")
        owner_raw[name] = raw.decode().splitlines()
    matching = [line.split("\t") for line in owner_raw["submissions.tsv"] if line.split("\t", 1)[0] == job]
    if (owner_raw["submitted_jobids_uva.txt"].count(job) != 1 or matching != [owner.get("submission_row")]
            or len(matching[0]) != 4 or not re.fullmatch(r"[0-9a-f]{40}", matching[0][2])
            or not Path(matching[0][-1]).is_absolute() or not Path(matching[0][-1]).is_relative_to(Path(snapshot["original_path"]).parent)
            or owner.get("submission_script", {}).get("path") != matching[0][-1]
            or type(owner.get("uid")) is not int or owner["uid"] < 0):
        raise ValueError("completed-study proof changed its exact owned compute ledger membership")
    pinned_evidence_hash(owner["submission_script"]["path"], owner["submission_script"]["sha256"], evidence)
    for item in proof["evidence"]:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("completed-study proof contains an invalid raw-file receipt")
        pinned_evidence_hash(item["path"], item["sha256"], evidence)
    reviewed = review.get("reviewed_at_seconds")
    if (type(review.get("schema_version")) is not int or review["schema_version"] != 1
            or review.get("kind") != "independent_completed_fifty_idea_review"
            or review.get("proof") != binding["proof"] or review.get("decision") != "accepted_completed_fifty_idea_evidence"
            or review.get("unresolved_anomalies") != [] or review.get("independent_of_producer") is not True
            or not isinstance(review.get("reviewer_identity"), str) or not review["reviewer_identity"].strip()
            or type(reviewed) not in (int, float) or not math.isfinite(reviewed)
            or reviewed < proof["producer"]["finished_at_seconds"]
            or before_seconds is not None and reviewed >= before_seconds):
        raise ValueError("final measurement needs an earlier independent review of the exact completed-study proof")
    return {"registry_sha256": snapshot["sha256"], "reviewed_at_seconds": reviewed,
            "proof_sha256": binding["proof"]["sha256"]}


def final_protocol(value):
    """Require literal final-phase settings without changing historical intermediate readers."""
    if (not isinstance(value, dict) or set(value) != set(FINAL_PROTOCOL)
            or any(type(value[name]) is not type(expected) or value[name] != expected
                   for name, expected in FINAL_PROTOCOL.items())):
        raise ValueError("new final evidence requires the exact eleven-by-twenty and short-learning protocol")
    return value


def require_current_final(record):
    """Prevent a historical five-flag receipt from authorizing new final training."""
    if (type(record.get("schema_version")) is not int or record["schema_version"] != 2
            or record.get("short_learning_checks_passed") is not True
            or record.get("fixed_total_scaling_complete") is not True
            or any(record.get(name) is not True for name in FINAL_FLAGS)):
        raise ValueError("new final admission requires version-two learning and fixed-total evidence")
    final_protocol(record.get("specification", {}).get("final_protocol"))


def complete_window(configurations, minimum):
    """Round only upward to complete selected groups and report windows."""
    # Before: minimum twenty, group three and report ten; after: thirty complete updates.
    multiple = 1
    for value in configurations:
        execution = value["execution"]
        multiple = math.lcm(multiple, execution["iterations_per_call"])
        if execution["report_readout"] != "none":
            multiple = math.lcm(multiple, execution["report_every_iterations"])
    return ((minimum + multiple - 1) // multiple) * multiple


def committed_measurement_sources(receipt, evidence, *, core=False):
    """Recheck exact copied files through their original or portable Git-root identity."""
    root, commit, files = receipt["source_root"], receipt["source_commit"], receipt["files"]
    if not Path(root).is_absolute() or not re.fullmatch(r"[0-9a-f]{40}", commit) or not files:
        raise ValueError("final producer sources require exact absolute Git roots and file receipts")
    physical = evidence.git_root(root, commit)
    if (Path(subprocess.check_output(["git", "-C", str(physical), "rev-parse", "--show-toplevel"], text=True).strip()).resolve() != Path(physical).resolve()
            or subprocess.check_output(["git", "-C", str(physical), "rev-parse", "HEAD"], text=True).strip() != commit):
        raise ValueError("final producer source uses another Git root or commit")
    if subprocess.run(["git", "-C", str(physical), "symbolic-ref", "--quiet", "HEAD"], capture_output=True).returncode != 1:
        raise ValueError("final producer sources require detached frozen Git roots")
    if core:
        scopes = ("10_jax_exploration_platform/src/exploration_platform/atari_pixels",
                  "10_jax_exploration_platform/scripts/atari_pixel_campaign/worker_environment.sh")
        roster = subprocess.check_output(["git", "-C", str(physical), "ls-tree", "-rz", commit, "--", *scopes])
        names = {row.split(b"\t", 1)[1].decode() for row in roster.split(b"\0") if row}
        if set(files) != names:
            raise ValueError("new final numerical source must cover its complete original Git roster")
    for relative, expected in files.items():
        if (not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts
                or not relative.startswith("10_jax_exploration_platform/")):
            raise ValueError("final source receipts require bounded project-relative paths")
        original = subprocess.check_output(["git", "-C", str(physical), "show", commit + ":" + relative])
        digest = expected["sha256"] if core else expected
        if hashlib.sha256(original).hexdigest() != digest or evidence.file(Path(root) / relative) != original:
            raise ValueError("a final producer source differs from its recorded Git bytes")
        if core and hashlib.sha1(b"blob " + str(len(original)).encode() + b"\0" + original).hexdigest() != expected["git_blob"]:
            raise ValueError("a final numerical file changed its Git blob")
    changed = subprocess.check_output(["git", "-C", str(physical), "status", "--porcelain", "--untracked-files=all", "--", *files], text=True).strip()
    if changed:
        raise ValueError("final producer source has changes in its exact file closure")


def measurement_binding(path, mode, specification, evidence):
    """Bind new final records to prepared source, assets, helpers, physical process and full configurations."""
    directory = Path(path).parent
    result, plan = evidence.read(path), evidence.read(directory / "measurement_plan.json")
    if (Path(path).name != "result.json" or directory.parent != Path(plan.get("campaign_run_dir", "")) / "data"
            or directory.name != plan.get("unit_id")
            or evidence.read(directory / "measurement_prepared.json") != {
                "plan_sha256": hashlib.sha256(evidence.file(directory / "measurement_plan.json")).hexdigest()}
            or type(plan.get("schema_version")) is not int or plan["schema_version"] != 1
            or plan.get("kind") != "selected_configuration_measurement" or plan.get("mode") != mode
            or result.get("measurement_mode") != mode or plan.get("compilation_cache_policy") != "disabled"
            or plan.get("source_commit") != specification["source_commit"]
            or plan.get("hardware_class") != specification["hardware_class"]
            or runtime_environment(plan["runtime_environment"]) != specification["runtime_environment"]
            or result.get("helpers") != plan.get("helpers")
            or result.get("configuration" if mode == "packing-point" else "candidate") != plan.get("configuration")
            or result.get("reference") != plan.get("reference_configuration")):
        raise ValueError("new final evidence differs from its original prepared producer plan")
    identity(result, specification["hardware_class"], specification["source_commit"], specification["runtime_environment"])
    process = execution_identity(result)
    if plan.get("completed_study_manifest") is not None:
        mapping = plan["completed_study_manifest"]
        if hashlib.sha256(evidence.file(mapping["path"])).hexdigest() != mapping["sha256"]:
            raise ValueError("the final producer's exact native proof mapping changed")
    seeds = plan["copy_seeds"]
    if (result.get("copy_seeds") != seeds or len(seeds) != plan["configuration"]["learner"]["n_copies"]
            or len(set(seeds)) != len(seeds) or any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in seeds)):
        raise ValueError("new final evidence changed its independent copy seeds")
    if (plan["core"]["source_root"], plan["core"]["source_commit"]) != (plan["source_root"], plan["source_commit"]):
        raise ValueError("new final evidence changed the pinned numerical root")
    for name, core in (("core", True), ("paired_source", False), ("helpers", False)):
        committed_measurement_sources(plan[name], evidence, core=core)
    if (plan["paired_source"]["source_root"], plan["paired_source"]["source_commit"]) != (plan["source_root"], plan["source_commit"]):
        raise ValueError("new final evidence changed the frozen Executor source")
    required = {"measure_selected.py", "post_comparison_validation.py", "train_optimized.py", SHORT_LEARNING_MODULE}
    if not required <= {Path(name).name for name in plan["helpers"]["files"]}:
        raise ValueError("new final evidence omitted an executed producer dependency")
    for relative, digest in plan["helpers"]["files"].items():
        if hashlib.sha256(evidence.file(directory / "code" / Path(relative).name)).hexdigest() != digest:
            raise ValueError("a staged final producer helper changed")
    hardware = evidence.read(directory / "hardware.json")
    backend = evidence.read(directory / "installed_backend.json")
    if (result.get("hardware") != hardware or result.get("installed_backend") != backend
            or (hardware.get("hostname"), hardware.get("pid"), hardware.get("process_start_time")) != process
            or hardware.get("gpu_uuid") != plan.get("gpu_uuid")
            or not isinstance(plan.get("gpu_uuid"), str) or not plan["gpu_uuid"].startswith("GPU-")
            or runtime_environment({key: hardware.get("runtime_environment", {}).get(key) for key in RUNTIME_KEYS}) != specification["runtime_environment"]):
        raise ValueError("new final evidence changed its physical process, backend or actual runtime")
    assets = plan["asset_record"]
    manifest = evidence.read(assets["manifest"])
    if (set(assets) != {"root", "manifest", "sha256"}
            or hashlib.sha256(evidence.file(assets["manifest"])).hexdigest() != assets["sha256"]
            or Path(assets["manifest"]).is_relative_to(assets["root"])
            or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or manifest.get("source_commit") != JAXTARI_COMMIT
            or not isinstance(manifest.get("files"), dict) or not manifest["files"]):
        raise ValueError("new final evidence changed its complete asset manifest")
    for name, digest in manifest["files"].items():
        if (not isinstance(name, str) or not name or "\0" in name or name == "."
                or Path(name).is_absolute() or ".." in Path(name).parts or str(Path(name)) != name
                or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("new final assets have invalid names or hashes")
    asset_identity = {"manifest_sha256": assets["sha256"], "source_commit": manifest["source_commit"],
                      "file_count": len(manifest["files"]), "tree_digest": canonical_digest(manifest["files"])}
    if evidence.read(directory / "asset_receipt.json") != {"root": evidence.path_identity(assets["root"]), **asset_identity}:
        raise ValueError("new final evidence lacks its actual exhaustive asset proof")
    return result, plan, process, asset_identity


def measurement_learning_contract(plan, evidence):
    """Read only the verified prepared helper bytes for learning and latency metadata contracts."""
    source = Path(plan["helpers"]["source_root"]) / "10_jax_exploration_platform/scripts/atari_pixel_campaign/short_learning.py"
    raw_source = evidence.file(source)
    module = types.ModuleType("verified_short_learning_contract")
    module.__file__ = str(evidence.physical_file(source))
    exec(compile(raw_source, module.__file__, "exec"), module.__dict__)
    return module


def validate_final_confirmation(specification, evidence):
    """Require the eleven-by-twenty producer and real single-update synchronization observations."""
    result, plan, process, assets = measurement_binding(specification["confirmation_record"], "confirmation", specification, evidence)
    final_protocol(plan.get("final_confirmation_protocol"))
    completed_study = verify_completed_study(plan.get("completed_study"), specification["source_commit"], specification["runtime_environment"], evidence,
        before_seconds=result["execution_identity"]["started_at_seconds"], registry_sha256=specification.get("completed_registry_sha256"),
        hardware_class=specification["hardware_class"], selected_configuration=result["candidate"])
    updates = complete_window([result["reference"], result["candidate"]], 20)
    if (plan.get("rounds") != 11 or plan.get("minimum_updates_per_round") != 20 or plan.get("iterations") != updates
            or result.get("protocol") != {"rounds": 11, "requested_minimum_updates_per_round": 20, "actual_updates_per_round": updates}
            or len(result.get("rounds", [])) != 11):
        raise ValueError("final confirmation changed its eleven complete twenty-update-or-longer windows")
    for pair in result["rounds"]:
        for arm in ("reference", "candidate"):
            if pair[arm].get("iterations") != updates or pair[arm].get("memory", {}).get("gpu_uuid") != plan["gpu_uuid"]:
                raise ValueError("final confirmation changed its prepared work or physical GPU")
    memory = evidence.read(Path(specification["confirmation_record"]).parent / "single_update_latency_memory.json")
    if (result.get("synchronized_per_update_latency_memory") != memory or memory.get("other_gpu_processes") != []
            or memory.get("gpu_uuid") != plan["gpu_uuid"] or len(memory.get("device_memory_samples", [])) < 2):
        raise ValueError("final single-update latency lacks uncontaminated physical GPU observations")
    contract = measurement_learning_contract(plan, evidence)
    declared = {arm: contract.latency_protocol(result[arm]) for arm in ("reference", "candidate")}
    if canonical_digest(plan.get("single_update_latency_protocol")) != canonical_digest(declared):
        raise ValueError("final confirmation changed its prepared single-update/reporting protocol")
    for arm in ("reference", "candidate"):
        latency = result.get("synchronized_per_update_latency", {}).get(arm, {})
        durations = latency.get("seconds_per_synchronized_call", [])
        options = declared[arm]
        window = options["complete_window_updates"]
        if (canonical_digest(latency.get("protocol")) != canonical_digest(options) or type(latency.get("calls")) is not int
                or latency["calls"] != options["calls"] or len(durations) != latency["calls"]
                or type(latency.get("updates_per_call")) is not int or latency["updates_per_call"] != 1 or latency.get("synchronization_boundary") != "each_original_complete_update"
                or latency.get("selected_iterations_per_call") != result[arm]["execution"]["iterations_per_call"]
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in durations)
                or latency.get("median_seconds_per_synchronized_call") != statistics.median(durations)
                or latency.get("mean_seconds_per_synchronized_call") != sum(durations) / len(durations)
                or latency.get("median_amortized_seconds_per_update") != statistics.median(durations)):
            raise ValueError("final confirmation requires separate actual synchronized single-update latency")
        for name in ("warmup", "selected_executor_warmup"):
            warmup = latency.get(name, [])
            if (len(warmup) < 2 or warmup[-1].get("compiled_signatures") != warmup[-2].get("compiled_signatures")
                    or any(row.get("updates") != window or not row.get("compiled_signatures") for row in warmup)):
                raise ValueError("single-update latency must warm complete selected report and execution windows")
        checks = latency.get("window_agreement", [])
        if len(checks) != options["measurement_windows"]:
            raise ValueError("single-update latency omitted a complete selected comparison window")
        for index, check in enumerate(checks):
            report_rows = options["report_rows_per_window"]
            if (check.get("passed") is not True or any(type(check.get(name)) is not int for name in ("from_iteration", "to_iteration", "report_rows"))
                    or check.get("from_iteration") != index * window
                    or check.get("to_iteration") != (index + 1) * window or check.get("report_rows") != report_rows
                    or type(check.get("state_and_final_metrics", {}).get("leaves")) is not int
                    or check["state_and_final_metrics"]["leaves"] < 1
                    or len(check.get("published_reports", [])) != report_rows
                    or report_rows and check.get("selected_reports", {}).get("leaves", 0) < 1):
                raise ValueError("single-update latency must match actual selected Executor state and scheduled reports")
            for report_index, report in enumerate(check["published_reports"], 1):
                iteration = index * window + report_index * options["selected_report_every_iterations"]
                if (type(report.get("iteration")) is not int or type(report.get("window_updates")) is not int
                        or report.get("iteration") != iteration or report.get("window_updates") != options["selected_report_every_iterations"]
                        or report.get("metrics", {}).get("steps_per_policy") != iteration * 128 * 128):
                    raise ValueError("single-update latency dropped or changed a selected scheduled report boundary")
                for value in report["metrics"].values():
                    flattened_numbers(value)
    return {"process": process, "assets": assets, "gpu_uuid": plan["gpu_uuid"], "updates_per_round": updates,
            "copy_seeds": plan["copy_seeds"], "backend": result["installed_backend"],
            "completed_study_reviewed_at_seconds": completed_study["reviewed_at_seconds"]}


def flattened_numbers(value):
    """Validate every literal numeric diagnostic rather than accepting an aggregate success marker."""
    if isinstance(value, list):
        return [item for row in value for item in flattened_numbers(row)]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("learning curves contain a nonfinite or nonnumeric observation")
    return [value]


def validate_learning_reports(result, result_path, rows, configurations, evidence):
    """Read the complete selected report stream and bind every scheduled window to its update curves."""
    expected = []
    for arm, selected_configuration in zip(("reference", "candidate"), configurations):
        options = selected_configuration["execution"]
        if options["report_readout"] == "none":
            continue
        current = [row for row in rows if row.get("variant") == arm]
        interval = options["report_every_iterations"]
        if len(current) % interval:
            raise ValueError("short learning did not retain a complete selected report window")
        expected.extend((arm, current[index - 1], interval) for index in range(interval, len(current) + 1, interval))
    receipt = result.get("reports")
    if not expected:
        if receipt is not None:
            raise ValueError("short learning saved report rows despite disabled report readout")
        return None
    path = Path(result_path).parent / "learning_reports.jsonl"
    if (not isinstance(receipt, dict) or set(receipt) != {"path", "sha256"}
            or evidence.path_identity(receipt["path"]) != evidence.path_identity(path)):
        raise ValueError("short learning omitted its exact scheduled report stream")
    raw = evidence.file(path)
    if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
        raise ValueError("the complete short-learning reports changed")
    reports = [json.loads(line) for line in raw.splitlines()]
    if len(reports) != len(expected):
        raise ValueError("short learning omitted or duplicated a selected report window")
    # The producer alternates arms by complete window. Preserve each arm's actual chronological order.
    ordered = [row for arm in ("reference", "candidate") for row in reports if row.get("variant") == arm]
    if len(ordered) != len(reports):
        raise ValueError("short learning reports contain an unknown trajectory")
    for report, (arm, last, interval) in zip(ordered, expected):
        metrics = report.get("metrics", {})
        if (set(report) != {"variant", "iteration", "window_updates", "metrics"}
                or report["variant"] != arm or type(report["iteration"]) is not int
                or report["iteration"] != last["iteration"] or type(report["window_updates"]) is not int
                or report["window_updates"] != interval or set(metrics) != set(last["metrics"])
                or metrics.get("steps_per_policy") != last["metrics"]["steps_per_policy"]):
            raise ValueError("short learning reports changed their selected schedule, counters or metric fields")
        for name, value in metrics.items():
            flattened_numbers(value)
            if name != "steps_per_policy" and isinstance(last["metrics"][name], list) and (
                    not isinstance(value, list) or len(value) != len(last["metrics"][name])):
                raise ValueError("short learning reports lost a per-copy metric vector")
    return {"path": evidence.path_identity(path), "sha256": receipt["sha256"]}


def validate_short_learning(specification, confirmation, evidence):
    """Require full independent curves, healthy updates, strict replay and a separate comparative review."""
    path = specification["learning_record"]
    result, plan, process, assets = measurement_binding(path, "learning-check", specification, evidence)
    verify_completed_study(plan.get("completed_study"), specification["source_commit"], specification["runtime_environment"], evidence,
        before_seconds=result["execution_identity"]["started_at_seconds"], registry_sha256=specification.get("completed_registry_sha256"),
        hardware_class=specification["hardware_class"], selected_configuration=result["candidate"])
    configurations = [configuration(result[name]) for name in ("reference", "candidate")]
    module = measurement_learning_contract(plan, evidence)
    declared = module.protocol(configurations)
    selected = configuration(evidence.read(specification["confirmation_record"])["candidate"])
    if (plan.get("learning_protocol") != declared or result.get("learning_protocol") != declared
            or result.get("candidate") != selected
            or result.get("reference") != evidence.read(specification["confirmation_record"])["reference"]
            or result.get("completed") is not True
            or result.get("learning_equivalence_established") is not False or result.get("zero_reward_only_evidence") is not False
            or result.get("independent_comparative_analysis_required") is not True
            or assets != confirmation["assets"] or plan["gpu_uuid"] != confirmation["gpu_uuid"]
            or result["installed_backend"] != confirmation["backend"] or plan["copy_seeds"] != confirmation["copy_seeds"]
            or process == confirmation["process"]):
        raise ValueError("short learning changed its predeclared independent full-configuration contract")
    curves = result["curves"]
    if evidence.path_identity(curves["path"]) != evidence.path_identity(Path(path).parent / "learning_curves.jsonl"):
        raise ValueError("short learning curves must remain in their exact prepared data unit")
    raw = evidence.file(curves["path"])
    if hashlib.sha256(raw).hexdigest() != curves["sha256"]:
        raise ValueError("the complete short-learning curves changed")
    rows = [json.loads(line) for line in raw.splitlines()]
    memory = evidence.read(Path(path).parent / "learning_memory.json")
    if (result.get("memory") != memory or memory.get("other_gpu_processes") != []
            or memory.get("gpu_uuid") != plan["gpu_uuid"] or len(memory.get("device_memory_samples", [])) < 2):
        raise ValueError("short learning lacks whole-check uncontaminated physical-GPU observations")
    copies, updates = len(plan["copy_seeds"]), declared["updates"]
    for arm in ("reference", "candidate"):
        current = [row for row in rows if row.get("variant") == arm]
        final = result.get("arms", {}).get(arm, {})
        if (len(current) != updates or result.get("training_start_iteration", {}).get(arm) != 0
                or final.get("completed_updates") != updates or final.get("optimizer_step") != updates * 16
                or final.get("frozen_target_unchanged") is not True):
            raise ValueError("short learning did not complete its original full schedule or frozen target")
        if final.get("parameter_change_scope") != "initial_versus_final_existing_host_snapshots":
            raise ValueError("short learning must check actual initial/final parameters without extra GPU buffers")
        for name in ("agent_change_squared", "predictor_change_squared"):
            values = final.get(name)
            if not isinstance(values, list) or len(values) != copies or any(value <= 0 for value in flattened_numbers(values)):
                raise ValueError("short learning has missing actual per-copy policy or predictor updates")
        action_counts = [[0] * 18 for unused in range(copies)]
        intrinsic = [[] for unused in range(copies)]
        previous_counts = {"observation_count": [0.] * copies, "reward_count": [0.] * copies}
        for index, row in enumerate(current, 1):
            if (row.get("update_in_check") != index or row.get("iteration") != index
                    or row.get("optimizer_step") != index * 16 or row.get("copy_seeds") != plan["copy_seeds"]
                    or row.get("floating_state_nonfinite_count") != 0
                    or not module.DIAGNOSTIC_FIELDS <= set(row)
                    or not module.METRICS <= set(row.get("metrics", {}))):
                raise ValueError("short learning curves changed policy identity, counters or required diagnostics")
            for name, values in row.items():
                if name not in {"variant", "metrics"}:
                    flattened_numbers(values)
            for values in row["metrics"].values():
                flattened_numbers(values)
            for name in module.METRICS - {"steps_per_policy"}:
                values = row["metrics"][name]
                if not isinstance(values, list) or len(values) != copies:
                    raise ValueError("short learning lost a per-copy metric vector")
            if row["metrics"]["steps_per_policy"] != index * 128 * 128:
                raise ValueError("short learning changed the complete per-policy training work")
            for name in ("observation_variance_mean", "reward_variance"):
                values = row.get(name)
                if not isinstance(values, list) or len(values) != copies or any(value < 0 for value in values):
                    raise ValueError("short learning lacks independent parameter/statistics diagnostics")
            for name in previous_counts:
                values = row[name]
                if (not isinstance(values, list) or len(values) != copies
                        or any(value <= previous for value, previous in zip(values, previous_counts[name]))):
                    raise ValueError("short learning observation and reward statistics must advance for every copy")
                previous_counts[name] = values
            keys = flattened_numbers(row["environment_key"])
            if (len(keys) != copies * 128 * 2 or any(type(value) is not int or not 0 <= value < 2 ** 32 for value in keys)
                    or len({tuple(keys[copy * 256:(copy + 1) * 256]) for copy in range(copies)}) != copies):
                raise ValueError("short learning changed its independent exact per-copy environment keys")
            for name in ("frame_count", "step_count", "reset_count"):
                values = flattened_numbers(row[name])
                if len(values) != copies * 128 or any(type(value) is not int or value < 0 for value in values):
                    raise ValueError("short learning lacks exact per-lane game counters")
            for name, count in (("action", 18), ("room", 33)):
                histogram = row.get("boundary_" + name + "_histogram")
                unknown = row.get("boundary_" + name + "_unknown_count")
                bins = row.get("boundary_" + name + "_bins")
                if (bins != list(range(count)) or any(type(value) is not int for value in bins)
                        or not isinstance(histogram, list) or len(histogram) != copies
                        or unknown != [0] * copies or any(type(value) is not int for value in unknown)
                        or any(len(values) != count or any(type(value) is not int or value < 0 for value in values)
                               or sum(values) != 128 for values in histogram)):
                    raise ValueError("short learning boundary bins must cover every observed action and room exactly once")
            for copy in range(copies):
                intrinsic[copy].append(row["metrics"]["intrinsic_reward_mean"][copy])
                action_counts[copy] = [first + second for first, second in zip(action_counts[copy], row["boundary_action_histogram"][copy])]
        if (any(sum(value > 0 for value in counts) < 2 for counts in action_counts)
                or any(statistics.pvariance(values) <= 0 for values in intrinsic)
                or any(value <= 0 for value in current[-1]["observation_variance_mean"] + current[-1]["reward_variance"])):
            raise ValueError("short learning has missing parameter updates, action collapse or absent intrinsic/statistics dynamics")
        for stage in ("initial", "latest"):
            precision = final.get("precision", {}).get(stage, {})
            if (precision.get("all_state_finite") is not True or precision.get("network_dtype") != "float32"
                    or precision.get("statistics_dtype") != "float64"
                    or any(type(precision.get(name)) is not int or precision[name] < 1
                           for name in ("network_leaves", "statistics_leaves", "full_state_leaves"))):
                raise ValueError("short learning lost actual full-state precision or finiteness evidence")
        replay = final.get("replay", {})
        if (replay.get("passed") is not True or replay.get("window_updates") != declared["replay_window_updates"]
                or replay.get("from_iteration") != updates - declared["replay_window_updates"]
                or replay.get("to_iteration") != updates or replay.get("scope") != "same_implementation_same_full_state_inputs"
                or replay.get("agreement_policy") != module.REPLAY_AGREEMENT
                or type(replay.get("agreement", {}).get("leaves")) is not int or replay["agreement"]["leaves"] < 1):
            raise ValueError("short learning lacks strict same-implementation full-state replay")
        observer = final.get("observer_agreement", {})
        expected_reports = 0 if configurations[("reference", "candidate").index(arm)]["execution"]["report_readout"] == "none" else declared["complete_window_updates"] // configurations[("reference", "candidate").index(arm)]["execution"]["report_every_iterations"]
        if (observer.get("passed") is not True or observer.get("window_updates") != declared["complete_window_updates"]
                or observer.get("scope") != "instrumented_window_versus_actual_selected_Executor_run_same_full_state"
                or observer.get("automatic_tolerance_adjustment") is not False
                or observer.get("agreement_policy") != module.OBSERVER_AGREEMENT
                or type(observer.get("state_and_final_metrics", {}).get("leaves")) is not int
                or observer["state_and_final_metrics"]["leaves"] < 1 or observer.get("report_rows") != expected_reports
                or expected_reports and observer.get("selected_reports", {}).get("leaves", 0) < 1):
            raise ValueError("short learning observer must match actual selected Executor.run and its report outputs")
        warmup = result.get("warmup", {}).get(arm, [])
        if len(warmup) < 2 or warmup[-1].get("cache_size") != warmup[-2].get("cache_size"):
            raise ValueError("short learning requires at least two stable disposable compilation warmups")
    if len(rows) != 2 * updates:
        raise ValueError("short learning curves contain an unknown or extra trajectory")
    reports = validate_learning_reports(result, path, rows, configurations, evidence)

    # A separate reviewer must compare all learning evidence; zero extrinsic reward alone is insufficient.
    analysis = evidence.read(specification["learning_analysis_record"])
    required_topics = {"losses", "policy_updates", "predictor_updates", "intrinsic_statistics",
                       "action_distributions", "episodes_and_exploration", "trajectory_differences"}
    findings = analysis.get("comparative_findings", {})
    reviewer = analysis.get("reviewer", {})
    if (type(analysis.get("schema_version")) is not int or analysis["schema_version"] != 1
            or analysis.get("kind") != "independent_short_learning_analysis"
            or analysis.get("decision") != "accepted_short_learning_checks"
            or analysis.get("result_record") != {"path": evidence.path_identity(path), "sha256": hashlib.sha256(evidence.file(path)).hexdigest()}
            or analysis.get("curves") != {"path": evidence.path_identity(curves["path"]), "sha256": curves["sha256"]}
            or analysis.get("reports") != reports
            or analysis.get("source_commit") != specification["source_commit"]
            or analysis.get("hardware_class") != specification["hardware_class"]
            or analysis.get("configuration_digest") != canonical_digest(selected) or analysis.get("copy_seeds") != plan["copy_seeds"]
            or analysis.get("learning_equivalence_established") is not False or analysis.get("zero_reward_only_evidence") is not False
            or not required_topics <= set(findings) or any(not isinstance(findings[name], str) or not findings[name].strip() for name in required_topics)
            or reviewer.get("independent_of_producer") is not True or not isinstance(reviewer.get("identity"), str) or not reviewer["identity"].strip()
            or analysis.get("unresolved_anomalies") != []):
        raise ValueError("short learning requires a separate complete comparative analysis with no unresolved anomalies")
    return {"passed": True, "process": list(process), "updates_per_arm": updates, "copies": copies,
            "learning_equivalence_established": False, "analysis": evidence.path_identity(specification["learning_analysis_record"]),
            "curves": {"path": evidence.path_identity(curves["path"]), "sha256": curves["sha256"]}, "reports": reports}


def fixed_total_group_binding(path, specification, expected_cards, total, evidence):
    """Verify one fixed-total group and all its separately hashed physical-GPU worker results."""
    directory = Path(path).parent
    result, plan = evidence.read(path), evidence.read(directory / "scaling_plan.json")
    identity(result, "jaguar03_rtx_a4500", specification["source_commit"], specification["runtime_environment"])
    process = execution_identity(result)
    verify_completed_study(plan.get("completed_study"), specification["source_commit"], specification["runtime_environment"], evidence,
        before_seconds=result["execution_identity"]["started_at_seconds"],
        registry_sha256=hashlib.sha256(evidence.file(specification["study_registry"])).hexdigest(),
        hardware_class="jaguar03_rtx_a4500", selected_configuration=result["configuration"])
    copies = total // expected_cards
    if (Path(path).name != "result.json" or directory.parent != Path(plan.get("campaign_run_dir", "")) / "data"
            or directory.name != plan.get("unit_id") or plan.get("kind") != "synchronized_a4500_scaling_measurement"
            or type(plan.get("schema_version")) is not int or plan["schema_version"] != 1
            or evidence.read(directory / "scaling_prepared.json") != {
                "plan_sha256": hashlib.sha256(evidence.file(directory / "scaling_plan.json")).hexdigest()}
            or plan.get("source_commit") != specification["source_commit"]
            or plan.get("hardware_class") != "jaguar03_rtx_a4500" or plan.get("compilation_cache_policy") != "disabled"
            or result.get("completed") is not True or plan.get("gpu_count") != expected_cards or result.get("gpu_count") != expected_cards
            or plan.get("configuration") != result.get("configuration") or result["configuration"]["learner"]["n_copies"] != copies
            or result.get("source_root") != plan.get("source_root")
            or plan.get("rounds") != 7 or plan.get("minimum_updates_per_round") != 10
            or plan.get("iterations") != complete_window([result["configuration"]], 10)
            or result.get("protocol") != {"rounds": 7, "requested_minimum_updates_per_round": 10,
                "actual_updates_per_round": plan["iterations"], "group_clock": "one shared host monotonic clock"}
            or result.get("normalization_warmup_iterations") != 50 or result.get("training_start_iteration") != 0
            or plan.get("helpers") != result.get("helpers") or runtime_environment(plan["runtime_environment"]) != specification["runtime_environment"]):
        raise ValueError("fixed-total scaling changed its exact prepared group, source, configuration or protocol")
    protocol = plan.get("scaling_protocol", {})
    seeds = [seed for block in plan.get("copy_seeds", []) for seed in block]
    if (protocol != {"kind": "fixed_total_copies", "total_copies": total, "global_copy_seeds": seeds,
                     "gradient_averaging": False, "measurement_scope": "synchronized_gpu_group_wall_time"}
            or result.get("scaling_protocol") != protocol or len(seeds) != total or len(set(seeds)) != total
            or any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in seeds)
            or result.get("copy_seeds") != plan["copy_seeds"]
            or len(plan.get("gpu_uuids", [])) != expected_cards or len(set(plan["gpu_uuids"])) != expected_cards
            or result.get("gpu_uuids") != plan["gpu_uuids"]):
        raise ValueError("fixed-total scaling changed its stable global policy identities or distinct physical GPUs")
    for name, core in (("core", True), ("paired_source", False), ("helpers", False)):
        committed_measurement_sources(plan[name], evidence, core=core)
    if any((plan[name]["source_root"], plan[name]["source_commit"]) != (plan["source_root"], plan["source_commit"])
           for name in ("core", "paired_source")):
        raise ValueError("fixed-total scaling changed its frozen numerical root")
    for relative, digest in plan["helpers"]["files"].items():
        if hashlib.sha256(evidence.file(directory / "code" / Path(relative).name)).hexdigest() != digest:
            raise ValueError("fixed-total scaling changed a staged executed helper")
    assignment = evidence.read(directory / "assignment.json")
    if (result.get("assignment_sha256") != hashlib.sha256(evidence.file(directory / "assignment.json")).hexdigest()
            or assignment.get("plan_sha256") != hashlib.sha256(evidence.file(directory / "scaling_plan.json")).hexdigest()
            or assignment.get("gpu_uuids") != plan["gpu_uuids"]
            or (assignment.get("hostname"), assignment.get("parent_pid"), assignment.get("parent_process_start_time")) != process):
        raise ValueError("fixed-total scaling changed its original physical parent assignment")
    if len(result.get("workers", [])) != expected_cards:
        raise ValueError("fixed-total scaling omitted a completed numerical worker")
    workers = []
    for index, receipt in enumerate(result["workers"]):
        worker_directory = directory / "workers" / ("physical_gpu_" + str(index + 1))
        worker_path = worker_directory / "result.json"
        worker = evidence.read(worker_path)
        hardware = evidence.read(worker_directory / "hardware.json")
        backend = evidence.read(worker_directory / "installed_backend.json")
        loaded = evidence.read(worker_directory / "loaded_gpu_libraries.json")
        warmup = evidence.read(worker_directory / "warmup.json")
        query = list(csv.reader(hardware.get("gpu_query", "").splitlines()))
        if len(query) != 1 or len(query[0]) < 3:
            raise ValueError("fixed-total worker needs its original memory.total query")
        gpu, model, memory_total = [value.strip() for value in query[0][:3]]
        size = re.fullmatch(r"(\d+) MiB", memory_total)
        if not size or gpu != receipt.get("gpu_uuid") or "A4500" not in model or int(size.group(1)) <= 0:
            raise ValueError("fixed-total worker hardware UUID, type or physical total memory changed")
        total_bytes = int(size.group(1)) * 2 ** 20
        if any(row.get("gpu_uuid") != gpu or row.get("memory", {}).get("gpu_uuid") != gpu
                or any(sample.get("gpu_uuid") != gpu or sample.get("total_bytes") != total_bytes
                       for sample in row.get("memory", {}).get("device_memory_samples", []))
               for row in worker.get("rounds", [])):
            raise ValueError("fixed-total worker round/sample UUID or physical memory.total differs from its assignment")
        if (receipt.get("worker_result_sha256") != hashlib.sha256(evidence.file(worker_path)).hexdigest()
                or receipt.get("gpu_uuid") != plan["gpu_uuids"][index] or worker.get("gpu_uuid") != receipt["gpu_uuid"]
                or worker.get("completed") is not True or worker.get("configuration") != plan["configuration"]
                or worker.get("copy_seeds") != plan["copy_seeds"][index]
                or worker.get("source_commit") != specification["source_commit"]
                or runtime_environment(worker["runtime_environment"]) != specification["runtime_environment"]
                or any(worker.get("hardware", {}).get(name) != hardware.get(name) for name in worker["hardware"])
                or hardware.get("gpu_uuid") != receipt["gpu_uuid"] or hardware.get("hostname") != "jaguar03"
                or type(hardware.get("pid")) is not int or hardware["pid"] < 1
                or not isinstance(hardware.get("process_start_time"), str) or not hardware["process_start_time"]
                or hardware["pid"] == process[1] or any(previous["hardware"]["pid"] == hardware["pid"] for previous in workers)
                or worker.get("loaded_gpu_libraries") != loaded or loaded.get("pid") != hardware["pid"]
                or len(warmup) < 2 or warmup[-1].get("cache_size") != warmup[-2].get("cache_size")
                or any(pair["workers"][index] != observed for pair, observed in zip(result["rounds"], worker["rounds"]))
                or len(worker.get("rounds", [])) != len(result["rounds"])
                or any(row.get("pid") != hardware["pid"] or row.get("process_start_time") != hardware["process_start_time"]
                       or row.get("copy_seeds") != worker["copy_seeds"] for row in worker["rounds"])):
            raise ValueError("fixed-total scaling changed a raw worker process, policy identity or complete-round result")
        identity_fields = {"assets": evidence.read(worker_directory / "asset_receipt.json"), "backend": backend}
        if workers and identity_fields != workers[0]["identity"]:
            raise ValueError("fixed-total scaling workers crossed asset or backend groups")
        workers.append({"identity": identity_fields, "hardware": hardware})
    identities = {process} | {(row["hardware"]["hostname"], row["hardware"]["pid"], row["hardware"]["process_start_time"])
                             for row in workers}
    return result, plan, process, workers[0]["identity"], seeds, identities


def validate_fixed_total_scaling(specification, selected, evidence):
    """Require equal total policy work, or exact-eight capacity evidence without changing packing choice."""
    fixed = specification["fixed_total_scaling"]
    if not isinstance(fixed, dict):
        raise ValueError("fixed-total scaling requires explicit measured or verified not-applicable evidence")
    common = {"source_commit": specification["source_commit"], "hardware_class": "jaguar03_rtx_a4500",
              "runtime_environment": specification["runtime_environment"]}
    selected_context = {**specification["contexts"]["jaguar03_rtx_a4500"], **common,
        "completed_registry_sha256": hashlib.sha256(evidence.file(specification["study_registry"])).hexdigest()}
    confirmed = validate_final_confirmation(selected_context, evidence)
    if fixed.get("status") == "not_applicable":
        if set(fixed) != {"status", "eligibility_kind", "eligibility_record"}:
            raise ValueError("fixed-total not-applicable evidence needs its exact eight-copy eligibility record")
        if fixed["eligibility_kind"] == "completed_below_memory_headroom":
            record, plan, process, assets = measurement_binding(fixed["eligibility_record"], "packing-point", common, evidence)
            point = packing_point(fixed["eligibility_record"], "jaguar03_rtx_a4500", common["source_commit"], common["runtime_environment"], evidence, require_headroom=False)
            if point["copies"] != 8 or point["free_memory_fraction"] >= .15:
                raise ValueError("fixed-total ineligibility requires actual exact-eight memory below fifteen-percent headroom")
            configuration_value = record["configuration"]
            eligible_identity = (plan["gpu_uuid"], assets, record["installed_backend"])
            observed_start = record["execution_identity"]["started_at_seconds"]
        elif fixed["eligibility_kind"] == "instrumented_exact_eight_capacity_exclusion":
            module_spec = importlib.util.spec_from_file_location("final_exact_eight_capacity", Path(__file__).with_name("device_memory_limit.py"))
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            point = module.validate_exclusion(fixed["eligibility_record"], common, evidence)
            if point["copies"] != 8:
                raise ValueError("a smaller failed count cannot establish exact-eight fixed-total ineligibility")
            configuration_value = point["configuration"]
            directory = Path(fixed["eligibility_record"]).parent
            eligible_identity = (point["gpu_uuid"], point["assets"]["identity"], evidence.read(directory / "installed_backend.json"))
            observed_start = evidence.read(directory / "capacity_parent.json")["started_at_seconds"]
            process = tuple(point["process"])
        else:
            raise ValueError("fixed-total ineligibility requires reviewed exact-eight physical memory evidence")
        if configuration_without_copies(configuration_value) != configuration_without_copies(selected):
            raise ValueError("exact-eight eligibility changed the final selected implementation")
        if (eligible_identity != (confirmed["gpu_uuid"], confirmed["assets"], confirmed["backend"])
                or process == confirmed["process"] or type(observed_start) not in (int, float)
                or not math.isfinite(observed_start) or observed_start <= confirmed["completed_study_reviewed_at_seconds"]):
            raise ValueError("exact-eight eligibility must be fresh after fifty ideas on the selected physical/backend/asset group")
        return {"complete": True, "status": "not_applicable", "exact_total_copies": 8,
                "eligibility_kind": fixed["eligibility_kind"], "eligibility_record": evidence.path_identity(fixed["eligibility_record"]),
                "reason": "exact_eight_copies_do_not_fit_one_gpu_with_fifteen_percent_memory_headroom"}
    if set(fixed) != {"status", "total_copies", "single_gpu_record", "eight_gpu_record"} or fixed.get("status") != "measured":
        raise ValueError("fixed-total comparison requires two exact independently measured GPU-group records")
    total = fixed["total_copies"]
    if type(total) is not int or total < 8 or total % 8:
        raise ValueError("fixed-total policy count must be a positive multiple of eight")
    groups = [fixed_total_group_binding(fixed[name], specification, count, total, evidence)
              for name, count in (("single_gpu_record", 1), ("eight_gpu_record", 8))]
    if groups[0][5] & groups[1][5] or groups[0][3] != groups[1][3] or groups[0][4] != groups[1][4]:
        raise ValueError("fixed-total groups must be fresh processes with the same assets/backend/global copy seeds")
    if (groups[0][0]["gpu_uuids"] != [confirmed["gpu_uuid"]]
            or confirmed["gpu_uuid"] not in groups[1][0]["gpu_uuids"]
            or {key: value for key, value in groups[0][3]["assets"].items() if key != "root"} != confirmed["assets"]
            or groups[0][3]["backend"] != confirmed["backend"]):
        raise ValueError("fixed-total scaling must retain the selected physical GPU, full backend and asset identity")
    rounds = [group[0].get("rounds", []) for group in groups]
    if len(rounds[0]) < 7 or len(rounds[0]) != len(rounds[1]):
        raise ValueError("fixed-total comparison requires seven matched complete common-wall-time rounds")
    speedups = []
    previous_ends = [0., 0.]
    for one, eight in zip(*rounds):
        durations = []
        iterations = None
        for row, cards, group in ((one, 1, groups[0]), (eight, 8, groups[1])):
            group_index = 0 if cards == 1 else 1
            if configuration_without_copies(group[0]["configuration"]) != configuration_without_copies(selected):
                raise ValueError("fixed-total scaling changed the final selected optimized implementation")
            duration = positive(row.get("seconds"), "fixed-total common wall duration")
            start, end = row.get("start_time_seconds"), row.get("end_time_seconds")
            workers = row.get("workers", [])
            execution = group[0]["execution_identity"]
            if (row.get("measurement_scope") != "synchronized_gpu_group_wall_time" or len(workers) != cards
                    or len({worker.get("gpu_uuid") for worker in workers}) != cards
                    or type(start) not in (int, float) or type(end) not in (int, float)
                    or not math.isfinite(start) or not math.isfinite(end) or abs(end - start - duration) > max(1e-6, duration * 1e-6)
                    or start < execution["started_at_seconds"] or end > execution["finished_at_seconds"]
                    or start < previous_ends[group_index]):
                raise ValueError("fixed-total scaling lacks its actual concurrent common wall interval")
            previous_ends[group_index] = end
            for worker in workers:
                measured_round(worker, total // cards)
                if iterations is None:
                    iterations = worker["iterations"]
                if (worker["iterations"] != iterations or worker.get("gpu_uuid") != worker["memory"]["gpu_uuid"]
                        or worker.get("start_time_seconds", -1) < start or worker.get("end_time_seconds", math.inf) > end
                        or abs(worker["end_time_seconds"] - worker["start_time_seconds"] - worker["seconds"]) > max(1e-6, worker["seconds"] * 1e-6)):
                    raise ValueError("fixed-total scaling changed matched work, physical memory or contained worker timings")
                samples = worker["memory"].get("device_memory_samples", [])
                if len(samples) < 2 or any(sample.get("gpu_uuid") != worker["gpu_uuid"]
                        or any(type(sample.get(name)) is not int for name in ("free_bytes", "used_bytes", "total_bytes"))
                        or sample["total_bytes"] <= 0 or min(sample["free_bytes"], sample["used_bytes"]) < 0
                        or sample["free_bytes"] + sample["used_bytes"] > sample["total_bytes"]
                        or sample["free_bytes"] / sample["total_bytes"] < .15 for sample in samples):
                    raise ValueError("fixed-total arrangements must both fit with fifteen-percent actual memory headroom")
            if max(worker["start_time_seconds"] for worker in workers) >= min(worker["end_time_seconds"] for worker in workers):
                raise ValueError("fixed-total workers did not execute concurrently")
            if sum(worker["training_transitions"] for worker in workers) != total * 128 * 128 * iterations:
                raise ValueError("fixed-total comparison changed its total completed training work")
            durations.append(duration)
        speedups.append(durations[0] / durations[1])
    return {"complete": True, "status": "measured", "total_copies": total,
            "rounds": len(rounds[0]), "global_copy_seeds": groups[0][4],
            "median_fixed_total_speedup": statistics.median(speedups)}


def validate_final_verification(specification, evidence=None, specification_path=None):
    """Derive all five final gates from source-matched raw evidence rather than supplied Boolean flags."""
    evidence = Evidence() if evidence is None else evidence
    if specification_path is not None and evidence.read(specification_path) != specification:
        raise ValueError("the final evidence specification changed")
    source = specification["source_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", source) or set(specification["contexts"]) != set(CONTEXTS):
        raise ValueError("final verification needs one full source commit and both hardware contexts")
    runtime = runtime_environment(specification["runtime_environment"])
    final = "final_protocol" in specification
    if final:
        final_protocol(specification["final_protocol"])
    require_fifty_investigated(evidence.read(specification["study_registry"]), evidence)
    selected = {}
    learning = {}
    for context in CONTEXTS:
        common = {**specification["contexts"][context], "source_commit": source,
                  "hardware_class": context, "runtime_environment": runtime}
        if final:
            common["completed_registry_sha256"] = hashlib.sha256(evidence.file(specification["study_registry"])).hexdigest()
        combination = validate_combination(common, evidence, final=final)
        packing = validate_packing(common, evidence)
        if not packing["complete"] or packing["configuration"] != combination["configuration"]:
            raise ValueError("the selected packing count needs its own combination correctness and confirmation")
        selected[context] = {"configuration": combination["configuration"], "copies": combination["copies"],
                             "combination": combination, "packing": packing}
        if final:
            confirmation = validate_final_confirmation(common, evidence)
            unused, unused_plan, correctness_process, correctness_assets = measurement_binding(common["correctness_record"], "correctness", common, evidence)
            learning[context] = validate_short_learning(common, confirmation, evidence)
            for path in common["packing_records"] + common["packing_confirmation_records"]:
                record, plan, process, assets = measurement_binding(path, "packing-point", common, evidence)
                if (record["execution_identity"]["started_at_seconds"] <= confirmation["completed_study_reviewed_at_seconds"]
                        or assets != confirmation["assets"] or record["installed_backend"] != confirmation["backend"]
                        or plan["gpu_uuid"] != confirmation["gpu_uuid"]):
                    raise ValueError("final packing must be freshly repeated after fifty ideas on the same physical/backend/asset group")
            used = {tuple(point["process"]) for point in packing["points"] + packing["confirmation"]}
            if (correctness_assets != confirmation["assets"]
                    or tuple(learning[context]["process"]) in used | {confirmation["process"], correctness_process}):
                raise ValueError("final learning reused another measurement process or changed its assets")
    scaling = validate_scaling({**specification["scaling"], "source_commit": source, "runtime_environment": runtime}, evidence)
    if scaling["configuration"] != selected["jaguar03_rtx_a4500"]["configuration"]:
        raise ValueError("one-versus-eight scaling did not use the final selected A4500 configuration")
    fixed_total = validate_fixed_total_scaling({**specification, "runtime_environment": runtime},
        selected["jaguar03_rtx_a4500"]["configuration"], evidence) if final else None
    result = {"schema_version": 2 if final else 1, "kind": "final_verification", "source_commit": source,
            **{name: True for name in FINAL_FLAGS}, "runtime_environment": runtime,
            "specification": specification, "specification_digest": canonical_digest(specification),
            "specification_path": evidence.path_identity(specification_path) if specification_path is not None else None,
            "cohort_runtime_provenance": [{"result_path": path, **evidence.runtime_provenance[path]}
                                          for path in sorted(evidence.runtime_provenance)],
            "selected": selected, "scaling": scaling, "evidence": evidence.records()}
    if final:
        result.update(short_learning_checks_passed=True, fixed_total_scaling_complete=True,
                      learning_checks=learning, fixed_total_scaling=fixed_total)
    return result


def validate_training_admission(path, resolver=None, *, historical_read_only=False):
    """Revalidate original final evidence before admitting the exact source, configuration and policy seeds."""
    evidence = Evidence(resolver)
    specification = evidence.read(path)
    saved = evidence.read(specification["final_verification"])
    if not historical_read_only:
        require_current_final(saved)
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


def validate_training_completion(specification_path, unit_directory, resolver=None, *, historical_read_only=False):
    """Verify the actual 2B schedule and checkpoint bytes without importing or unpickling arrays."""
    admitted = validate_training_admission(specification_path, resolver=resolver, historical_read_only=historical_read_only)
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
    operation.add_argument("--completed-study-proof", action="store_true")
    operation.add_argument("--training-completion", type=Path, metavar="UNIT_DIRECTORY")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evidence-manifest", type=Path)
    parser.add_argument("--evidence-manifest-sha256")
    parser.add_argument("--historical-read-only", action="store_true")
    args = parser.parse_args()
    if args.historical_read_only and (args.training_admission or args.completed_study_proof):
        parser.error("historical read-only inspection cannot create a current admission or completed-study proof")
    if args.completed_study_proof and (args.output is None or args.evidence_manifest is not None):
        parser.error("completed-study proof needs a fresh output under owned UVA compute and original local evidence")
    if args.evidence_manifest_sha256 and args.evidence_manifest is None:
        parser.error("an expected evidence manifest hash requires --evidence-manifest")
    resolver = None
    if args.evidence_manifest is not None:
        from portable_evidence import ExactPathResolver
        resolver = ExactPathResolver(args.evidence_manifest, args.evidence_manifest_sha256)
    if args.completed_study_proof:
        result = produce_completed_study_proof(json.loads(args.specification.read_bytes()), args.output)
    elif args.training_completion:
        result = validate_training_completion(args.specification, args.training_completion, resolver=resolver,
                                             historical_read_only=args.historical_read_only)
    elif args.training_admission:
        result = validate_training_admission(args.specification, resolver=resolver)
    else:
        evidence = Evidence(resolver)
        result = validate_final_verification(evidence.read(args.specification), evidence, args.specification)
        if not args.historical_read_only:
            require_current_final(result)
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
