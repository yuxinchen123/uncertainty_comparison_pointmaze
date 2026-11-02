"""Require genuine direct common evidence before one new H100 profile.

Historical consumers run in fresh standard-library children using their own
staged code. A profile is never relabelled as a confirmation. This module
selects no configuration, submits no work and grants no scientific admission.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

NAME = "direct_profile_prerequisites.py"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
CAPABILITY = "exact_historical_direct_consumers_and_profile_backend_version_1"
CARD_ONE = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
PROOF_NAME = "direct_profile_prerequisite_validation.json"
NUMERICAL = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "cupy", "scipy", "jaxtari", "exploration_platform"}
ZERO_FIELDS = ("scientific_credit", "case_credit", "idea_credit", "throughput_credit", "final_training_credit", "site_qualification_credit")


def require(value, message):
    """Reject an unknown or incomplete prerequisite rather than substitute another target."""
    if not value:
        raise ValueError(message)


def encoded(value):
    """Retain stable finite metadata encoding for every child and saved hash."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(raw):
    """Identify the exact bytes used by a saved input or child return."""
    return hashlib.sha256(raw).hexdigest()


def read(path):
    """Read one ordinary exact JSON document with unique keys and finite numbers."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "exact ordinary metadata required")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate profile prerequisite key"); value[key] = item
        return value
    raw = path.read_bytes()
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite prerequisite metadata"))), raw


def reference(path):
    """Keep an existing metadata artifact's exact actual path and bytes."""
    unused, raw = read(path)
    return {"path": str(path), "sha256": digest(raw)}


def checked(value):
    """Recheck the actual artifact named by a reviewed reference."""
    require(type(value) is dict and set(value) == {"path", "sha256"}
        and reference(value["path"]) == value, "an actual reviewed profile input changed")
    return read(value["path"])[0]


def load_exact(path, name):
    """Execute only exact source bytes, without accepting cached bytecode or ambient modules."""
    path = Path(path)
    require(path.resolve(strict=True) == path and path.is_file(), "exact ordinary staged source required")
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification); sys.modules[name] = module
    try:
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None); raise
    return module


def contract(plan):
    """Keep this capability limited to the explicit direct full-configuration profile."""
    value = plan.get("direct_profile_prerequisites")
    require(plan.get("kind") == "selected_configuration_profile" and plan.get("hardware_class") == "h100"
        and plan.get("gpu_uuid") == CARD_ONE
        and plan.get("direct_binding", {}).get("consumer") == "component_profile"
        and not any(key in plan for key in ("configured_isolation", "configured_group", "device_memory_limit_protocol",
            "final_protocol", "final_confirmation_protocol", "completed_study", "scaling_protocol")),
        "direct profile prerequisites cannot authorize another target, final run, packing or group")
    require(type(value) is dict and set(value) == {"schema_version", "capability", "mode", "correctness", "timing",
            "lifecycle", "capture_qualification", "metadata_timeout_seconds"}
        and type(value["schema_version"]) is int and value["schema_version"] == 1
        and value["capability"] == CAPABILITY and value["mode"] in {"qualification", "measurement"},
        "new direct profiles require the separate version-one prerequisite protocol")
    require(len(plan["helpers"]["files"]) == 28
        and plan["helpers"]["files"].get(PREFIX + NAME) == digest(Path(__file__).read_bytes())
        and PREFIX + "direct_backend_image.py" in plan["helpers"]["files"]
        and PREFIX + "direct_common_confirmation.py" not in plan["helpers"]["files"],
        "the exact28 current helpers and separate historical closures are required")
    for name in ("correctness", "timing", "lifecycle"):
        require(type(value[name]) is dict and set(value[name]) == {"result", "review"},
            "each actual prerequisite requires its result and genuine independent review")
    if value["mode"] == "qualification":
        require(value["capture_qualification"] is None, "a zero-credit capture qualification cannot require itself")
    else:
        require(type(value["capture_qualification"]) is dict
            and set(value["capture_qualification"]) == {"result", "review"},
            "scientific profiling requires a separate actual reviewed full capture qualification")
    require(type(value["metadata_timeout_seconds"]) is int and value["metadata_timeout_seconds"] > 0,
        "supply an explicit finite positive metadata-child time budget")
    copies = plan["configuration"]["learner"]["n_copies"]
    require(type(copies) is int and copies in (1, 2) and plan["copy_seeds"] == list(range(1, copies + 1))
        and all(type(seed) is int for seed in plan["copy_seeds"]),
        "this initial common profile must retain its exact one/two independent copy identities")
    require(type(plan.get("profile_updates")) is int and plan["profile_updates"] > 0
        and plan["configuration"]["learner"]["n_envs"] == plan["configuration"]["environment"]["num_envs"] == 128
        and plan["configuration"]["learner"]["num_steps"] == 128
        and plan["configuration"]["learner"]["update_epochs"] == plan["configuration"]["learner"]["num_minibatches"] == 4
        and plan["configuration"]["learner"]["observation_warmup_iterations"] == 50
        and plan["configuration"]["learner"]["total_timesteps"] == 2_000_000_000,
        "profiles keep the complete original configuration and annealing horizon")
    require(type(plan.get("direct_backend_image")) is dict and plan["direct_backend_image"].get("schema_version") == 2,
        "a new direct profile must declare its actual pre-GPU package-image record")
    return value


def metadata_review(result_ref, review_ref, kind):
    """Validate genuine earlier review metadata without recomputing scientific evidence during preparation."""
    result, review = checked(result_ref), checked(review_ref)
    require(type(review.get("schema_version")) is int and review["schema_version"] == 1
        and review.get("kind") == kind and review.get("result") == result_ref
        and review.get("approved") is True and review.get("passed") is True
        and review.get("independent_of_producer") is True
        and review.get("unresolved_anomalies") == [] and review.get("admission_granted") is False
        and review.get("worker_admitted") is False
        and type(review.get("reviewer_identity")) is str and bool(review["reviewer_identity"].strip()),
        "an exact genuine complete independent prerequisite review is required")
    reviewed = review.get("reviewed_at_seconds")
    require(type(reviewed) in (int, float) and math.isfinite(reviewed)
        and reviewed >= result["execution_identity"]["finished_at_seconds"],
        "the prerequisite review predates the actual completed work")
    require(all(type(review.get(name)) is int and review[name] == 0 for name in ZERO_FIELDS),
        "a prerequisite review cannot grant profile/scientific admission")
    raw = checked(review["raw_audit"])
    require(raw.get("passed") is True and raw.get("result") == result_ref,
        "a genuine prerequisite review must bind its complete actual raw audit")
    if kind in {"independent_actual_direct_common_configuration_correctness_review",
            "independent_actual_direct_common_configuration_timing_review"}:
        require(raw.get("unchanged_staged_consumer_passed") is True,
            "accepted common evidence requires the exact historical saved consumer")
    return result, review


def historical_metadata(plan):
    """Bind both accepted historical units, their own source closures and their exact configuration/seeds."""
    value = contract(plan)
    first, first_review = metadata_review(value["correctness"]["result"], value["correctness"]["review"],
        "independent_actual_direct_common_configuration_correctness_review")
    second, second_review = metadata_review(value["timing"]["result"], value["timing"]["review"],
        "independent_actual_direct_common_configuration_timing_review")
    backend = load_exact(Path(__file__).with_name("direct_backend_image.py"), "profile_prepared_backend_metadata")
    current_package = backend.expected_for_plan(plan)
    package_fields = ("selected_python", "site_packages", "image", "distribution")
    qualification_directory = Path(plan["direct_qualification_inputs"]["result"]["path"]).parent
    qualified = read(qualification_directory / "measurement_plan.json")[0]
    qualified_package = checked(qualified["direct_backend_image"]["expected"])
    require(all(encoded(current_package[key]) == encoded(qualified_package[key]) for key in package_fields),
        "the new profile package differs from the actual qualified package image or owning distribution")
    units = {}
    for name, result in (("correctness", first), ("timing", second)):
        directory = Path(value[name]["result"]["path"]).parent
        require(Path(value[name]["result"]["path"]).name == "result.json", "accepted common evidence must be an actual ordinary result")
        previous = read(directory / "measurement_plan.json")[0]
        historical_package = checked(previous["direct_backend_image"]["expected"])
        require(all(encoded(current_package[key]) == encoded(historical_package[key]) for key in package_fields),
            "the new profile package differs from its actual accepted common package")
        require(previous["kind"] == "selected_configuration_measurement"
            and previous["mode"] == ("correctness" if name == "correctness" else "confirmation")
            and "direct_binding" in previous and previous["gpu_uuid"] == CARD_ONE
            and previous["configuration"] == plan["configuration"] and previous["copy_seeds"] == plan["copy_seeds"]
            and all(previous[key] == plan[key] for key in ("campaign_run_dir", "source_root", "source_commit",
                "core", "paired_source", "python", "hardware_class", "runtime_environment", "asset_record", "direct_qualification_inputs")),
            "profile source/configuration/seeds differ from actual accepted common evidence")
        require(result.get("candidate") == plan["configuration"] and result.get("copy_seeds") == plan["copy_seeds"]
            and result.get("helpers") == previous["helpers"]
            and result.get("measurement_mode") == previous["mode"],
            "accepted result changed its own historical source/configuration")
        review = first_review if name == "correctness" else second_review
        require(review.get("plan") == reference(directory / "measurement_plan.json")
            and review.get("configuration") == previous["configuration"] and review.get("copy_seeds") == previous["copy_seeds"]
            and review.get("helper_commit") == previous["helpers"]["source_commit"],
            "genuine review does not name the exact historical unit")
        units[name] = {"directory": str(directory), "plan": reference(directory / "measurement_plan.json"),
            "helpers": previous["helpers"], "result": value[name]["result"], "review": value[name]["review"]}
    require(tuple(first["execution_identity"][key] for key in ("hostname", "pid", "process_start_time"))
        != tuple(second["execution_identity"][key] for key in ("hostname", "pid", "process_start_time"))
        and first.get("correctness_attempted") is True and first.get("correctness_passed") is True
        and first.get("implementation_distinct") is True and first.get("measurement_valid") is False
        and second.get("measurement_valid") is True and second.get("contaminated") is False
        and second.get("decision") == "accepted"
        and len(second.get("rounds", [])) == 7,
        "separate successful correctness and seven-pair accepted timing are mandatory")
    return units


def own_metadata_process():
    """Record only this metadata process, with its actual birth and all four kernel UIDs."""
    raw = Path("/proc/self/stat").read_text()
    lines = [line for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("Uid:")]
    require(os.getuid() == os.geteuid() == 2618919 and len(lines) == 1
        and list(map(int, lines[0].split()[1:])) == [2618919] * 4,
        "the actual metadata process must retain all four study-owner UIDs")
    return {"hostname": socket.gethostname().split(".")[0], "pid": os.getpid(),
        "process_start_time": raw.rsplit(")", 1)[1].split()[19], "uid": 2618919}


def historical_event_guard(guard, event, arguments, roots, nested_command):
    """Allow only the exact historical timing reader's required isolated child in addition to its Git reads."""
    if event == "subprocess.Popen" and nested_command is not None and arguments[1] == nested_command:
        require(arguments[0] == nested_command[0] and arguments[2] is None and arguments[3] is None,
            "the exact historical metadata child cannot replace its executable, working directory or environment")
        return
    guard.metadata_event_guard(event, arguments, roots)


def validate_child_proof(plan, phase, proof, parent, started, finished):
    """Recheck actual child claims and their historical inputs, rather than accept rehashed contradictory metadata."""
    value = contract(plan)
    require(type(proof) is dict and type(proof.get("schema_version")) is int and proof["schema_version"] == 1
        and proof.get("kind") == "exact_historical_direct_profile_prerequisite_validation"
        and proof.get("passed") is True and proof.get("phase") == phase
        and proof.get("profile_plan_sha256") == digest(encoded(plan))
        and type(parent) is int and parent > 0 and type(proof.get("parent_pid")) is int and proof["parent_pid"] == parent
        and type(proof.get("validation_pid")) is int and proof["validation_pid"] > 0 and proof["validation_pid"] != parent
        and proof.get("numerical_imports") == [] and type(proof.get("gpu_operations")) is int and proof["gpu_operations"] == 0
        and type(proof.get("scientific_credit")) is int and proof["scientific_credit"] == 0,
        "the historical metadata child changed its exact phase, success, process or operation claims")
    process = proof.get("validation_process", {})
    require(set(process) == {"hostname", "pid", "process_start_time", "uid"}
        and process["hostname"] == "serval02" and type(process["pid"]) is int and process["pid"] == proof["validation_pid"]
        and type(process["uid"]) is int and process["uid"] == 2618919
        and type(process["process_start_time"]) is str and re.fullmatch(r"[1-9][0-9]*", process["process_start_time"])
        and encoded(proof.get("compute")) == encoded({"hostname": "serval02", "uid": 2618919, "allocation_kind": "authorized_direct_compute"}),
        "the historical child lost its actual direct metadata process identity")
    child_start, child_finish = proof.get("started_at_seconds"), proof.get("finished_at_seconds")
    require(all(type(item) in (int, float) and math.isfinite(item) and item > 0
        for item in (started, child_start, child_finish, finished))
        and started <= child_start <= child_finish <= finished,
        "historical metadata child times do not fall inside the actual pre-numerical validation interval")
    pair = value[phase]
    unit = Path(pair["result"]["path"]).parent
    old = read(unit / "measurement_plan.json")[0]
    require(proof.get("historical_plan") == reference(unit / "measurement_plan.json")
        and proof.get("result") == pair["result"] and proof.get("helpers") == old["helpers"]
        and proof.get("resolved_binding") == reference(unit / "direct_device_binding.json")
        and encoded(proof.get("execution_identity")) == encoded(checked(pair["result"])["execution_identity"]),
        "historical child returned another accepted unit, actual binding or numerical execution")
    timing_unit = Path(value["timing"]["result"]["path"]).parent
    timing_plan = read(timing_unit / "measurement_plan.json")[0]
    guard_path = timing_unit / "code/direct_common_confirmation.py"
    require(digest(guard_path.read_bytes()) == timing_plan["helpers"]["files"].get(PREFIX + guard_path.name),
        "the exact historical bootstrap reader changed")
    guard = load_exact(guard_path, "profile_historical_bootstrap_reader")
    guard.validate_bootstrap_record(proof.get("stdlib_bootstrap"))
    if phase == "timing":
        baseline = proof.get("accepted_combination", {})
        require(baseline.get("configuration") == plan["configuration"] and baseline.get("hardware_class") == "h100"
            and baseline.get("source_commit") == plan["source_commit"] and baseline.get("runtime_environment") == plan["runtime_environment"]
            and type(baseline.get("copies")) is int and baseline["copies"] == plan["configuration"]["learner"]["n_copies"]
            and type(baseline.get("paired_rounds")) is int and baseline["paired_rounds"] == 7,
            "the historical timing child changed its actual accepted seven-pair common result")
    else:
        require(proof.get("accepted_combination") is None, "a non-timing child cannot claim a timing result")
    return proof


def configure_plan(plan, correctness_result, correctness_review, timing_result, timing_review,
        lifecycle_result, lifecycle_review, *, metadata_timeout_seconds, qualification=False, capture_result=None, capture_review=None):
    """Bind only actual supplied evidence; never choose settings or create future references."""
    require(all(path is not None for path in (correctness_result, correctness_review, timing_result,
        timing_review, lifecycle_result, lifecycle_review)), "all actual accepted common and lifecycle inputs are required")
    require(qualification and capture_result is None and capture_review is None
        or not qualification and capture_result is not None and capture_review is not None,
        "measurement needs a separate capture qualification; qualification cannot depend on itself")
    pair = lambda result, review: {"result": reference(result), "review": reference(review)}
    plan["direct_profile_prerequisites"] = {"schema_version": 1, "capability": CAPABILITY,
        "mode": "qualification" if qualification else "measurement", "metadata_timeout_seconds": metadata_timeout_seconds,
        "correctness": pair(correctness_result, correctness_review), "timing": pair(timing_result, timing_review),
        "lifecycle": pair(lifecycle_result, lifecycle_review),
        "capture_qualification": None if qualification else pair(capture_result, capture_review)}
    historical_metadata(plan)
    return plan


def lifecycle_metadata(plan):
    """Require real direct-tool descriptor/wait proof, without claiming that a CPU fixture proves CUDA."""
    value = contract(plan)
    result, review = metadata_review(value["lifecycle"]["result"], value["lifecycle"]["review"],
        "independent_actual_direct_nsight_lifecycle_review")
    require(result.get("kind") == "direct_nsight_private_descriptor_and_exact_wait_qualification"
        and result.get("completed") is True and result.get("hostname") == "serval02"
        and result.get("uid") == 2618919 and result.get("profile_tool") == plan["profile_tool"]
        and result.get("profile_entrypoint_sha256") == plan["helpers"]["files"][PREFIX + "profile_selected.py"]
        and result.get("wait_for_primary") is True and result.get("private_descriptor_survived") is True
        and result.get("exact_child_wait_passed") is True and result.get("actual_gpu_identity_verified") is False
        and result.get("gpu_standing_lock_custody_verified") is False
        and type(result.get("environment_transitions")) is int and result["environment_transitions"] == 0,
        "separately reviewed direct lifecycle proof is missing or does not match this exact tool/source")
    if value["mode"] == "measurement":
        capture, capture_review = metadata_review(value["capture_qualification"]["result"],
            value["capture_qualification"]["review"], "independent_actual_direct_full_profile_capture_qualification_review")
        directory = Path(value["capture_qualification"]["result"]["path"]).parent
        qualified = read(directory / "measurement_plan.json")[0]
        require(qualified.get("direct_profile_prerequisites", {}).get("mode") == "qualification"
            and capture.get("profile_qualification_only") is True and capture.get("measurement_valid") is False
            and capture.get("completed") is True and capture.get("capture_completed") is True
            and capture.get("direct_profile_backend_completion") == reference(directory / "direct_profile_backend_completion.json")
            and all(qualified[key] == plan[key] for key in ("source_root", "source_commit", "core", "paired_source", "helpers",
                "configuration", "copy_seeds", "runtime_environment", "asset_record", "python", "profile_tool"))
            and qualified["direct_profile_prerequisites"]["lifecycle"] == value["lifecycle"],
            "actual current-source full-configuration capture qualification is absent or differs")
    return {"lifecycle": value["lifecycle"], "capture_qualification": value["capture_qualification"]}


def compute_context(plan):
    """Allow metadata recomputation only on the approved direct compute host or exact owned CPU step."""
    require(os.getuid() == os.geteuid() == 2618919, "the exact study owner is required")
    host = socket.gethostname().split(".")[0]
    if host == "serval02":
        require(not any(name.startswith("SLURM_") for name in os.environ)
            and os.environ.get("CUDA_VISIBLE_DEVICES") == CARD_ONE,
            "the direct metadata child must retain its exact original selector")
        return {"hostname": host, "uid": 2618919, "allocation_kind": "authorized_direct_compute"}
    job, step = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID")
    require(job == "6783003" and type(step) is str and step.isdecimal(),
        "profile evidence is recomputed only on the exact existing owned CPU allocation")
    run = Path(plan["campaign_run_dir"])
    for name in ("submitted_jobids.txt", "submitted_jobids_uva.txt"):
        require((run / "slurm" / name).read_text().splitlines().count(job) == 1, "owned metadata ledger missing or duplicated")
    expected = "/system.slice/slurmstepd.scope/job_" + job + "/step_" + step + "/user/task_0"
    require(Path("/proc/self/cgroup").read_text().splitlines() == ["0::" + expected],
        "metadata child must retain its exact owned numeric CPU cgroup")
    return {"hostname": host, "uid": 2618919, "allocation_kind": "owned_cpu_step", "job_id": job, "step_id": step}


def historical_child(plan, phase):
    """Run the accepted unit's own exact staged consumer in this fresh metadata interpreter."""
    started = time.time()
    value = contract(plan)
    historical = historical_metadata(plan)
    lifecycle_metadata(plan)
    require(phase in {"correctness", "timing", "capture_qualification"}, "unknown historical prerequisite phase")
    timing_directory = Path(historical["timing"]["directory"])
    timing_plan = read(timing_directory / "measurement_plan.json")[0]
    guard_path = timing_directory / "code/direct_common_confirmation.py"
    require(digest(guard_path.read_bytes()) == timing_plan["helpers"]["files"].get(PREFIX + guard_path.name),
        "the accepted timing unit omitted its exact historical common reader")
    guard = load_exact(guard_path, "historical_direct_profile_common_guard")
    bootstrap = guard.stdlib_ctypes_bootstrap()
    context = compute_context(plan)
    roots = guard.declared_git_roots(timing_plan)
    if phase == "capture_qualification":
        # The full profile qualification has another exact current-source history namespace.
        unit = Path(value["capture_qualification"]["result"]["path"]).parent
        old = read(unit / "measurement_plan.json")[0]
        expected, declared = guard.pinned_history_declarations(old)
        roots.update(declared)
        for field in ("core", "paired_source", "helpers"):
            source = Path(old[field]["source_root"])
            require(source.is_absolute() and source.resolve(strict=True) == source and source.is_dir()
                and re.fullmatch(r"[0-9a-f]{40}", old[field]["source_commit"]), "invalid exact capture source root")
            roots.add(str(source))
        frozen = checked(old["direct_binding"]["frozen_inputs"])
        manifest = checked(frozen["evidence_manifest"])
        require(len(manifest["git_roots"]) == 8, "capture qualification omitted its original eight-root history")
        require(set(manifest) == {"schema_version", "kind", "files", "git_roots"}
            and type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
            and manifest["kind"] == "portable_raw_evidence", "unknown capture-qualification history manifest")
        seen, targets = set(), set()
        staging = Path(frozen["evidence_manifest"]["path"]).parent
        for row in manifest["git_roots"]:
            relative = row["relative_path"]
            require(row["original_root"] in expected and row["original_root"] not in seen
                and row["commit"] == expected[row["original_root"]]
                and type(relative) is str and relative and relative != "." and "\0" not in relative and not Path(relative).is_absolute()
                and ".." not in Path(relative).parts and str(Path(relative)) == relative,
                "capture qualification changed a pinned historical source")
            target = staging / relative
            require(target.resolve(strict=True) == target and target.is_relative_to(staging) and target.is_dir()
                and str(target) not in targets, "ordinary exact distinct capture history target required")
            roots.add(str(target)); targets.add(str(target)); seen.add(row["original_root"])
        require(seen == set(expected), "capture qualification must retain all original source names")
        result_ref = value["capture_qualification"]["result"]
    else:
        unit = Path(historical[phase]["directory"]); old = read(unit / "measurement_plan.json")[0]
        result_ref = value[phase]["result"]
    nested_command = ["/usr/bin/python3", "-I", "-S", "-B", str(guard_path), "--plan-stdin"] if phase == "timing" else None
    sys.addaudithook(lambda event, args: historical_event_guard(guard, event, args, roots, nested_command))
    require(not NUMERICAL.intersection(sys.modules), "historical metadata child imported numerical code")
    for relative, checksum in old["helpers"]["files"].items():
        require(digest((unit / "code" / Path(relative).name).read_bytes()) == checksum,
            "an exact historical staged helper changed before import")
    sys.path.insert(0, str(unit / "code"))
    support = load_exact(unit / "code/direct_measurement.py", "direct_measurement")
    result = checked(result_ref)
    returned, binding = support.verify_saved_measurement(unit, result)
    require(returned == old, "the historical consumer returned another unit's plan")
    validation = support.adjacent("post_comparison_validation")
    baseline = None
    if phase in {"correctness", "timing"}:
        validation.correctness(result)
    if phase == "timing":
        baseline = validation.validate_combination(plan["accepted_baseline_specification"], validation.Evidence())
        require(baseline["configuration"] == plan["configuration"], "actual accepted timing changed the profile configuration")
    require(not NUMERICAL.intersection(sys.modules), "historical metadata validation imported a numerical package")
    return {"schema_version": 1, "kind": "exact_historical_direct_profile_prerequisite_validation", "passed": True,
        "phase": phase, "profile_plan_sha256": digest(encoded(plan)), "historical_plan": reference(unit / "measurement_plan.json"),
        "result": result_ref, "helpers": old["helpers"], "resolved_binding": reference(unit / "direct_device_binding.json"),
        "execution_identity": result["execution_identity"], "accepted_combination": baseline,
        "compute": context, "stdlib_bootstrap": bootstrap, "parent_pid": os.getppid(), "validation_pid": os.getpid(),
        "validation_process": own_metadata_process(), "started_at_seconds": started, "finished_at_seconds": time.time(),
        "numerical_imports": [], "gpu_operations": 0, "scientific_credit": 0}


def publish(path, raw):
    """Publish a new immutable validation artifact without replacing an attempted phase."""
    require(path.parent.resolve(strict=True) == path.parent and not path.exists() and not path.is_symlink(),
        "preserve the previous profile prerequisite attempt")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def before_compute(directory, plan):
    """Replay the real historical consumers in isolated children before any current numerical import."""
    started = time.time()
    value = contract(plan); historical_metadata(plan); lifecycle_metadata(plan)
    context = compute_context(plan)
    require(context == {"hostname": "serval02", "uid": 2618919, "allocation_kind": "authorized_direct_compute"},
        "only the actual direct profile supervisor may produce its pre-numerical prerequisite record")
    parent_process = own_metadata_process()
    require(not NUMERICAL.intersection(sys.modules), "profile prerequisite replay must precede numerical imports")
    directory = Path(directory)
    raw = (directory / "measurement_plan.json").read_bytes()
    require(json.loads(raw) == plan, "the current prepared profile plan changed")
    path = directory / PROOF_NAME
    require(not path.exists() and not path.is_symlink(), "preserve the previous exact consumer replay")
    publish(directory / "direct_profile_prerequisites_attempt.json", encoded({"plan_sha256": digest(raw),
        "inputs": value, "parent_pid": os.getpid(), "retry_permitted": False}))
    phases = ["correctness", "timing"] + (["capture_qualification"] if value["mode"] == "measurement" else [])
    rows = []
    for phase in phases:
        command = ["/usr/bin/python3", "-I", "-S", "-B", str(directory / "code" / NAME), "--historical-phase", phase]
        prefix = "direct_profile_" + phase + "_validation"
        try:
            returned = subprocess.run(command, input=encoded(plan), capture_output=True,
                timeout=value["metadata_timeout_seconds"], close_fds=True, check=False)
        except BaseException as error:
            stdout, stderr = getattr(error, "stdout", None) or b"", getattr(error, "stderr", None) or b""
            publish(directory / (prefix + "_stdout.raw"), stdout)
            publish(directory / (prefix + "_stderr.raw"), stderr)
            publish(directory / (prefix + "_return.json"), encoded({"command": command, "returncode": None,
                "error_type": type(error).__name__, "stdout_sha256": digest(stdout), "stderr_sha256": digest(stderr)}))
            raise
        publish(directory / (prefix + "_stdout.raw"), returned.stdout)
        publish(directory / (prefix + "_stderr.raw"), returned.stderr)
        publish(directory / (prefix + "_return.json"), encoded({"command": command, "returncode": returned.returncode,
            "stdout_sha256": digest(returned.stdout), "stderr_sha256": digest(returned.stderr)}))
        require(returned.returncode == 0 and returned.stderr == b"" and len(returned.stdout) <= 1024 * 1024,
            "the exact historical prerequisite child failed or exceeded its metadata bound")
        proof = json.loads(returned.stdout)
        validate_child_proof(plan, phase, proof, parent_process["pid"], started, time.time())
        rows.append({"phase": phase, "proof": proof, "stdout": {"path": str(directory / (prefix + "_stdout.raw")), "sha256": digest(returned.stdout)},
            "return_record": reference(directory / (prefix + "_return.json"))})
    proof = {"schema_version": 1, "kind": "direct_profile_exact_historical_prerequisites", "passed": True,
        "plan_sha256": digest(raw), "inputs": value, "phases": rows, "parent_pid": os.getpid(),
        "parent_process": parent_process, "compute": context, "started_at_seconds": started,
        "finished_at_seconds": time.time(), "numerical_imports": [], "gpu_operations": 0, "scientific_credit": 0}
    publish(path, encoded(proof))
    return proof


def require_saved_validation(plan):
    """Require the actual pre-numerical replay and recheck every exact child byte sequence."""
    value = contract(plan); historical_metadata(plan); lifecycle_metadata(plan)
    directory = Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
    proof = read(directory / PROOF_NAME)[0]
    phases = ["correctness", "timing"] + (["capture_qualification"] if value["mode"] == "measurement" else [])
    supervisor = read(directory / "supervisor/direct_process_metadata.json")[0]
    process = supervisor["after"]["identity"]
    parent = proof.get("parent_process", {})
    require(type(proof.get("schema_version")) is int and proof["schema_version"] == 1
        and proof.get("kind") == "direct_profile_exact_historical_prerequisites" and proof.get("passed") is True
        and proof.get("plan_sha256") == digest((directory / "measurement_plan.json").read_bytes())
        and proof.get("inputs") == value and [row["phase"] for row in proof.get("phases", [])] == phases
        and set(parent) == {"hostname", "pid", "process_start_time", "uid"}
        and parent["hostname"] == "serval02" and type(parent["pid"]) is int and parent["pid"] > 0
        and type(parent["uid"]) is int and parent["uid"] == 2618919
        and type(parent["process_start_time"]) is str and re.fullmatch(r"[1-9][0-9]*", parent["process_start_time"])
        and type(proof.get("parent_pid")) is int and proof["parent_pid"] == parent["pid"]
        and all(encoded(parent[key]) == encoded(process[key]) for key in parent)
        and supervisor["before"]["identity"] == process
        and encoded(proof.get("compute")) == encoded({"hostname": "serval02", "uid": 2618919, "allocation_kind": "authorized_direct_compute"})
        and proof.get("numerical_imports") == [] and type(proof.get("gpu_operations")) is int and proof["gpu_operations"] == 0
        and type(proof.get("scientific_credit")) is int and proof["scientific_credit"] == 0,
        "the current profile lacks its exact successful pre-numerical historical replay")
    seen_processes = set()
    previous_finish = proof.get("started_at_seconds")
    for row in proof["phases"]:
        phase = row["phase"]; prefix = "direct_profile_" + phase + "_validation"
        returned = checked(row["return_record"]); stdout = (directory / (prefix + "_stdout.raw")).read_bytes()
        stderr = (directory / (prefix + "_stderr.raw")).read_bytes()
        command = ["/usr/bin/python3", "-I", "-S", "-B", str(directory / "code" / NAME), "--historical-phase", phase]
        require(returned["command"] == command and type(returned["returncode"]) is int and returned["returncode"] == 0 and stderr == b""
            and returned["stdout_sha256"] == digest(stdout) and returned["stderr_sha256"] == digest(stderr)
            and row["stdout"] == {"path": str(directory / (prefix + "_stdout.raw")), "sha256": digest(stdout)}
            and json.loads(stdout) == row["proof"] and row["proof"]["profile_plan_sha256"] == digest(encoded(plan)),
            "an actual historical child return or complete raw output changed")
        child = validate_child_proof(plan, phase, row["proof"], parent["pid"], proof.get("started_at_seconds"), proof.get("finished_at_seconds"))
        require(previous_finish <= child["started_at_seconds"], "historical replay children did not run sequentially")
        previous_finish = child["finished_at_seconds"]
        child_process = child["validation_process"]
        key = (child_process["hostname"], child_process["pid"], child_process["process_start_time"])
        require(key not in seen_processes, "separate historical replay phases reused a child process")
        seen_processes.add(key)
    return proof


def after_capture(directory, plan, result):
    """Complete the current mapped package record outside the unchanged numerical capture core."""
    contract(plan); proof = require_saved_validation(plan)
    path = Path(directory)
    direct = load_exact(path / "code/direct_measurement.py", "direct_measurement")
    completed = direct.capture_backend_completion(path, plan, result)
    marker = {"schema_version": 1, "kind": "direct_profile_actual_package_capture_completion",
        "capture_result": reference(path / "capture_result.json"), "backend_capture": reference(path / "direct_backend_capture.json"),
        "prerequisite_validation": reference(path / PROOF_NAME), "profile_protocol": plan["direct_profile_prerequisites"],
        "execution_identity": result["execution_identity"], "scientific_credit": 0}
    require(completed["numerical_execution"] == result["execution_identity"]
        and proof["finished_at_seconds"] <= result["execution_identity"]["started_at_seconds"],
        "actual profile execution did not follow its completed metadata prerequisite replay")
    publish(path / "direct_profile_backend_completion.json", encoded(marker))
    return marker


def result_binding(directory, plan):
    """Bind the final profile result to its original complete capture and package evidence."""
    value = contract(plan); proof = require_saved_validation(plan)
    directory = Path(directory); marker = read(directory / "direct_profile_backend_completion.json")[0]
    capture = read(directory / "capture_result.json")[0]
    backend = read(directory / "direct_backend_capture.json")[0]
    execution = capture.get("execution_identity", {})
    require(set(marker) == {"schema_version", "kind", "capture_result", "backend_capture", "prerequisite_validation",
            "profile_protocol", "execution_identity", "scientific_credit"}
        and type(marker["schema_version"]) is int and marker["schema_version"] == 1
        and marker["kind"] == "direct_profile_actual_package_capture_completion"
        and type(marker["scientific_credit"]) is int and marker["scientific_credit"] == 0
        and marker["capture_result"] == reference(directory / "capture_result.json")
        and marker["backend_capture"] == reference(directory / "direct_backend_capture.json")
        and marker["prerequisite_validation"] == reference(directory / PROOF_NAME)
        and marker["profile_protocol"] == value
        and encoded(marker["execution_identity"]) == encoded(execution) == encoded(backend.get("numerical_execution")),
        "profile finalization lost its original actual capture, execution or package proof")
    require(type(execution) is dict and set(execution) == {"hostname", "pid", "process_start_time",
            "started_at_seconds", "finished_at_seconds"}
        and execution["hostname"] == "serval02" and type(execution["pid"]) is int and execution["pid"] > 0
        and type(execution["process_start_time"]) is str and re.fullmatch(r"[1-9][0-9]*", execution["process_start_time"])
        and all(type(item) in (int, float) and math.isfinite(item) and item > 0 for item in (
            proof.get("started_at_seconds"), proof.get("finished_at_seconds"), execution["started_at_seconds"], execution["finished_at_seconds"]))
        and proof["started_at_seconds"] <= proof["finished_at_seconds"] <= execution["started_at_seconds"] < execution["finished_at_seconds"],
        "saved historical prerequisite validation did not finish before the actual bound numerical capture")
    return {"direct_profile_prerequisites": value,
        "direct_profile_prerequisite_validation": reference(directory / PROOF_NAME),
        "direct_profile_backend_completion": reference(directory / "direct_profile_backend_completion.json"),
        "profile_qualification_only": value["mode"] == "qualification",
        **({"measurement_valid": False, **{name: 0 for name in ZERO_FIELDS}} if value["mode"] == "qualification" else {})}


def main():
    """Expose only the fresh metadata child, with no job, device or numerical operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-phase", choices=("correctness", "timing", "capture_qualification"), required=True)
    args = parser.parse_args()
    raw = sys.stdin.buffer.read(1024 * 1024 + 1)
    require(len(raw) <= 1024 * 1024, "profile plan exceeds its metadata input bound")
    plan = json.loads(raw)
    print(json.dumps(historical_child(plan, args.historical_phase), allow_nan=False))


if __name__ == "__main__":
    main()
