"""Keep full-original profile target qualification separate from ordinary profiling admission.

The metadata controller owns no GPU descriptor. Qualification capture runs the
unchanged profile core, then all unchanged original learner/renderer/reference
gates in that same bound capture process. Its completion is always zero credit.
"""
import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import subprocess
import sys
import time

NAME = "profile_target_qualification.py"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
KIND = "full_original_profile_source_target_qualification"
CONTROLLER_KIND = "profile_target_qualification_controller"
QUALIFICATION_BUDGET = {"schema_version": 1, "work_seconds": 10200, "cleanup_seconds": 600, "total_seconds": 10800}
PROTOCOL = "full-original-profile-source-target-qualification-version-1"
ADMISSION_KIND = "reviewed_profile_source_target_admission"
REVIEW_KIND = "independent_full_original_profile_target_qualification_review"
SOURCE = "d28e6735e79788f618acb0ebfde8609db1f776af"
MAXIMUM = 16 * 1024 * 1024
ZERO = {"measurement_valid": False, "scientific_credit": 0, "profile_credit": 0,
    "case_credit": 0, "idea_credit": 0, "packing_point_credit": 0,
    "throughput_credit": 0, "final_training_credit": 0, "admission_granted": False,
    "worker_admitted": False}


def require(value, message):
    """Refuse a missing or changed qualification fact without weakening ordinary readers."""
    if not value:
        raise ValueError(message)


def encoded(value):
    """Serialize exact finite JSON types for immutable source/configuration comparisons."""
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def same(left, right):
    """Keep Boolean seeds and counts distinct from actual integers."""
    return encoded(left) == encoded(right)


def read(path, maximum=MAXIMUM):
    """Read an owned bounded regular file through one unchanged exact descriptor."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(),
        "qualification input must be its exact absolute ordinary path")
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid()
        and 0 <= before.st_size <= maximum, "qualification metadata exceeds its owned regular-file bound")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_uid", "st_gid", "st_nlink")
        require(all(getattr(os.fstat(descriptor), name) == getattr(before, name) for name in fields),
            "qualification input changed before its descriptor opened")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(maximum + 1)
        require(len(raw) == before.st_size and all(getattr(before, name) == getattr(os.fstat(descriptor), name)
            == getattr(path.lstat(), name) for name in fields), "qualification input changed during capture")
        return raw
    finally:
        os.close(descriptor)


def document(path):
    """Reject duplicate keys and nonfinite numbers in qualification metadata."""
    def unique(pairs):
        """Require one actual value for every JSON field."""
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate qualification metadata field")
            value[key] = item
        return value
    return json.loads(read(path), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite qualification metadata")))


def reference(path):
    """Derive a hash only after bounded owned stable capture."""
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(read(path)).hexdigest()}


def checked(value):
    """Require the exact declared reference before interpreting its metadata."""
    require(type(value) is dict and set(value) == {"path", "sha256"}
        and type(value["path"]) is str and type(value["sha256"]) is str
        and re.fullmatch("[0-9a-f]{64}", value["sha256"])
        and reference(value["path"]) == value, "qualification reference changed")
    return document(value["path"])


def publish(path, value):
    """Fsync exclusive actual bytes and their directory without replacing an attempt."""
    path = Path(path)
    require(path.parent.resolve(strict=True) == path.parent, "qualification output parent changed")
    raw = value if type(value) is bytes else encoded(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return reference(path)


def adjacent(plan, name):
    """Load the exact path/hash namespace shared by all new process-role callers."""
    path = Path(__file__).with_name(name + ".py")
    raw = read(path)
    expected = plan["helpers"]["files"].get(PREFIX + path.name)
    require(hashlib.sha256(raw).hexdigest() == expected, "qualification helper source changed: " + name)
    key = "profile_target_" + hashlib.sha256((str(path) + expected).encode()).hexdigest()
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        exec(compile(raw, str(path), "exec"), module.__dict__)
        module._profile_target_executed_sha256 = expected
        sys.modules[key] = module
    module = sys.modules[key]
    require(Path(module.__file__) == path and module._profile_target_executed_sha256 == expected,
        "an ambient helper cannot replace the exact qualification namespace")
    return module


def selected(plan):
    """Recognize only the distinct zero-credit capture kind and explicit declaration."""
    return plan.get("kind") == KIND and "profile_target_qualification" in plan


def helpers():
    """Declare the small metadata/backend/controller closure without numerical imports."""
    return (NAME, "profile_target_process.py", "profile_target_direct_process.py", "profile_target_release.py", "profile_qualification_backend.py", "direct_backend_image.py",
        "direct_common_timing_controller.py", "configured_profile_prerequisites.py")


def target_fields(plan):
    """Freeze exact scientific/helper/configuration identities independently of a unit name."""
    # Before: the complete prepared plan; after: only the fields a later ordinary profile must preserve.
    names = ("source_root", "source_commit", "core", "paired_source", "helpers", "configuration",
        "runtime_environment", "asset_record", "hardware_class", "copy_seeds", "python", "profile_updates", "profile_tool")
    return {name: plan[name] for name in names}


def contract(plan):
    """Require the immutable actual parent, real hardware policy and exact full-size learner."""
    require(selected(plan) and plan.get("source_commit") == SOURCE
        and plan.get("hardware_class") in {"h100", "jaguar03_rtx_a4500"},
        "only the distinct full-original profile target qualifier may use this path")
    value = plan["profile_target_qualification"]
    require(type(value) is dict and set(value) == {"schema_version", "protocol", "parent"}
        and type(value["schema_version"]) is int and value["schema_version"] == 1
        and value["protocol"] == PROTOCOL and "profile_admission_inputs" not in plan,
        "qualification cannot borrow ordinary or unreviewed admission")
    parent = checked(value["parent"])
    unit = Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
    require(parent.get("kind") == CONTROLLER_KIND and parent.get("protocol") == PROTOCOL
        and parent.get("profile_unit") == str(unit)
        and parent.get("campaign_run_dir") == plan["campaign_run_dir"]
        and parent.get("source_commit") == SOURCE and type(parent.get("scientific_credit")) is int
        and parent["scientific_credit"] == 0 and parent.get("admission_granted") is False
        and same(parent.get("qualification_time_budget"), QUALIFICATION_BUDGET)
        and type(parent.get("controller_maximum_seconds")) is int
        and parent["controller_maximum_seconds"] == QUALIFICATION_BUDGET["total_seconds"],
        "qualification child differs from its own immutable controller plan")
    require((plan["hardware_class"] == "h100" and "direct_binding" in plan and "configured_isolation" not in plan)
        or (plan["hardware_class"] == "jaguar03_rtx_a4500" and "configured_isolation" in plan and "direct_binding" not in plan),
        "the qualifier must retain its real direct or scheduler policy")
    require(same(parent["target_fields"], target_fields(plan)), "qualification target changed after immutable parent staging")
    learner = plan["configuration"]["learner"]
    require(type(learner["n_copies"]) is int and learner["n_copies"] in (1, 2)
        and same(plan["copy_seeds"], list(range(1, learner["n_copies"] + 1)))
        and type(plan["profile_updates"]) is int and plan["profile_updates"] >= 1
        and all(type(learner[key]) is int and learner[key] == expected for key, expected in (
            ("n_envs", 128), ("num_steps", 128), ("update_epochs", 4), ("num_minibatches", 4),
            ("observation_warmup_iterations", 50), ("total_timesteps", 2000000000))),
        "qualification must retain every full original learner count and independent seed")
    require(plan["runtime_environment"] == {"JAX_DEFAULT_MATMUL_PRECISION": "highest", "JAX_ENABLE_X64": "true",
        "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"}, "qualification precision/determinism changed")
    require(all(key not in plan for key in ("final_protocol", "final_confirmation_protocol", "completed_study",
        "device_memory_limit_protocol", "scaling_protocol", "configured_group", "direct_backend_image", "group_backend_image")),
        "profile qualification cannot grant final, memory, group or scaling admission")
    return parent


def original_gate_plan(plan):
    """Reuse original gate metadata with one fresh reference copy, never reinterpret profile counters."""
    parent = contract(plan)
    original = checked(parent["original_qualification_plan"])
    require(original["source_commit"] == plan["source_commit"] and same(original["core"], plan["core"])
        and same(original["paired_source"], plan["paired_source"])
        and same(original["runtime_environment"], plan["runtime_environment"])
        and same(original["asset_record"], plan["asset_record"]),
        "the original full gate source/runtime/assets differ from the current profile")
    # Before: a two-copy profile; after: a separate one-copy original gate projection with its own explicit count.
    projected = copy.deepcopy(plan)
    projected.update(kind="selected_configuration_measurement", mode="correctness",
        configuration=original["configuration"], reference_configuration=original["reference_configuration"], copy_seeds=[1])
    policy = "direct_binding" if "direct_binding" in plan else "configured_isolation"
    projected[policy]["consumer"] = "selected_measurement"
    for name in ("profile_target_qualification", "profile_target_process_control", "profile_admission_inputs"):
        projected.pop(name, None)
    for name in ("qualification_sources", "cleanrl_reference", "qualification_software", "renderer_contract"):
        projected[name] = original[name]
    require(type(projected["configuration"]["learner"]["n_copies"]) is int
        and projected["configuration"]["learner"]["n_copies"] == 1
        and same(projected["configuration"], projected["reference_configuration"]),
        "unchanged original gate runner requires one original reference copy")
    return projected


def controlled_work(plan, captured, original, started, finished):
    """Count actual profile and original-reference work separately, leaving fixture work unknown."""
    rows = captured["compilation_warmup"]
    require(type(rows) is list and 2 <= len(rows) <= 6 and all(type(row["updates"]) is int
        and row["updates"] == plan["profile_updates"] for row in rows), "actual compile warmup history is incomplete")
    compile_updates = sum(row["updates"] for row in rows)
    require(type(captured["normalization_warmup_iterations"]) is int and captured["normalization_warmup_iterations"] == 50
        and type(captured["profile_end_iteration"]) is int
        and type(captured["profile_start_iteration"]) is int and captured["profile_start_iteration"] == compile_updates
        and captured["profile_end_iteration"] - captured["profile_start_iteration"] == plan["profile_updates"]
        and captured["compiled_signatures_before"] == captured["compiled_signatures_after"],
        "qualification profile update origins or compiled signatures changed")
    identity = captured["execution_identity"]
    require(identity["pid"] == os.getpid() and 0 <= identity["finished_at_seconds"] <= started <= finished,
        "original gates must follow the actual same-process profile capture")
    return {"kind": "full_original_profile_source_target_qualification_work", "all_original_gates_passed": True,
        "configuration": plan["configuration"], "copy_seeds": plan["copy_seeds"],
        "actual_runtime_environment": plan["runtime_environment"],
        "original_reference": {"copies": 1, "normalization_warmup_iterations": 50,
            "reference_iteration": original["reference_iteration"], "reference_optimizer_step": original["reference_optimizer_step"],
            "executed_work": {name: original["executed_work"][old] for name, old in (
                ("normalization_warmup_transitions", "full_reference_warmup_transitions"),
                ("training_transitions", "full_reference_training_transitions"),
                ("complete_updates", "full_reference_updates"),
                ("minibatch_updates", "full_reference_optimizer_minibatch_updates"))}},
        "original_gate_raw_executed_work": original["executed_work"],
        "profile_capture": {"copies": len(plan["copy_seeds"]), "capture_completed": True,
            "profile_updates": plan["profile_updates"], "profile_start_iteration": captured["profile_start_iteration"],
            "profile_end_iteration": captured["profile_end_iteration"], "normalization_warmup_iterations": 50,
            "compiled_signatures_before": captured["compiled_signatures_before"],
            "compiled_signatures_after": captured["compiled_signatures_after"], "compile_warmup_updates": compile_updates},
        "execution_identity": identity,
        "original_gate_execution": {"hostname": identity["hostname"], "pid": identity["pid"],
            "process_start_time": identity["process_start_time"], "started_at_seconds": started, "finished_at_seconds": finished},
        "test_fixture_work": "unknown", "whole_job_transition_total_known": False, **ZERO}


def finish_capture(directory, plan):
    """Run the unchanged original gates inside the already bound actual capture child."""
    if not selected(plan):
        return None
    directory = Path(directory)
    parent = contract(plan)
    captured = document(directory / "capture_result.json")
    require(captured["kind"] == KIND and captured["capture_completed"] is True,
        "a partial or ordinary capture cannot qualify a profile target")
    # The unchanged core has already loaded only the frozen numerical source in this exact child.
    import jax
    import paired_benchmark as paired
    from exploration_platform.atari_pixels.config import PixelPPOConfig
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment
    from exploration_platform.atari_pixels.training import PixelPPORND
    support = adjacent(plan, "direct_measurement" if "direct_binding" in plan else "configured_measurement")
    original = original_gate_plan(plan)
    support.verify_qualification_backend(original, document(directory / "installed_backend.json"))
    target = directory / "full_original_gates"
    target.mkdir(exist_ok=False)
    binding = captured["resolved_device_binding"]
    started = time.time()
    with support.observed_work(jax.devices()[0], binding, directory / "qualification_memory") as observed:
        original_result = support.run_qualification(target, original, paired, jax, PixelPPOConfig, PixelPPORND, AtariPixelEnvironment)
    finished = time.time()
    require(original_result["all_original_gates_passed"] is True and type(original_result["reference_iteration"]) is int
        and original_result["reference_iteration"] == 1 and type(original_result["reference_optimizer_step"]) is int
        and original_result["reference_optimizer_step"] == 16, "full original gate/reference work did not complete")
    publish(target / "original_qualification_result.json", {**original_result, "qualification_memory": observed,
        "original_gate_projection": original, **ZERO})
    work = controlled_work(plan, captured, original_result, started, finished)
    publish(directory / "profile_target_qualification_work.json", work)
    backend = adjacent(plan, "profile_qualification_backend")
    backend.complete(directory, plan, captured["loaded_gpu_libraries"], work)
    return work


def authorize(plan):
    """Permit only a distinct qualified attempt or an independently reviewed exact ordinary admission."""
    if selected(plan):
        contract(plan)
        return {"qualification_only": True, "ordinary_profile_admitted": False, **ZERO}
    require("profile_target_qualification" not in plan and plan.get("kind") == "selected_configuration_profile",
        "qualification markers cannot enter ordinary profiling")
    inputs = plan.get("profile_admission_inputs")
    require(type(inputs) is dict and set(inputs) == {"record", "review"},
        "ordinary profiles remain closed without a separate actual reviewed source/target admission")
    record, review = checked(inputs["record"]), checked(inputs["review"])
    require(record.get("kind") == ADMISSION_KIND and review.get("kind") == "independent_profile_source_target_admission_review"
        and review.get("record") == inputs["record"] and review.get("approved") is review.get("passed") is True
        and review.get("independent_of_producer") is True and review.get("unresolved_anomalies") == [],
        "the exact ordinary admission lacks its genuine independent review")
    require(same(record.get("target_fields"), target_fields(plan))
        and record.get("source_commit") == SOURCE and record.get("profile_admission_granted") is True,
        "an old source/configuration/seed/copy target cannot admit this profile")
    qualification_review = checked(record["qualification_review"])
    require(qualification_review.get("kind") == REVIEW_KIND and qualification_review.get("approved") is True
        and qualification_review.get("passed") is True and qualification_review.get("independent_of_producer") is True
        and qualification_review.get("qualification_result") == record["qualification_result"]
        and qualification_review.get("all_original_gates_and_whole_tool_target_verified") is True
        and qualification_review.get("unresolved_anomalies") == [], "target qualification needs independent raw acceptance")
    completed = checked(record["qualification_result"])
    require(completed.get("kind") == KIND and completed.get("completed") is True
        and completed.get("measurement_valid") is False and completed.get("scientific_credit") == 0
        and same(completed.get("qualified_target_fields"), target_fields(plan)),
        "qualification result cannot supply a partial or different profile source")
    require(record.get("four_context_proof") == completed.get("four_context_proof")
        and record.get("four_context_review") == completed.get("four_context_review"),
        "ordinary admission changed the actual complete common baseline proof")
    require_four_contexts(plan, record["four_context_proof"], record["four_context_review"])
    return record


def require_four_contexts(plan, proof_ref, review_ref):
    """Reuse exact historical namespaces for every actual common context; no raw history rerun."""
    reader = adjacent(plan, "configured_profile_prerequisites")
    files = reader.MetadataFiles()
    proof, review = files.document(proof_ref), files.document(review_ref)
    require(proof.get("kind") == reader.PROTOCOL["kind"] and proof.get("source_commit") == SOURCE
        and proof.get("campaign_run_dir") == plan["campaign_run_dir"] and proof.get("operational_admission") is False
        and review.get("kind") == "independent_reviewed_four_context_configured_profile_prerequisites"
        and review.get("proof") == proof_ref and review.get("approved") is review.get("passed") is True
        and review.get("independent_of_producer") is True and review.get("all_four_raw_reviews_verified") is True
        and review.get("exact_historical_consumer_namespaces_verified") is True
        and review.get("unresolved_anomalies") == [] and review.get("admission_granted") is False,
        "all four genuine common results and their independent proof review remain required")
    entries = proof.get("contexts")
    require(type(entries) is list and len(entries) == 4 and all(type(row.get("copies")) is int for row in entries)
        and {(row["hardware_class"], row["copies"]) for row in entries} == reader.CONTEXTS,
        "missing H100 two-copy evidence cannot be replaced by another context")
    contexts = [reader.reviewed_context(files, row, plan) for row in entries]
    historical = next(row["historical_plan"] for row in contexts if row["hardware_class"] == "h100" and row["copies"] == 1)
    reader.initial_snapshot(files, proof["initial_snapshot_inputs"], historical)
    selected_context = next(row for row in contexts if row["hardware_class"] == plan["hardware_class"]
        and row["copies"] == len(plan["copy_seeds"]))
    require(selected_context["result"] == plan["inputs"]["accepted_confirmation"],
        "the profile must retain its exact independently accepted common context")
    return {"proof": proof_ref, "review": review_ref, "contexts": contexts}


def complete_result(directory, plan, result):
    """Publish qualification only outside ordinary result.json and never grant scientific credit."""
    if not selected(plan):
        return False
    parent = contract(plan)
    require(result.get("completed") is True and result.get("capture_completed") is True,
        "an incomplete profile cannot publish qualification completion")
    work_ref = reference(Path(directory) / "profile_target_qualification_work.json")
    work = checked(work_ref)
    require(work.get("all_original_gates_passed") is True and same(work["configuration"], plan["configuration"])
        and same(work["copy_seeds"], plan["copy_seeds"]), "profile/original work identities differ")
    backend = adjacent(plan, "profile_qualification_backend")
    backend.validate_saved(directory, plan, result["resolved_device_binding"], result["loaded_gpu_libraries"], work)
    require_four_contexts(plan, parent["four_context_proof"], parent["four_context_review"])
    completed = {**result, **ZERO, "kind": KIND, "completed": True,
        "qualification_plan": reference(Path(directory) / "measurement_plan.json"),
        "installed_backend": document(Path(directory) / "installed_backend.json"),
        "qualified_target_fields": target_fields(plan), "qualification_work": work_ref,
        "profile_qualification_backend": plan["profile_qualification_backend"],
        "qualification_software": plan["qualification_software"],
        "qualification_parent": plan["profile_target_qualification"]["parent"],
        "four_context_proof": parent["four_context_proof"], "four_context_review": parent["four_context_review"]}
    publish(Path(directory) / "profile_target_qualification_result.json", completed)
    return True


def verify_installed_before_tools(plan, installed):
    """Keep complete current package/source/library receipts tied to original software before tools."""
    if selected(plan):
        original = original_gate_plan(plan)
        support = adjacent(plan, "direct_measurement" if "direct_binding" in plan else "configured_measurement")
        support.verify_qualification_backend(original, installed)
        adjacent(plan, "profile_qualification_backend").validate_prepared(plan)
        return {"original_full_software_matched": True, **ZERO}
    record = authorize(plan)
    completed = checked(record["qualification_result"])
    require(same(installed, completed["installed_backend"]),
        "current profile packages, source or GPU library bytes changed after target qualification")
    expected = checked(completed["profile_qualification_backend"]["expected"])
    backend = adjacent(plan, "profile_qualification_backend")
    backend.validate_expected(expected, plan["python"])
    files = adjacent(plan, "direct_backend_image")
    actual = files.describe_image(expected["site_packages"], expected["image"]["path"])
    require(all(same(actual[name], expected[name]) for name in ("site_packages", "image", "distribution")),
        "actual current owning package/METADATA/RECORD/image bytes differ from the qualified target")
    executable = files.stream_image(Path("/proc/self/exe").resolve(strict=True))
    require(str(Path(sys.executable).absolute()) == plan["python"]
        and all(executable[name] == expected["python_image"][name] for name in ("path", "sha256", "size_bytes")),
        "actual running profile Python bytes changed after qualification")
    return {"qualification_result": record["qualification_result"], "current_profile_software_matched": True,
        "ordinary_admission_review": plan["profile_admission_inputs"]["review"]}


def verify_bound_backend_begin(directory, plan, binding):
    """Reuse the new same-process BEGIN written at actual binding, never start another capture."""
    contract(plan)
    backend = adjacent(plan, "profile_qualification_backend")
    backend.expected_for_plan(plan)
    directory = Path(directory)
    value = document(directory / backend.BEGIN)
    process = backend.process_join(directory, plan, binding, current=True)
    require(type(value.get("schema_version")) is int and value["schema_version"] == 2
        and value.get("kind") == "profile_qualification_backend_begin" and value.get("capability") == backend.CAPABILITY
        and value.get("plan") == reference(directory / "measurement_plan.json")
        and value.get("expected") == plan["profile_qualification_backend"]["expected"]
        and value.get("binding") == reference(directory / backend.binding_file(plan))
        and same(value.get("process"), process)
        and all(same(value.get(key), binding[key]) for key in ("gpu_uuid", "pci_address", "physical_minor"))
        and not (directory / backend.FAILURE).exists(),
        "the post-binding hook requires its already captured exact new backend BEGIN")
    return value


def verify_completed_backend(directory, plan, captured):
    """Require the new backend FINAL and both complete phases without entering an old profile reader."""
    contract(plan)
    directory = Path(directory)
    require(same(captured, document(directory / "capture_result.json")) and captured.get("capture_completed") is True,
        "post-work validation cannot substitute or complete a partial numerical capture")
    work = document(directory / "profile_target_qualification_work.json")
    backend = adjacent(plan, "profile_qualification_backend")
    backend.validate_saved(directory, plan, captured["resolved_device_binding"], captured["loaded_gpu_libraries"], work)
    return {"profile_target_backend_completion": reference(directory / backend.FINAL),
        "profile_target_backend_begin": reference(directory / backend.BEGIN),
        "profile_target_qualification_work": reference(directory / "profile_target_qualification_work.json")}


def on_binding(directory, plan, binding):
    """Start qualification package capture or recheck an ordinary admitted physical target before training."""
    require(reference(Path(__file__))["sha256"] == plan["helpers"]["files"].get(PREFIX + NAME),
        "the actual post-binding qualification/admission source changed")
    if selected(plan):
        contract(plan)
        return adjacent(plan, "profile_qualification_backend").begin(directory, plan, binding)
    record = authorize(plan)
    completed = checked(record["qualification_result"])
    original = completed["resolved_device_binding"]
    require(all(same(binding[name], original[name]) for name in ("gpu_uuid", "pci_address", "physical_minor")),
        "this actual physical GPU was not qualified for the exact admitted profile source")
    verify_installed_before_tools(plan, document(Path(directory) / "installed_backend.json"))
    return {"qualified_physical_target_matched": True, "qualification_result": record["qualification_result"]}


def validate_configured_capture_records(directory, plan, result, metadata):
    """Retain every original raw saved profile check while selecting only the new target proof."""
    authorize(plan)
    require('configured_isolation' in plan and 'direct_binding' not in plan and ('profile_target_process_control' in plan), 'the new saved allocation reader requires its exact new target policy')
    directory = Path(directory)
    intent = metadata.validate_saved_intent(directory, plan, metadata.read_json(directory / 'profile_process_intent.json'))
    require(intent['plan'] == metadata.receipt(directory / 'measurement_plan.json') and intent['source'] == metadata.receipt(directory / 'code/profile_selected.py') and (intent['source']['sha256'] == plan['helpers']['files']['10_jax_exploration_platform/scripts/atari_pixel_campaign/profile_selected.py']), 'saved profile process evidence changed its source or plan')
    (waiter_command, capture_command) = metadata.commands(directory, plan)
    require(intent['waiter_command'] == waiter_command and intent['capture_command'] == capture_command and (intent['nsight_command'] == metadata.nsight_command(directory, plan)) and (intent['nsight_command'] == metadata.read_json(directory / 'executed_command.json')), 'saved profile process commands changed')
    claim = metadata.read_json(directory / 'capture_identity.json')
    require(type(claim.get('schema_version')) is int and claim['schema_version'] == 1 and (claim.get('protocol') == metadata.PROTOCOL) and (claim['challenge'] == intent['challenge']) and (claim['intent'] == metadata.receipt(directory / 'profile_process_intent.json')) and (claim['source'] == intent['source']) and (claim['plan'] == intent['plan']), 'saved capture challenge or source changed')
    metadata.validate_snapshot(claim['process'], intent['parent']['uids'][0], intent['parent']['cgroup'])
    require(type(claim['pid']) is int and claim['pid'] == claim['process']['pid'] and (claim['process_start_time'] == claim['process']['process_start_time']) and (claim['process']['command'] == capture_command), 'saved actual capture claim differs from its raw process')
    observation = metadata.read_json(directory / 'capture_process_observation.json')
    rebuilt = metadata.validate_observation(observation, claim, capture_command, intent['parent'])
    require(metadata.stable(rebuilt['capture']) == metadata.stable(claim['process']), 'the saved blocked capture differs from the released observed process')
    owner = metadata.read_json(directory / 'profiler_process.json')
    require(owner['pid'] == observation['root_pid'] and owner['process_start_time'] == observation['root_start_time'] and (owner['command'] == intent['nsight_command']) and (owner['parent_pid'] == intent['parent']['pid']) and (owner['parent_start_time'] == intent['parent']['process_start_time']), 'saved Nsight launch differs from its actual Popen identity')
    metadata.validate_root_wait(directory, intent, owner)
    launch = metadata.read_json(directory / 'capture_actual_launch.json')
    waited = metadata.read_json(directory / 'capture_actual_wait.json')
    require(launch['challenge'] == waited['challenge'] == intent['challenge'] and launch['claim'] == metadata.receipt(directory / 'capture_identity.json') and (launch['child']['pid'] == claim['pid']) and (metadata.stable(launch['child']) == metadata.stable(claim['process'])) and (launch['command'] == waited['command'] == capture_command) and (waited['launch'] == metadata.receipt(directory / 'capture_actual_launch.json')) and (waited['waited_pid'] == claim['pid']) and (type(waited['returncode']) is int) and (waited['returncode'] == 0) and (waited['exited_normally'] is True) and (waited['interventions'] == []), 'the actual capture was not waited successfully by its owned parent')
    waiter = launch['waiter']
    metadata.validate_snapshot(waiter, claim['process']['uids'][0], claim['process']['cgroup'])
    require(waiter['command'] == waiter_command and claim['process']['parent_pid'] == waiter['pid'] and (str(waiter['pid']) in observation['nodes']) and (metadata.stable(observation['nodes'][str(waiter['pid'])]) == metadata.stable(waiter)), 'saved actual Python parent is not the observed Nsight descendant')
    release = metadata.read_json(directory / 'capture_release.json')
    require(release == {**owner, 'capture_identity': claim, 'process_observation': metadata.receipt(directory / 'capture_process_observation.json')}, 'saved numerical release differs from the actual observed source child')
    identity = result['execution_identity']
    require(identity['pid'] == claim['pid'] and identity['process_start_time'] == claim['process_start_time'] and (result.get('capture_process_protocol') == metadata.PROTOCOL) and (result.get('capture_process_observation') == metadata.receipt(directory / 'capture_process_observation.json')) and (result.get('capture_actual_wait') == metadata.receipt(directory / 'capture_actual_wait.json')), 'the captured numerical result belongs to another process or omitted its actual wait')
    require(result.get('profiler_actual_wait') == metadata.receipt(directory / 'profiler_actual_wait.json'), 'the numerical result omitted the actual original Nsight wait')
    reader = adjacent(plan, 'profile_target_process')
    proof = reader.validate_saved(directory, plan, metadata, intent, observation, launch, waited, reader.saved_deadline(directory, plan, metadata))
    require(same(result.get('configured_native_process_proof'), proof), 'the captured result omitted its exact new target process proof')
    return {'capture_identity': metadata.receipt(directory / 'capture_identity.json'), 'process_observation': metadata.receipt(directory / 'capture_process_observation.json'), 'capture_wait': metadata.receipt(directory / 'capture_actual_wait.json'), 'capture_pid': rebuilt['capture']['pid']}


def create_admission(qualification_result, qualification_review, output):
    """Write a separate scope-limited admission only after actual target and four-context review."""
    result_ref, review_ref = reference(qualification_result), reference(qualification_review)
    completed, review = checked(result_ref), checked(review_ref)
    require(completed.get("kind") == KIND and completed.get("completed") is True
        and completed.get("measurement_valid") is False and completed.get("scientific_credit") == 0
        and review.get("kind") == REVIEW_KIND and review.get("approved") is review.get("passed") is True
        and review.get("independent_of_producer") is True and review.get("qualification_result") == result_ref
        and review.get("all_original_gates_and_whole_tool_target_verified") is True
        and review.get("unresolved_anomalies") == [], "actual raw target qualification must precede recorded admission")
    qualified_plan = checked(completed["qualification_plan"])
    require_four_contexts(qualified_plan, completed["four_context_proof"], completed["four_context_review"])
    record = {"schema_version": 1, "kind": ADMISSION_KIND, "source_commit": SOURCE,
        "qualification_result": result_ref, "qualification_review": review_ref,
        "target_fields": completed["qualified_target_fields"], "four_context_proof": completed["four_context_proof"],
        "four_context_review": completed["four_context_review"], "profile_admission_granted": True,
        "worker_admission_granted": False, "scientific_credit": 0,
        "scope": "exact profile source/helper/configuration/copy seeds/tool/Python and qualified physical target only",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    return publish(output, record)


def configure_child(plan, parent_path, backend_path):
    """Bind one actual controller before writing a distinct qualification capture plan."""
    parent_ref = reference(parent_path)
    parent = checked(parent_ref)
    require(parent.get("kind") == CONTROLLER_KIND and parent.get("profile_unit") == str(
        Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]), "qualification parent names another capture")
    plan["kind"] = KIND
    plan["profile_target_qualification"] = {"schema_version": 1, "protocol": PROTOCOL, "parent": parent_ref}
    require(same(parent["target_fields"], target_fields(plan)), "qualification child target differs from its prior parent")
    adjacent(plan, "profile_qualification_backend").configure_plan(plan, backend_path)
    contract(plan)
    require_four_contexts(plan, parent["four_context_proof"], parent["four_context_review"])
    return plan


def wait_controller_release(directory, plan):
    """Require actual durable own parent/child identities before qualification numerical startup."""
    if not selected(plan):
        return None
    parent = contract(plan)
    directory = Path(directory)
    release_path = directory / "profile_target_controller_release.json"
    deadline = time.monotonic() + 30
    control = adjacent(plan, "direct_common_timing_controller")
    current = control.read_process(os.getpid(), os.getuid())
    while not os.path.lexists(release_path):
        require(time.monotonic() < deadline, "qualification controller did not release its recorded child")
        observed = control.read_process(current["parent_pid"], os.getuid())
        require(observed["pid"] == current["parent_pid"], "qualification parent changed before release")
        time.sleep(.05)
    release = document(release_path)
    require(release.get("kind") == "actual_profile_target_controller_child_release"
        and release.get("child_plan") == reference(directory / "measurement_plan.json")
        and release.get("parent_plan") == plan["profile_target_qualification"]["parent"]
        and release["child"]["pid"] == current["pid"]
        and release["child"]["process_start_time"] == current["process_start_time"]
        and release["child"]["parent_pid"] == release["controller"]["pid"] == current["parent_pid"],
        "qualification child lacks its exact genuinely returned parent/Popen identity")
    actual_parent = control.read_process(current["parent_pid"], os.getuid())
    require(actual_parent["process_start_time"] == release["controller"]["process_start_time"]
        and actual_parent["uid"] == current["uid"] == os.getuid(), "qualification metadata parent is no longer exact")
    return release


def prepare(request_path):
    """Stage a fresh metadata controller and its distinct full-size profile capture; launch nothing."""
    request_path = Path(request_path).resolve(strict=True)
    request = document(request_path)
    require(request.get("kind") == "profile_target_qualification_preparation_request"
        and request.get("protocol") == PROTOCOL and request.get("source_commit") == SOURCE
        and type(request.get("profile_prepare_arguments")) is list
        and all(type(value) is str for value in request["profile_prepare_arguments"]),
        "a reviewed concrete profile qualification preparation request is required")
    profile_path = Path(__file__).with_name("profile_selected.py")
    specification = importlib.util.spec_from_file_location("profile_target_metadata_preparation", profile_path)
    profile = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(profile)
    args = profile.parse_arguments(request["profile_prepare_arguments"])
    require(args.prepare_only and args.profile_target_qualification_parent is None
        and args.profile_admission_record is None and args.profile_admission_review is None,
        "qualification request cannot supply another parent or ordinary admission")
    run = args.run_dir.resolve(strict=True)
    unit = run / "data" / request["controller_unit_id"]
    child = run / "data" / args.unit_id
    require(Path(request["controller_unit_id"]).name == request["controller_unit_id"]
        and unit != child and not unit.exists() and not child.exists(), "prepare fresh separate controller/capture units")
    configuration = document(args.configuration)
    runtime = document(args.runtime_environment)
    assets = document(args.asset_record)
    source_root = args.source_root.resolve(strict=True)
    baseline = document(args.accepted_baseline_specification)
    native = args.configured_native_process_qualification is not None
    direct = args.direct_binding_profile is not None
    configured = args.configured_isolation_profile is not None
    require(configured != direct and (not configured or native), "qualification needs one real explicit hardware policy")
    helper = profile.helper_receipt(configured, direct, False, configured, direct_profile=direct,
        configured_native=native, configured_prerequisites=configured, profile_target=True)
    desired = {"source_root": str(source_root), "source_commit": args.source_commit,
        "core": profile.diagnostic.core_identity(source_root, args.source_commit),
        "paired_source": profile.training.source_receipt(source_root, args.source_commit,
            [PREFIX + "paired_benchmark.py", PREFIX + "ideas.py"]), "helpers": helper,
        "configuration": configuration, "runtime_environment": runtime, "asset_record": assets,
        "hardware_class": args.hardware_class, "copy_seeds": list(range(args.seed_start,
            args.seed_start + configuration["learner"]["n_copies"])), "python": str(args.python.absolute()),
        "profile_updates": profile.profile_updates(configuration, args.minimum_updates),
        "profile_tool": document(args.profile_tool_record)}
    proposal = {**desired, "campaign_run_dir": str(run), "unit_id": args.unit_id,
        "inputs": {"accepted_confirmation": {"path": baseline["confirmation_record"],
            "sha256": request["accepted_confirmation_sha256"]}}}
    # Missing fourth evidence fails before any actual unit directory is created.
    require_four_contexts(proposal, request["four_context_proof"], request["four_context_review"])
    original = checked(request["original_qualification_plan"])
    require(original["source_commit"] == SOURCE and original["configuration"]["learner"]["n_copies"] == 1,
        "the unchanged full-original gate plan must already exist")
    parent = {"schema_version": 1, "kind": CONTROLLER_KIND, "protocol": PROTOCOL,
        "campaign_run_dir": str(run), "unit_id": request["controller_unit_id"], "profile_unit": str(child),
        "source_commit": SOURCE, "target_fields": desired, "original_qualification_plan": request["original_qualification_plan"],
        "four_context_proof": request["four_context_proof"], "four_context_review": request["four_context_review"],
        "backend_image_record": request["backend_image_record"], "request": reference(request_path),
        "controller_maximum_seconds": request["controller_maximum_seconds"],
        "git": {"numerical_commit": SOURCE, "helper_commit": helper["source_commit"],
            "helper_root": helper["source_root"], "selected_helper_files_match_committed_bytes": True},
        "qualification_time_budget": dict(QUALIFICATION_BUDGET), **ZERO}
    require(type(parent["controller_maximum_seconds"]) is int and parent["controller_maximum_seconds"] == QUALIFICATION_BUDGET["total_seconds"],
        "qualification controller needs its reviewed finite wall limit")
    unit.mkdir()
    (unit / "code").mkdir()
    publish(unit / "measurement_plan.json", parent)
    for name in (NAME, "direct_common_timing_controller.py"):
        publish(unit / "code" / name, Path(__file__).with_name(name).read_bytes())
    for name, value in (("manifest.yaml", parent), ("config_resolved.yaml", configuration)):
        publish(unit / name, value)
    command = ["/usr/bin/python3", "-I", "-S", "-B", str(unit / "code" / NAME), "--prepared-unit", str(unit)]
    publish(unit / "command.txt", (shlex.join(command) + "\n").encode())
    publish(unit / "experiment_background.md", (
        "## Purpose\n\nQualify this exact full-size profile source and physical target with zero scientific credit.\n\n"
        "## Key hyperparameters\n\nEach profile copy retains 128 environments, 128 rollout steps, 4 epochs, 4 minibatches and 50 normalization warmups. "
        f"The profile has {configuration['learner']['n_copies']} independent copies and {desired['profile_updates']} complete captured updates. "
        "A separate unchanged original gate projection has one full reference update and one original copy.\n\n"
        "## Code and config changes\n\nThe unchanged capture core runs before the unchanged 48 learner/environment tests, renderer and full reference update in the same bound capture child. "
        "Actual backend begin/completion and closed-tool process evidence surround both phases. Fixture work remains unknown.\n\n"
        f"## Git state\n\nNumerical source: `{SOURCE}`. Helper source: `{helper['source_commit']}` at `{helper['source_root']}`. "
        "Every selected helper file matches its committed bytes; all input hashes are in the immutable controller and capture plans.\n").encode())
    args.profile_target_qualification_parent = unit / "measurement_plan.json"
    args.profile_backend_image_record = Path(request["backend_image_record"]["path"])
    result = profile.prepare(args)
    require(result == child, "metadata preparation returned another child")
    publish(unit / "prepared_child_binding.json", {"controller_plan": reference(unit / "measurement_plan.json"),
        "child_plan": reference(child / "measurement_plan.json"), "command": shlex.split((child / "command.txt").read_text()),
        "controller_source": reference(unit / "code" / NAME), "helpers": helper, **ZERO})
    return unit


def exact_child_startup(control, child, command, affinity, unit):
    """Wait only on zero raw command bytes while every own process-generation predicate stays exact."""
    started = time.monotonic()
    deadline = started + 30
    rows, birth, failure, ready = [], None, None, False
    expected = b"\0".join(value.encode() for value in command) + b"\0"
    current = None
    try:
        while True:
            require(time.monotonic() <= deadline, "qualification child exact-command startup deadline expired")
            current = control.read_process(child.pid, os.getuid())
            birth = current["process_start_time"] if birth is None else birth
            raw = bytes.fromhex(current["command_bytes_hex"])
            predicates = {"pid": current["pid"] == child.pid, "birth": current["process_start_time"] == birth,
                "parent": current["parent_pid"] == os.getpid(), "uid": current["uid"] == os.getuid(),
                "affinity": current["affinity"] == affinity,
                "raw_hash": hashlib.sha256(raw).hexdigest() == current["command_sha256"],
                "raw_decode": current["command"] == ([part.decode() for part in raw.rstrip(b"\0").split(b"\0")] if raw else [])}
            returned = child.poll()
            rows.append({"process": current, "predicates": predicates, "raw_command_empty": raw == b"",
                "expected_raw_command": raw == expected, "actual_popen_returncode": returned,
                "observed_monotonic": time.monotonic()})
            require(all(predicates.values()) and returned is None, "qualification child identity or live Popen status changed")
            if raw == expected:
                ready = True
                return current
            require(raw == b"", "unexpected nonempty qualification child command cannot wait or register")
            time.sleep(min(.05, max(0, deadline - time.monotonic())))
    except BaseException as error:
        failure = {"error_type": type(error).__name__, "error": str(error)}
        raise
    finally:
        publish(Path(unit) / "profile_target_child_startup_observations.json", {"kind": "actual_profile_target_empty_only_startup",
            "started_monotonic": started, "deadline_monotonic": deadline, "finished_monotonic": time.monotonic(),
            "maximum_seconds": 30, "actual_popen_pid": child.pid, "expected_command": command,
            "expected_parent_pid": os.getpid(), "observed_birth": birth, "selected_affinity": affinity,
            "observations": rows, "exact_command_ready": ready, "failure": failure,
            "controller_holds_standing_descriptor": False, "historical_failure_cause_inferred": False, **ZERO})


def retain_startup_custody(child):
    """Keep the caller's metadata barrier until this actual Popen child is positively waited, without signals."""
    previous = {number: signal.getsignal(number) for number in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)}
    for number in previous:
        signal.signal(number, signal.SIG_IGN)
    errors = []
    while True:
        try:
            returned = child.wait(timeout=.1)
            if type(returned) is int:
                break
            errors.append({"operation": "exact_wait", "error_type": "noninteger_return"})
        except subprocess.TimeoutExpired:
            pass
        except Exception as error:
            if len(errors) < 32:errors.append({"operation": "exact_wait", "error_type": type(error).__name__})
        try:
            time.sleep(.1)
        except Exception as error:
            if len(errors) < 32:errors.append({"operation": "wait_sleep", "error_type": type(error).__name__})
    for number, handler in previous.items():
        signal.signal(number, handler)
    return {"actual_popen_pid": child.pid, "returncode": returned, "actual_exact_wait": True,
        "signal_sent": False, "ordinary_observer_errors": errors}


def controller(directory):
    """Wait on one exact recorded producer without owning, opening or signaling its GPU descriptor."""
    directory = Path(directory).resolve(strict=True)
    parent = document(directory / "measurement_plan.json")
    require(parent.get("kind") == CONTROLLER_KIND and Path(__file__).resolve(strict=True) == directory / "code" / NAME,
        "execute only the actual staged qualification controller")
    binding = document(directory / "prepared_child_binding.json")
    require(binding["controller_plan"] == reference(directory / "measurement_plan.json")
        and binding["controller_source"] == reference(Path(__file__)), "controller prepared bytes changed")
    child_plan = checked(binding["child_plan"])
    child_unit = Path(parent["profile_unit"])
    require(child_plan["profile_target_qualification"]["parent"] == binding["controller_plan"]
        and binding["command"] == shlex.split((child_unit / "command.txt").read_text()), "exact prepared qualification child changed")
    contract(child_plan)
    review = document(directory / "controller_launch_independent_review.json")
    require(review.get("kind") == "independent_profile_target_qualification_controller_launch_review"
        and review.get("approved") is review.get("passed") is True and review.get("controller_plan") == binding["controller_plan"]
        and review.get("child_plan") == binding["child_plan"] and review.get("independent_of_producer") is True,
        "actual source/unit launch needs its genuine independent review")
    require(not any(os.path.lexists(directory / name) for name in ("controller_intent.json", "controller_spawn.json",
        "controller_wait.json", "controller_failure.json", "controller_inconclusive.json")), "never repeat an attempted controller")
    control = adjacent(child_plan, "direct_common_timing_controller")
    owner = control.read_process(os.getpid(), os.getuid())
    require(owner["command"] == shlex.split((directory / "command.txt").read_text())
        and not ({"jax", "jaxlib", "numpy", "torch", "flax", "optax", "jaxtari"} & sys.modules.keys()),
        "only the exact metadata controller may start its reviewed child")
    if child_plan["hardware_class"] == "h100":
        require(owner["hostname"] == "serval02" and os.environ.get("CUDA_VISIBLE_DEVICES") == "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
            and not any(key.startswith("SLURM_") for key in os.environ), "direct qualification controller target changed")
    else:
        job, step = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID")
        cgroup = Path("/proc/self/cgroup").read_text()
        require(type(job) is str and job.isdecimal() and type(step) is str and step.isdecimal()
            and "/job_" + job + "/step_" + step + "/" in cgroup
            and (Path(parent["campaign_run_dir"]) / "slurm/submitted_jobids_uva.txt").read_text().splitlines().count(job) == 1,
            "scheduler qualification controller is not in its exact owned numeric step")
    affinity = control.chosen_affinity(owner["affinity"], 8)
    publish(directory / "controller_intent.json", {"controller": owner, "plan": binding["controller_plan"],
        "child_plan": binding["child_plan"], "command": binding["command"], "standing_gpu_descriptor_held": False, **ZERO})
    child, observed = None, None
    try:
        with control.metadata_lock(Path(parent["campaign_run_dir"]) / "queue/metadata.lock", os.getuid()):
            try:
                previous = sorted(os.sched_getaffinity(0))
                try:
                    os.sched_setaffinity(0, affinity)
                    with (directory / "producer_stdout.raw").open("xb") as stdout, (directory / "producer_stderr.raw").open("xb") as stderr:
                        child = subprocess.Popen(binding["command"], stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                            close_fds=True, pass_fds=(), start_new_session=True)
                finally:
                    os.sched_setaffinity(0, previous)
                observed = exact_child_startup(control, child, binding["command"], affinity, child_unit)
                spawn = {"controller": owner, "child": observed, "actual_popen_pid": child.pid,
                    "child_plan": binding["child_plan"], "command": binding["command"], "standing_gpu_descriptor_held": False, **ZERO}
                publish(directory / "controller_spawn.json", spawn)
                if child_plan["hardware_class"] == "h100":
                    control.append_registration(child_unit, child_plan, observed, binding["command"], os.getuid())
                publish(child_unit / "profile_target_controller_release.json", {"kind": "actual_profile_target_controller_child_release",
                    "controller": owner, "child": observed, "child_plan": binding["child_plan"],
                    "parent_plan": binding["controller_plan"], "controller_spawn": reference(directory / "controller_spawn.json"), **ZERO})
            except BaseException:
                if child is not None:
                    custody = retain_startup_custody(child)
                    publish(directory / "controller_failed_startup_exact_wait.json", custody)
                raise

        try:
            returned = child.wait(timeout=parent["controller_maximum_seconds"])
        except subprocess.TimeoutExpired:
            publish(directory / "controller_inconclusive.json", {"actual_popen_pid": child.pid,
                "child": observed, "signal_sent": False, "producer_left_running": True, "completed": False,
                "reason": "the sole GPU descriptor owner has not proved completion; no successor is admitted", **ZERO})
            raise
        publish(directory / "controller_wait.json", {"waited_pid": child.pid, "returncode": returned,
            "exited_normally": returned == 0, "command": binding["command"], "child": observed,
            "interventions": [], "standing_gpu_descriptor_held": False, **ZERO})
        require(type(returned) is int and returned == 0, "qualification producer did not return normal zero")
        completed = document(child_unit / "profile_target_qualification_result.json")
        require(completed.get("completed") is True and completed.get("kind") == KIND,
            "normal process exit alone cannot qualify the source/target")
        publish(directory / "controller_outcome.json", {"passed": True,
            "qualification_result": reference(child_unit / "profile_target_qualification_result.json"),
            "wait": reference(directory / "controller_wait.json"), **ZERO})
        return completed
    except BaseException as error:
        publish(directory / "controller_failure.json", {"error_type": type(error).__name__, "error": str(error),
            "actual_popen_pid": None if child is None else child.pid, "actual_identity": observed,
            "actual_returncode": None if child is None else child.returncode, "signal_sent": False,
            "standing_gpu_descriptor_held": False, **ZERO})
        raise


def main(argv=None):
    """Separate metadata-only staging from one explicitly reviewed actual controller attempt."""
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--request", type=Path)
    operation.add_argument("--prepared-unit", type=Path)
    operation.add_argument("--write-admission", type=Path, help="fresh output, still requiring a separate independent admission review")
    parser.add_argument("--qualification-result", type=Path)
    parser.add_argument("--qualification-review", type=Path)
    args = parser.parse_args(argv)
    if args.write_admission is not None:
        if args.qualification_result is None or args.qualification_review is None:
            parser.error("--write-admission needs both actual reviewed qualification inputs")
        result = create_admission(args.qualification_result, args.qualification_review, args.write_admission)
    else:
        if args.qualification_result is not None or args.qualification_review is not None:
            parser.error("qualification inputs belong only to --write-admission")
        result = prepare(args.request) if args.request is not None else controller(args.prepared_unit)
    print(json.dumps({"completed": True, "path": str(result) if isinstance(result, Path) else None, **ZERO}))


if __name__ == "__main__":
    main()
