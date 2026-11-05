"""Require reviewed actual common correctness before intermediate direct H100 confirmation."""
import argparse
import ast
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import time

NAME = "direct_common_confirmation.py"
HISTORICAL_NAME = "direct_historical_consumer.py"
HISTORICAL_PROTOCOL = {"schema_version": 1, "capability": "exact_old_staged_direct_consumer_read_only_process",
    "old_helper_commit": "0a880984b4a03a2ff7335337cc18576628bc9fc1", "old_helper_count": 26,
    "current_common_helper_count": 28, "digest_cache": "disabled", "operational_admission": False}
_ACTIVE_HISTORICAL_SOURCE_SHA256 = None
REPOSITORY_CONFIGURATION_QUERY = ["config", "--null", "--includes", "--name-only", "--get-regexp",
    r"^(filter\.|include\.|includeif\.)"]
_REPOSITORY_CONFIGURATION = {}
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
NUMERICAL = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "scipy", "cupy", "jaxtari", "exploration_platform"}
PROOF_NAME = "direct_common_correctness_validation.json"
LIMIT = 1024 * 1024
HISTORY_DECLARATIONS_SHA256 = "3da2c4491a0c9c123b7de529e0e25c916e7654bb0fd535fcb0169edde49860e0"
HISTORY_DECLARATIONS_RELATIVE_PATH = (
    "remote_setup/2026-11-01-20-53_original-qualification_h100-nvl_serval02-card-1_driver-590.48.01_"
    "gpu-count-1_processor-count-8_policies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_"
    "warmup-rollouts-50_reference-updates-1_seed-1_memory-mib-95830/history_preparation_inputs.json")

CARD_ONE = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
DEFAULT_EXECUTION = {"compilation_mode": "whole_iteration", "iterations_per_call": 1,
    "report_readout": "none", "report_every_iterations": 10}
ZERO_FIELDS = {"scientific_credit", "idea_credit", "case_credit", "throughput_credit", "final_training_credit", "site_qualification_credit"}
REVIEW_FIELDS = {"schema_version", "kind", "approved", "passed", "result", "plan", "source_commit", "helper_commit",
    "copies", "copy_seeds", "configuration", "reference_configuration", "gpu_uuid", "resolved_binding", "terminal_process",
    "raw_audit", "hostname", "pid", "process_start_time", "actual_returncode", "independent_of_producer", "reviewer_identity",
    "reviewed_at_seconds", "unresolved_anomalies", "admission_granted", "worker_admitted", *ZERO_FIELDS}
RAW_FIELDS = {"schema_version", "kind", "passed", "result", "plan", "source_commit", "helper_commit", "copies", "copy_seeds",
    "configuration", "reference_configuration", "gpu_uuid", "resolved_binding", "execution_identity", "unchanged_staged_consumer_passed",
    "terminal_process", "wait_record", "controller_spawn", "controller_outcome", "controller_output_hashes", "asset_receipt",
    "installed_backend", "loaded_gpu_libraries_sha256", "captured_backend", "direct_whole_work_memory_sha256", "hostname", "pid",
    "process_start_time", "actual_returncode", "admission_granted", "worker_admitted", "backend_expected",
    "backend_preparation", "backend_capture_begin", "post_installed_package_sha256", "qualification_source_contract", *ZERO_FIELDS}


def require(condition, message):
    """Reject an incomplete or changed prerequisite without a replacement path."""
    if not condition:
        raise ValueError(message)


def encoded(value):
    """Serialize strict metadata with stable key order."""
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(raw):
    """Identify an exact saved byte sequence."""
    return hashlib.sha256(raw).hexdigest()


def read(path):
    """Read an ordinary exact JSON file with unique keys and finite numbers."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "an exact ordinary prerequisite file is required")
    def unique(pairs):
        """Reject conflicting copies of one metadata key."""
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate common-confirmation key")
            result[key] = value
        return result
    raw = path.read_bytes()
    return json.loads(raw, object_pairs_hook=unique, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite prerequisite metadata"))), raw


def reference(path):
    """Keep a real ordinary artifact's exact path and bytes."""
    unused, raw = read(path)
    return {"path": str(path), "sha256": digest(raw)}


def checked_reference(value):
    """Refuse a changed referenced result or review."""
    require(isinstance(value, dict) and set(value) == {"path", "sha256"} and reference(value["path"]) == value, "common correctness prerequisite changed")
    return read(value["path"])[0]


def accepted_snapshot_support(plan):
    """Load only the explicitly declared adjacent accepted-history metadata reader."""
    import direct_measurement as support
    module = support.adjacent("accepted_evidence_snapshot")
    module.require_plan(plan)
    return module


def contract(plan):
    """Limit this helper to the exact direct common seven-by-ten comparison."""
    require(plan.get("kind") == "selected_configuration_measurement" and plan.get("mode") == "confirmation"
        and "direct_binding" in plan and plan.get("hardware_class") == "h100"
        and plan.get("gpu_uuid") == CARD_ONE and not any(key in plan for key in (
            "configured_isolation", "configured_group", "target_qualification", "target_qualification_inputs",
            "direct_qualification", "final_confirmation_protocol", "final_protocol", "completed_study",
            "completed_study_manifest", "device_memory_limit_protocol", "fixed_total_scaling", "scaling_protocol")),
        "direct common correctness is only for the approved card-one intermediate comparison")
    require(all(type(plan.get(key)) is int and plan[key] == value for key, value in (
        ("rounds", 7), ("minimum_updates_per_round", 10), ("iterations", 10))),
        "direct common confirmation needs seven pairs of ten complete updates")
    backend = plan.get("direct_backend_image")
    require(type(backend) is dict and set(backend) == {"schema_version", "capability", "expected"}
        and type(backend["schema_version"]) is int and backend["schema_version"] == 2
        and backend["capability"] == "prepared_owning_package_image_and_same_process_mapping_version_2"
        and type(backend["expected"]) is dict and set(backend["expected"]) == {"path", "sha256"}
        and type(backend["expected"]["path"]) is str and Path(backend["expected"]["path"]).is_absolute()
        and type(backend["expected"]["sha256"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", backend["expected"]["sha256"])
        and len(plan["helpers"]["files"]) == (29 if "accepted_evidence_snapshot_protocol" in plan else 28)
        and plan["helpers"]["files"].get(PREFIX + "direct_backend_image.py") ==
            digest(Path(__file__).with_name("direct_backend_image.py").read_bytes()),
        "direct common confirmation requires its exact backend record and complete declared helper set")
    if "accepted_evidence_snapshot_protocol" in plan:
        require("direct_historical_consumer_protocol" not in plan,
            "accepted history and repeated historical raw recomputation cannot be mixed")
        accepted_snapshot_support(plan)
    else:
        historical = Path(__file__).with_name(HISTORICAL_NAME)
        require(encoded(plan.get("direct_historical_consumer_protocol")) == encoded(HISTORICAL_PROTOCOL)
            and historical.resolve(strict=True) == historical and historical.is_file()
            and plan["helpers"]["files"].get(PREFIX + HISTORICAL_NAME) == digest(historical.read_bytes()),
            "the explicit current read-only historical consumer protocol and source are required")
    inputs = plan.get("common_correctness_inputs")
    require(isinstance(inputs, dict) and set(inputs) == {"result", "review"},
        "actual direct common correctness and genuine independent review are required")
    require(all(set(plan["configuration"][scope]) == set(plan["reference_configuration"][scope])
        for scope in ("learner", "environment", "execution")), "common comparison configuration fields differ")
    changes = [(scope, key, plan["reference_configuration"][scope][key], value)
        for scope in ("learner", "environment", "execution") for key, value in plan["configuration"][scope].items()
        if type(value) is not type(plan["reference_configuration"][scope][key])
        or encoded(value) != encoded(plan["reference_configuration"][scope][key])]
    require(sorted(changes) == [("environment", "cache_room_backgrounds", False, True),
        ("environment", "render_final_two_only", False, True)]
        and all(type(plan["configuration"]["environment"][key]) is bool
            and type(plan["reference_configuration"]["environment"][key]) is bool
            for key in ("cache_room_backgrounds", "render_final_two_only")),
        "direct common candidate must change exactly its two accepted environment settings")
    copies = plan["configuration"]["learner"]["n_copies"]
    require(type(copies) is int and copies in (1, 2)
        and plan["copy_seeds"] == list(range(1, copies + 1))
        and all(type(seed) is int for seed in plan["copy_seeds"]),
        "direct common confirmation changed independent copy/seed identities")
    require(digest(Path(__file__).read_bytes()) == plan["helpers"]["files"].get(PREFIX + NAME),
        "the exact optional direct prerequisite helper must be staged")
    global _ACTIVE_HISTORICAL_SOURCE_SHA256
    _ACTIVE_HISTORICAL_SOURCE_SHA256 = (None if "accepted_evidence_snapshot_protocol" in plan
        else plan["helpers"]["files"][PREFIX + HISTORICAL_NAME])
    return inputs



def controller_evidence(support, directory, metadata):
    """Join actual child spawn, unchanged parent birth and uninterrupted normal wait."""
    identity = support.core()
    process = metadata["after"]["identity"]
    terminal = support.terminal_process(directory, metadata)
    wait = read(directory / "direct_process_wait_return.json")[0]
    spawn = read(directory / "direct_controller_spawn.json")[0]
    outcome = read(directory / "direct_controller_outcome.json")[0]
    outputs = read(directory / "direct_controller_output_hashes.json")[0]
    child, parent = spawn["observed_child"], spawn["controller"]
    require(spawn.get("kind") == "actual_direct_qualification_popen_return"
        and type(spawn.get("popen_pid")) is int and spawn["popen_pid"] == process["pid"]
        and all(identity.exact(child[key], process[key]) for key in ("hostname", "pid", "process_start_time", "parent_pid"))
        and parent["pid"] == process["parent_pid"] and spawn["command"] == child["command"] == wait["command"],
        "actual direct correctness child differs from its original observed controller spawn")
    before = wait["waiter_before"]["stat"].rsplit(")", 1)[1].split()[19]
    after = wait["waiter_after"]["stat"].rsplit(")", 1)[1].split()[19]
    require(parent["process_start_time"] == before == after and wait.get("timeouts") == []
        and type(wait.get("returncode")) is int and wait["returncode"] == 0
        and wait.get("exited_normally") is True, "direct correctness lacks an uninterrupted same-parent normal wait")
    require(outcome.get("kind") == "direct_qualification_controller_outcome"
        and outcome.get("passed") is True and outcome.get("actual_child_started") is True
        and outcome.get("actual_wait_saved") is True and outcome.get("failure") is None
        and outcome.get("qualification_independently_verified") is False,
        "the actual correctness controller did not complete its own child normally")
    for document in (spawn, wait, outcome):
        require(document.get("controller_owns_gpu_standing_descriptor") is False
            and document.get("producer_owns_gpu_standing_descriptor") is True,
            "the direct controller cannot claim the producer-owned standing descriptor")
    require(outputs.get("kind") == "actual_direct_child_output_hashes_after_wait"
        and outputs.get("wait") == reference(directory / "direct_process_wait_return.json"),
        "actual correctness output capture does not bind the completed exact wait")
    for key, name in (("stdout", "direct_controller_stdout.raw"), ("stderr", "direct_controller_stderr.raw")):
        path = directory / name
        raw = path.read_bytes()
        require(outputs[key] == {"path": str(path), "sha256": digest(raw), "size_bytes": len(raw)},
            "actual correctness output bytes changed after normal wait")
    return {name: reference(directory / filename) for name, filename in (
        ("terminal_process", "direct_terminal_process.json"), ("wait_record", "direct_process_wait_return.json"),
        ("controller_spawn", "direct_controller_spawn.json"), ("controller_outcome", "direct_controller_outcome.json"),
        ("controller_output_hashes", "direct_controller_output_hashes.json"))}


def expected_review_identity(previous, result, binding, metadata, directory):
    """Describe exact observed fields for an independently authored review; write no approval."""
    process = metadata["after"]["identity"]
    return {"result": reference(directory / "result.json"), "plan": reference(directory / "measurement_plan.json"),
        "source_commit": previous["source_commit"], "helper_commit": previous["helpers"]["source_commit"],
        "copies": previous["configuration"]["learner"]["n_copies"], "copy_seeds": previous["copy_seeds"],
        "configuration": previous["configuration"], "reference_configuration": previous["reference_configuration"],
        "gpu_uuid": binding["gpu_uuid"], "resolved_binding": reference(directory / "direct_device_binding.json"),
        "terminal_process": reference(directory / "direct_terminal_process.json"),
        **{key: process[key] for key in ("hostname", "pid", "process_start_time")}, "actual_returncode": 0}


def qualification_reuse_evidence(support, plan):
    """Limit reuse to the unchanged numerical work and the exact new package protocol."""
    identity = support.core()
    inputs = plan["direct_qualification_inputs"]
    qualified_result = checked_reference(inputs["result"])
    qualification_review = checked_reference(inputs["review"])
    directory = Path(inputs["result"]["path"]).parent
    qualified = read(directory / "measurement_plan.json")[0]
    require(len(qualified["helpers"]["files"]) == 26
        and PREFIX + NAME not in qualified["helpers"]["files"]
        and "common_correctness_inputs" not in qualified,
        "qualification reuse requires the separate original 26-helper qualification")
    for key in ("source_root", "source_commit", "core", "paired_source", "python", "hardware_class",
            "runtime_environment", "asset_record"):
        require(identity.exact(qualified[key], plan[key]), "qualification reuse changed protected input: " + key)
    require(qualification_review.get("raw_validation_passed_before_review_existed") is True
        and qualification_review.get("placeholder_or_partial_approval_used") is False
        and qualification_review.get("result") == inputs["result"],
        "qualification reuse needs the actual raw-first independent review")
    source_ref = qualification_review["source_contract"]
    source = checked_reference(source_ref)
    require(source.get("kind") == "exact_fresh_direct_qualification_raw_reader_two_review_statement_difference"
        and type(source.get("original_body_statement_count")) is int and source["original_body_statement_count"] == 55
        and type(source.get("retained_body_statement_count")) is int and source["retained_body_statement_count"] == 53
        and source.get("removed_statement_indices") == [5, 50]
        and source["staged_source"]["path"] == str(directory / "code/direct_measurement.py")
        and identity.receipt(source["staged_source"]["path"]) == source["staged_source"]
        and identity.receipt(source["new_raw_reader"]["path"]) == source["new_raw_reader"],
        "qualification reuse requires the exact new 55-statement consumer and all 53 raw gates")
    original = next(node for node in ast.parse(Path(source["staged_source"]["path"]).read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "verify_qualification_inputs")
    raw = next(node for node in ast.parse(Path(source["new_raw_reader"]["path"]).read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "verify_unreviewed_actual_raw_qualification")
    removed = source["removed_statement_indices"]
    retained = [ast.dump(node, include_attributes=False) for index, node in enumerate(original.body) if index not in removed]
    observed = [ast.dump(node, include_attributes=False) for node in raw.body]
    require(len(original.body) == 55 and len(raw.body) == 53 and retained == observed
        and ast.dump(original.args, include_attributes=False) == ast.dump(raw.args, include_attributes=False)
        and source["removed_statements"] == [ast.unparse(original.body[index]) for index in removed]
        and source["removed_statements"][0] == "review = identity.read_receipt(inputs['review'])"
        and "independent actual qualification review is required" in source["removed_statements"][1]
        and source["retained_body_ast_sha256"] == digest("\n".join(observed).encode()),
        "qualification reuse changed a raw gate or removed more than the genuine review dependency")
    current_backend = support.backend_support(plan).expected_for_plan(plan)
    qualified_backend = support.backend_support(qualified).expected_for_plan(qualified)
    require(all(identity.exact(current_backend[key], qualified_backend[key])
        for key in ("selected_python", "site_packages", "image", "distribution", "producer_source_sha256"))
        and identity.exact(plan["inputs"]["direct_backend_image_record"], qualified["inputs"]["direct_backend_image_record"]),
        "qualification reuse changed the prepared image, owning package or exact backend helper")
    protected = {}
    for label in ("core", "paired_source"):
        record = plan[label]
        for relative, wanted in record["files"].items():
            wanted = wanted["sha256"] if type(wanted) is dict else wanted
            require(relative not in protected and identity.receipt(Path(record["source_root"]) / relative)["sha256"] == wanted,
                "qualification reuse changed a numerical source file")
            protected[relative] = wanted
    require(len(protected) == 13, "qualification reuse must retain all thirteen numerical files")
    functions = {}
    scopes = {"measure_selected.py": ("_compute", "run_prepared", "save_measurement_result", "producer_support", "card_admission"),
        "direct_measurement.py": ("run_qualification", "support", "capture_backend",
            "capture_backend_completion", "post_binding_checks", "process_sample", "validate_window")}
    for filename, names in scopes.items():
        old_path = directory / "code" / filename
        new_path = Path(__file__).with_name(filename)
        relative = PREFIX + filename
        require(identity.receipt(old_path)["sha256"] == qualified["helpers"]["files"][relative]
            and identity.receipt(new_path)["sha256"] == plan["helpers"]["files"][relative],
            "qualification reuse lost its exact old/current helper source")
        def bodies(path):
            """Compare actual source syntax without importing numerical code."""
            return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef)}
        old, new = bodies(old_path), bodies(new_path)
        for name in names:
            require(name in old and old[name] == new.get(name), "qualification reuse changed protected execution: " + name)
            functions[filename + ":" + name] = digest(old[name].encode())
    extension = qualification_reader_extension(directory / "code/direct_measurement.py",
        Path(__file__).with_name("direct_measurement.py"))
    require(qualified_result["source_commit"] == qualified["source_commit"], "qualified numerical source label changed")
    return {"qualified_result": inputs["result"], "qualified_plan": reference(directory / "measurement_plan.json"),
        "source_contract": source_ref, "protected_numerical_files": protected, "protected_function_syntax": functions,
        "backend_preparation": plan["inputs"]["direct_backend_image_record"], "numerical_work_changed": False,
        "qualification_reader_metadata_extension": extension}


def qualification_reader_extension(original_path, current_path):
    """Retain the entire original reader body after one explicit read-only metadata dispatch."""
    def function(path):
        """Read source syntax without executing any old or current producer."""
        matches = [node for node in ast.parse(Path(path).read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "verify_qualification_inputs"]
        require(len(matches) == 1, "the exact qualification reader function is required")
        return matches[0]
    def source_reference(path):
        """Bind Python source bytes without parsing them as a JSON record."""
        path = Path(path)
        require(path.resolve(strict=True) == path and path.is_file(), "an exact ordinary source file is required")
        return {"path": str(path), "sha256": digest(path.read_bytes())}
    old, current = function(original_path), function(current_path)
    statement = ast.parse('if "common_correctness_inputs" in plan:\n    return historical_support(plan).request(plan, "qualification")\n').body[0]
    require(len(current.body) == len(old.body) + 1
        and ast.dump(current.body[0], include_attributes=False) == ast.dump(old.body[0], include_attributes=False)
        and ast.dump(current.body[1], include_attributes=False) == ast.dump(statement, include_attributes=False)
        and [ast.dump(node, include_attributes=False) for node in current.body[2:]] ==
            [ast.dump(node, include_attributes=False) for node in old.body[1:]]
        and ast.dump(current.args, include_attributes=False) == ast.dump(old.args, include_attributes=False)
        and current.decorator_list == old.decorator_list and current.returns == old.returns,
        "qualification reuse changed an original gate or broadened the metadata dispatch")
    return {"kind": "one_explicit_current_common_read_only_qualification_dispatch",
        "original_reader_source": source_reference(original_path), "current_reader_source": source_reference(current_path),
        "original_body_statements": len(old.body), "current_body_statements": len(current.body),
        "inserted_statement_index": 1, "inserted_statement": ast.unparse(statement),
        "entire_original_body_retained": True, "old_operational_path_changed": False}


def validate_actual(plan, plan_sha256):
    """Read actual direct correctness and terminal raw gates in this isolated metadata child."""
    inputs = contract(plan)
    if "accepted_evidence_snapshot_protocol" in plan:
        return accepted_snapshot_support(plan).common_validation(plan, plan_sha256)
    import direct_measurement as support
    import post_comparison_validation as validation
    import diagnose_gpu_execution as diagnostic
    import train_optimized as training
    identity = support.core()
    identity.validate_plan(plan)
    result, review = checked_reference(inputs["result"]), checked_reference(inputs["review"])
    directory = Path(inputs["result"]["path"]).parent
    require(Path(inputs["result"]["path"]).name == "result.json",
        "direct common correctness needs the actual ordinary correctness result")
    historical = support.historical_support(plan).request(plan, "correctness")
    previous = read(directory / "measurement_plan.json")[0]
    binding = read(directory / "direct_device_binding.json")[0]
    require(historical["proof"]["old_plan"] == reference(directory / "measurement_plan.json")
        and historical["proof"]["old_inputs"] == inputs
        and historical["proof"]["old_consumer_return_sha256"] == digest(encoded([previous, binding])),
        "the exact isolated historical correctness consumer returned another saved binding")
    reuse = qualification_reuse_evidence(support, plan)
    require(previous.get("mode") == "correctness" and "common_correctness_inputs" not in previous
        and result.get("measurement_mode") == "correctness" and result.get("measurement_valid") is False
        and result.get("implementation_distinct") is True, "timing cannot replace separately reviewed correctness")
    validation.correctness(result)
    validation.configuration(plan["configuration"])
    validation.configuration(plan["reference_configuration"])
    validation.execution_identity(result)
    identity.require_profile(plan["direct_binding"]["profile"])
    for key in ("campaign_run_dir", "source_root", "source_commit", "core", "paired_source", "hardware_class",
            "runtime_environment", "asset_record", "copy_seeds", "configuration", "reference_configuration",
            "direct_qualification_inputs"):
        require(key in previous and key in plan and identity.exact(previous[key], plan[key]),
            "reviewed direct correctness differs from confirmation: " + key)
    expected_backend = support.backend_support(plan).expected_for_plan(plan)
    previous_backend = support.backend_support(previous).expected_for_plan(previous)
    require(all(identity.exact(expected_backend[key], previous_backend[key])
        for key in ("selected_python", "site_packages", "image", "distribution", "producer_source_sha256"))
        and identity.exact(plan["inputs"]["direct_backend_image_record"], previous["inputs"]["direct_backend_image_record"]),
        "reviewed correctness changed its exact prepared backend package or producer")
    require(identity.exact(previous["direct_binding"]["profile"], plan["direct_binding"]["profile"])
        and binding["gpu_uuid"] == CARD_ONE, "direct common correctness changed its sole approved physical target")
    defaults = diagnostic.default_configuration(Path(plan["source_root"]))
    learner = json.loads(json.dumps(defaults["learner"]))
    learner["n_copies"] = plan["configuration"]["learner"]["n_copies"]
    require(identity.exact(plan["reference_configuration"]["learner"], learner)
        and identity.exact(plan["reference_configuration"]["environment"], json.loads(json.dumps(defaults["environment"])))
        and identity.exact(plan["reference_configuration"]["execution"], DEFAULT_EXECUTION),
        "common direct reference must preserve every original source configuration setting")
    metadata = read(directory / "direct_process_metadata.json")[0]
    terminal_refs = controller_evidence(support, directory, metadata)
    expected = expected_review_identity(previous, result, binding, metadata, directory)
    require(set(review) == REVIEW_FIELDS and type(review.get("schema_version")) is int and review["schema_version"] == 1
        and review.get("kind") == "independent_actual_direct_common_configuration_correctness_review"
        and review.get("approved") is True and review.get("passed") is True
        and review.get("independent_of_producer") is True and review.get("unresolved_anomalies") == []
        and type(review.get("reviewer_identity")) is str and bool(review["reviewer_identity"].strip()),
        "a genuine complete independent direct correctness review is required")
    for key, expected_value in expected.items():
        require(identity.exact(review.get(key), expected_value), "independent direct review changed actual field: " + key)
    for key in ZERO_FIELDS:
        require(type(review.get(key)) is int and review[key] == 0, "correctness review cannot grant timing/scientific credit")
    require(review["admission_granted"] is False and review["worker_admitted"] is False
        and type(review["reviewed_at_seconds"]) in (int, float) and math.isfinite(review["reviewed_at_seconds"])
        and result["execution_identity"]["finished_at_seconds"] <= review["reviewed_at_seconds"],
        "direct correctness review predates completion or changes admission")
    raw = checked_reference(review["raw_audit"])
    require(set(raw) == RAW_FIELDS and type(raw.get("schema_version")) is int and raw["schema_version"] == 1
        and raw.get("kind") == "independent_actual_direct_common_correctness_raw_validation"
        and raw.get("passed") is True and raw.get("unchanged_staged_consumer_passed") is True
        and identity.exact(raw.get("execution_identity"), result["execution_identity"]),
        "independent direct review lacks complete actual raw validation")
    for key, expected_value in expected.items():
        if key in RAW_FIELDS:
            require(identity.exact(raw.get(key), expected_value), "raw direct validation changed actual field: " + key)
    for key, expected_value in terminal_refs.items():
        require(raw.get(key) == expected_value, "raw direct validation changed exact normal-controller proof: " + key)
    for key in ZERO_FIELDS:
        require(type(raw.get(key)) is int and raw[key] == 0, "raw correctness cannot grant scientific credit")
    require(raw["admission_granted"] is False and raw["worker_admitted"] is False,
        "raw correctness cannot admit a worker")
    backend = read(directory / "installed_backend.json")[0]
    loaded = result["loaded_gpu_libraries"]
    training.validate_gpu_library_receipt(loaded, {"installed_gpu_libraries": backend["gpu_libraries"]})
    require(loaded["pid"] == metadata["after"]["identity"]["pid"]
        and binding["native"]["driver_library"] in loaded["loaded_shared_libraries"],
        "actual loaded libraries do not belong to this direct bound process")
    for path, image in loaded["verified_package_libraries"].items():
        require(identity.receipt(path)["sha256"] == image["sha256"], "actual installed CUDA/cuDNN image changed")
    actual_assets = training.verify_assets(plan["asset_record"], {})
    require(read(directory / "asset_receipt.json")[0] == actual_assets,
        "actual correctness assets differ from the exact checked original asset bytes")
    require(raw["asset_receipt"] == reference(directory / "asset_receipt.json")
        and raw["installed_backend"] == reference(directory / "installed_backend.json")
        and raw["captured_backend"] == reference(directory / "direct_backend_capture.json")
        and raw["loaded_gpu_libraries_sha256"] == digest(encoded(loaded))
        and raw["direct_whole_work_memory_sha256"] == digest(encoded(result["direct_whole_work_memory"])),
        "raw direct correctness changed actual asset, backend, library or complete sample evidence")
    capture = read(directory / "direct_backend_capture.json")[0]
    require(raw["backend_expected"] == previous["direct_backend_image"]["expected"]
        and raw["backend_preparation"] == previous["inputs"]["direct_backend_image_record"]
        and raw["backend_capture_begin"] == reference(directory / "direct_backend_capture_begin.json")
        and raw["post_installed_package_sha256"] == digest(encoded(capture["post_installed_package"]))
        and raw["qualification_source_contract"] == reuse["source_contract"],
        "independent correctness raw review omitted the new actual package or unchanged qualification scope")
    require(not NUMERICAL.intersection(sys.modules), "a numerical module entered the isolated direct metadata child")
    return {"schema_version": 1, "kind": "isolated_direct_common_correctness_validation", "passed": True,
        "plan_sha256": plan_sha256, "common_correctness_inputs": inputs,
        "correctness_plan": reference(directory / "measurement_plan.json"), "prior_correctness_identity": expected,
        "prior_correctness_reviewed_at_seconds": review["reviewed_at_seconds"],
        "qualification_reuse_evidence": reuse,
        "source_commit": plan["source_commit"], "helper_commit": plan["helpers"]["source_commit"],
        "parent_pid": os.getppid(), "validation_pid": os.getpid(), "finished_at_seconds": time.time(),
        "numerical_imports": [], "gpu_operations": 0, "scientific_credit": 0,
        "learning_equivalence_established": False, "admission_granted": False,
        "historical_correctness_consumer": historical}



def request(plan, raw=None, directory=None):
    """Run the full saved consumer in a fresh no-site process without changing parent imports."""
    contract(plan)
    require(not NUMERICAL.intersection(sys.modules), "common correctness must be checked before numerical initialization")
    raw = encoded(plan) if raw is None else raw
    executable = Path(directory) / "code" / NAME if directory is not None else Path(__file__).absolute()
    command = ["/usr/bin/python3", "-I", "-S", "-B", str(executable), "--plan-stdin"]
    try:
        result = subprocess.run(command, input=raw, capture_output=True, timeout=1800, check=False,
            env=metadata_environment(os.environ))
    except BaseException as error:
        if directory is not None:
            publish_child_capture(directory, command, None, getattr(error, "stdout", None) or b"",
                getattr(error, "stderr", None) or b"", type(error).__name__)
        raise
    if directory is not None:
        publish_child_capture(directory, command, result.returncode, result.stdout, result.stderr)
    require(len(result.stdout) <= LIMIT and len(result.stderr) <= LIMIT, "common correctness child output exceeded its bound")
    require(result.returncode == 0 and not result.stderr, "isolated common correctness validation failed: " + result.stderr.decode(errors="replace"))
    proof = json.loads(result.stdout)
    require(proof.get("passed") is True and proof.get("plan_sha256") == digest(raw)
        and proof.get("common_correctness_inputs") == plan["common_correctness_inputs"]
        and type(proof.get("parent_pid")) is int and proof["parent_pid"] == os.getpid()
        and proof.get("numerical_imports") == [] and type(proof.get("gpu_operations")) is int and proof["gpu_operations"] == 0,
        "the isolated correctness child returned another plan or parent")
    if "accepted_evidence_snapshot_protocol" in plan:
        require(proof.get("kind") == "reviewed_accepted_direct_common_correctness_validation"
            and proof.get("accepted_evidence_snapshot_inputs") == plan["accepted_evidence_snapshot_inputs"]
            and proof.get("historical_raw_gates_recomputed") is False
            and proof.get("cpu_snapshot_reused_as_reviewed_historical_fact") is True
            and proof.get("gpu_time_historical_byte_equality_asserted") is False
            and proof.get("repository_configuration_validation") == {
                "kind": "reviewed_historical_cpu_snapshot_no_repository_commands",
                "repository_commands_executed": 0, "current_live_source_validation_performed_elsewhere": True},
            "the metadata child did not retain the declared complete accepted historical fact")
    else:
        validate_repository_configuration_proof(proof.get("repository_configuration_validation"))
    validate_bootstrap_record(proof.get("stdlib_ctypes_bootstrap"))
    return proof


def publish(path, raw):
    """Save one new durable validation artifact without replacing an earlier attempt."""
    require(path.parent.resolve(strict=True) == path.parent and not path.exists() and not path.is_symlink(),
        "preserve every prior common correctness validation artifact")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_child_capture(directory, command, returncode, stdout, stderr, error=None):
    """Preserve actual child returns and raw bytes before any success parsing."""
    directory = Path(directory)
    publish(directory / "direct_common_correctness_validation_stdout.raw", stdout)
    publish(directory / "direct_common_correctness_validation_stderr.raw", stderr)
    publish(directory / "direct_common_correctness_validation_return.json", encoded({"command": command,
        "returncode": returncode, "stdout_sha256": digest(stdout), "stderr_sha256": digest(stderr), "error": error}))


def before_compute(directory, plan):
    """Publish one actual successful child check before any numerical import or device work."""
    path = Path(directory) / PROOF_NAME
    require(not path.exists() and not path.is_symlink(), "preserve the prior common correctness validation attempt")
    raw = (Path(directory) / "measurement_plan.json").read_bytes()
    publish(Path(directory) / "direct_common_correctness_validation_attempt.json", encoded({"plan_sha256": digest(raw),
        "common_correctness_inputs": plan["common_correctness_inputs"], "worker_pid": os.getpid(), "retry_permitted": False}))
    proof = request(plan, raw, directory)
    publish(path, encoded(proof))
    return proof


def result_binding(directory, plan):
    """Bind the saved result to the already completed pre-numerical validation."""
    contract(plan)
    path = Path(directory) / PROOF_NAME
    proof = read(path)[0]
    captured = read(Path(directory) / "direct_common_correctness_validation_return.json")[0]
    stdout = (Path(directory) / "direct_common_correctness_validation_stdout.raw").read_bytes()
    stderr = (Path(directory) / "direct_common_correctness_validation_stderr.raw").read_bytes()
    command = ["/usr/bin/python3", "-I", "-S", "-B", str(Path(directory) / "code" / NAME), "--plan-stdin"]
    require(captured.get("command") == command and type(captured.get("returncode")) is int
        and captured["returncode"] == 0 and captured.get("error") is None
        and captured.get("stdout_sha256") == digest(stdout) and captured.get("stderr_sha256") == digest(stderr)
        and stderr == b"" and json.loads(stdout) == proof, "the actual metadata child result changed")
    expected_kind = ("reviewed_accepted_direct_common_correctness_validation" if "accepted_evidence_snapshot_protocol" in plan
        else "isolated_direct_common_correctness_validation")
    require(proof.get("kind") == expected_kind and proof.get("passed") is True
        and proof.get("plan_sha256") == digest((Path(directory) / "measurement_plan.json").read_bytes())
        and proof.get("common_correctness_inputs") == plan["common_correctness_inputs"], "common correctness validation no longer binds this plan")
    validate_bootstrap_record(proof.get("stdlib_ctypes_bootstrap"))
    return {"common_correctness_inputs": plan["common_correctness_inputs"], "direct_common_correctness_validation": reference(path)}



def validate_pair_protocol(plan, result):
    """Require seven alternating pairs with ten actual full updates and matched schedule origins."""
    copies = plan["configuration"]["learner"]["n_copies"]
    pairs = result.get("rounds")
    require(type(pairs) is list and len(pairs) == 7, "direct common confirmation must retain all seven paired rounds")
    previous_origin = None
    for number, pair in enumerate(pairs, 1):
        order = ["reference", "candidate"] if number % 2 else ["candidate", "reference"]
        require(type(pair.get("round")) is int and pair["round"] == number and pair.get("order") == order,
            "direct common confirmation did not preserve alternating execution order")
        origins = []
        for arm in ("reference", "candidate"):
            row = pair[arm]
            require(all(type(row.get(key)) is int and row[key] == expected for key, expected in (
                ("iterations", 10), ("copies", copies), ("environments_per_copy", 128), ("rollout_steps", 128),
                ("training_transitions", copies * 128 * 128 * 10)))
                and type(row.get("seconds")) in (int, float) and math.isfinite(row["seconds"]) and row["seconds"] > 0
                and type(row.get("report_origin_iteration")) is int and row["report_origin_iteration"] >= 0,
                "direct common confirmation changed full-size actual work or its schedule origin")
            origins.append(row["report_origin_iteration"])
        require(origins[0] == origins[1] and (previous_origin is None or origins[0] == previous_origin + 10),
            "direct timing arms used different or incomplete supplied training schedules")
        previous_origin = origins[0]

def verify_saved(directory, plan, result):
    """Recheck the saved prerequisite and its actual validation before accepting timing evidence."""
    expected = result_binding(directory, plan)
    require(all(result.get(key) == value for key, value in expected.items()), "the confirmation result omitted its reviewed correctness binding")
    proof = checked_reference(expected["direct_common_correctness_validation"])
    require(type(proof.get("parent_pid")) is int and proof["parent_pid"] == result["execution_identity"]["pid"]
        and type(proof.get("finished_at_seconds")) in (int, float) and math.isfinite(proof["finished_at_seconds"])
        and proof["finished_at_seconds"] <= result["execution_identity"]["started_at_seconds"],
        "common correctness validation was not completed by this worker before numerical work")
    request(plan, (Path(directory) / "measurement_plan.json").read_bytes())
    validate_pair_protocol(plan, result)



def metadata_compute_context(plan):
    """Allow actual raw validation only on this owned compute job or the approved direct host."""
    import socket
    uid_lines = [line for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("Uid:")]
    require(os.getuid() == os.geteuid() == 2618919 and os.getresuid() == (2618919, 2618919, 2618919)
        and len(uid_lines) == 1 and list(map(int, uid_lines[0].split()[1:])) == [2618919] * 4,
        "direct prerequisite raw validation requires the exact owner")
    host = socket.gethostname().split(".")[0]
    job, step = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID")
    if host == "serval02":
        require(job is None and step is None, "direct metadata validation cannot use a Slurm fallback")
    else:
        require(type(job) is str and job.isdecimal() and int(job) > 0
            and type(step) is str and step.isdecimal(), "raw direct validation requires owned compute, never a login node")
        ledger = Path(plan["campaign_run_dir"]) / "slurm/submitted_jobids_uva.txt"
        require(ledger.resolve(strict=True) == ledger and ledger.read_text().splitlines().count(job) == 1,
            "metadata child is not in this campaign's exact owned UVA ledger")
        expected = "/system.slice/slurmstepd.scope/job_" + job + "/step_" + step + "/user/task_0"
        require(Path("/proc/self/cgroup").read_text().splitlines() == ["0::" + expected],
            "metadata child must retain its exact own numeric compute step")
    return {"hostname": host, "job_id": job, "step_id": step, "uid": 2618919}


def pinned_history_declarations(plan):
    """Use the actual published original-history declaration, before any Git operation."""
    from pathlib import PurePosixPath
    path = Path(plan["campaign_run_dir"]) / HISTORY_DECLARATIONS_RELATIVE_PATH
    declared = checked_reference({"path": str(path), "sha256": HISTORY_DECLARATIONS_SHA256})
    require(type(declared.get("schema_version")) is int and declared["schema_version"] == 1
        and declared.get("kind") == "exact_original_direct_history_preparation_inputs"
        and type(declared.get("git_sources")) is list and len(declared["git_sources"]) == 8,
        "the exact pinned eight-source original history declaration is required")
    expected, roots = {}, set()
    for row in declared["git_sources"]:
        require(type(row) is dict and set(row) == {"original_root", "commit", "transport_repository"},
            "unknown original historical source declaration")
        for key in ("original_root", "transport_repository"):
            value = row[key]
            require(type(value) is str and value.startswith("/") and not value.startswith("//")
                and "\0" not in value and ".." not in PurePosixPath(value).parts
                and str(PurePosixPath(value)) == value, "invalid pinned original source identity")
        require(row["original_root"] not in expected and type(row["commit"]) is str
            and re.fullmatch(r"[0-9a-f]{40}", row["commit"]), "duplicate or invalid original source commit")
        expected[row["original_root"]] = row["commit"]
        roots.update((row["original_root"], row["transport_repository"]))
    return expected, roots


def declared_git_roots(plan):
    """Admit only exact current/prior/qualification source roots and their pinned history mappings."""
    import re
    from pathlib import PurePosixPath
    require(Path(plan["campaign_run_dir"]).is_absolute(), "one exact campaign root is required")
    inputs = contract(plan)
    expected_history, historical_roots = pinned_history_declarations(plan)
    checked_reference(inputs["result"])
    previous_directory = Path(inputs["result"]["path"]).parent
    previous = read(previous_directory / "measurement_plan.json")[0]
    qualification_ref = previous["direct_qualification_inputs"]["result"]
    checked_reference(qualification_ref)
    qualification_directory = Path(qualification_ref["path"]).parent
    qualified = read(qualification_directory / "measurement_plan.json")[0]
    roots = {"/p/rlprojects/RND/08_cleanrl_ppo_rnd/cleanrl",
        "/p/rlprojects/RND/reference_repos/jaxtari_pixel_study"}
    roots.update(historical_roots)
    for current in (plan, previous, qualified):
        unit = Path(current["campaign_run_dir"]) / "data" / current["unit_id"]
        require(current["campaign_run_dir"] == plan["campaign_run_dir"] and unit.resolve(strict=True) == unit,
            "prerequisite source must belong to this same ordinary campaign data unit")
        for record in (current["core"], current["paired_source"], current["helpers"]):
            root = Path(record["source_root"])
            require(root.is_absolute() and root.resolve(strict=True) == root and root.is_dir()
                and re.fullmatch(r"[0-9a-f]{40}", record["source_commit"]), "invalid declared source Git root")
            roots.add(str(root))
        frozen = current["direct_binding"]["frozen_inputs"]
        require(frozen["path"] == str(unit / "direct_history/inputs.json"), "history input is outside its exact unit")
        historical = checked_reference(frozen)
        manifest_ref = historical["evidence_manifest"]
        require(manifest_ref["path"] == str(unit / "direct_history/evidence_manifest.json"),
            "historical mapping must use its reserved exact unit path")
        manifest = checked_reference(manifest_ref)
        require(set(manifest) == {"schema_version", "kind", "files", "git_roots"}
            and type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
            and manifest["kind"] == "portable_raw_evidence" and type(manifest["git_roots"]) is list
            and len(manifest["git_roots"]) == 8, "the exact eight-source history mapping is required")
        staging, seen, targets = Path(manifest_ref["path"]).parent, set(), set()
        for row in manifest["git_roots"]:
            require(type(row) is dict and set(row) == {"original_root", "relative_path", "commit"}, "unknown historical source mapping")
            relative, original = row["relative_path"], row["original_root"]
            require(type(relative) is str and relative and "\0" not in relative and relative != "."
                and not PurePosixPath(relative).is_absolute() and ".." not in PurePosixPath(relative).parts
                and str(PurePosixPath(relative)) == relative and type(original) is str
                and original.startswith("/") and not original.startswith("//") and "\0" not in original
                and str(PurePosixPath(original)) == original
                and ".." not in PurePosixPath(original).parts
                and re.fullmatch(r"[0-9a-f]{40}", row["commit"]), "invalid historical source path or commit")
            target = staging / relative
            require(target.resolve(strict=True) == target and target.is_relative_to(staging) and target.is_dir()
                and original not in seen and str(target) not in targets
                and original in expected_history and row["commit"] == expected_history[original],
                "historical source target is duplicated, unlisted or not an ordinary staged directory")
            seen.add(original)
            targets.add(str(target))
        require(seen == set(expected_history) and len(targets) == 8,
            "historical mappings do not retain all eight exact original names and commits")
        roots.update(targets)
    return roots

def metadata_environment(original):
    """Give only isolated metadata children a fixed source-inspection environment, never their numerical parent."""
    # Before: arbitrary inherited Git/loader overrides. After: fixed system executables and read-only Git settings.
    result = {"PATH": "/usr/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TERM": "dumb",
        "PYTHONDONTWRITEBYTECODE": "1", "GIT_OPTIONAL_LOCKS": "0", "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_ALLOW_PROTOCOL": "",
        "GIT_NO_LAZY_FETCH": "1", "GIT_NO_REPLACE_OBJECTS": "1", "GIT_TERMINAL_PROMPT": "0",
        "GIT_ATTR_NOSYSTEM": "1", "GIT_PAGER": "", "PAGER": ""}
    configuration = (("core.fsmonitor", "false"), ("core.hooksPath", "/dev/null"),
        ("core.untrackedCache", "false"), ("core.preloadIndex", "false"),
        ("protocol.allow", "never"), ("gc.auto", "0"), ("maintenance.auto", "false"))
    result["GIT_CONFIG_COUNT"] = str(len(configuration))
    for index, (key, value) in enumerate(configuration):
        result["GIT_CONFIG_KEY_" + str(index)] = key
        result["GIT_CONFIG_VALUE_" + str(index)] = value
    for name in ("SLURM_JOB_ID", "SLURM_STEP_ID"):
        if name in original:
            result[name] = original[name]
    return result


def require_metadata_environment(value):
    """Reject overridden executable search, dynamic loaders, Git configuration or transport settings."""
    actual = dict(os.environ) if value is None else value
    require(type(actual) is dict and actual == metadata_environment(actual),
        "source-inspection subprocesses require the exact isolated metadata environment")


def source_inspection_git(command, roots):
    """Admit only the actual fixed Git source-inspection forms without extra options or output files."""
    require(type(command) is list and all(type(value) is str for value in command)
        and len(command) > 3 and command[:2] == ["git", "-C"] and command[2] in roots,
        "only exact declared local read-only Git source commands are permitted")
    def relative(value):
        """Keep a plain source path from becoming a revision, option or Git pathspec program."""
        plain = value[:-1] if value.endswith("/") else value
        return bool(plain) and plain != "." and not value.startswith(("/", "-", ":")) and "\0" not in value \
            and ".." not in Path(value).parts and Path(value).as_posix() == plain \
            and all(character.isalnum() or character in "_./-" for character in value)
    arguments = command[3:]
    allowed = arguments == REPOSITORY_CONFIGURATION_QUERY or arguments in (["rev-parse", "HEAD"], ["rev-parse", "--show-toplevel"],
        ["rev-parse", "--abbrev-ref", "HEAD"], ["symbolic-ref", "--quiet", "HEAD"],
        ["status", "--porcelain", "--untracked-files=all"], ["status", "--porcelain=v1", "-uall"])
    if len(arguments) == 2 and (arguments[0] == "show" or arguments[0] == "rev-parse" and ":" in arguments[1]):
        revision, separator, path = arguments[1].partition(":")
        allowed = bool(separator and re.fullmatch(r"[0-9a-f]{40}", revision) and relative(path) and not path.endswith("/"))
    elif len(arguments) == 3 and arguments[:2] == ["cat-file", "blob"]:
        allowed = bool(re.fullmatch(r"[0-9a-f]{40}", arguments[2]))
    elif len(arguments) >= 5 and arguments[:2] == ["ls-tree", "-rz"] and arguments[3] == "--":
        allowed = bool(re.fullmatch(r"[0-9a-f]{40}", arguments[2]) and all(map(relative, arguments[4:])))
    elif len(arguments) >= 6 and arguments[:3] == ["ls-tree", "-r", "--name-only"] and arguments[4] == "--":
        allowed = bool(re.fullmatch(r"[0-9a-f]{40}", arguments[3]) and all(map(relative, arguments[5:])))
    elif len(arguments) >= 5 and arguments[:4] == ["status", "--porcelain", "--untracked-files=all", "--"]:
        allowed = all(map(relative, arguments[4:]))
    elif len(arguments) >= 4 and arguments[:3] == ["status", "--porcelain", "--"]:
        allowed = all(map(relative, arguments[3:]))
    require(allowed, "unlisted Git source-inspection arguments are forbidden")


def repository_configuration_inputs(root):
    """Pin exact same-user Git layout and configuration files without logging configuration values."""
    import stat
    files = {}
    def file(path, optional=False):
        """Bind current ordinary bytes and full file identity, or the declared absence of an optional file."""
        path = Path(path)
        if optional and not path.exists() and not path.is_symlink():
            files[str(path)] = {"exists": False}
            return None
        require(path.resolve(strict=True) == path and not path.is_symlink(), "Git configuration path changed")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            before = os.fstat(descriptor)
            require(stat.S_ISREG(before.st_mode) and before.st_uid == 2618919 and before.st_size <= LIMIT,
                "Git configuration needs bounded ordinary same-user files")
            with os.fdopen(os.dup(descriptor), "rb") as stream:
                raw = stream.read(LIMIT + 1)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        fields = ("st_dev", "st_ino", "st_uid", "st_gid", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
        identity = {name: getattr(before, name) for name in fields}
        require(len(raw) <= LIMIT and identity == {name: getattr(after, name) for name in fields}
            and identity == {name: getattr(path.stat(), name) for name in fields},
            "Git configuration changed during its pinned read")
        files[str(path)] = {"exists": True, "identity": identity, "sha256": digest(raw)}
        return raw
    root = Path(root)
    require(root.resolve(strict=True) == root and root.is_dir(), "Git source root changed")
    entry = root / ".git"
    if entry.is_dir() and not entry.is_symlink():
        git_directory = entry
    else:
        raw = file(entry)
        require(raw.startswith(b"gitdir: ") and raw.endswith(b"\n") and raw.count(b"\n") == 1,
            "unrecognized Git worktree pointer")
        target = Path(os.fsdecode(raw[8:-1]))
        git_directory = (entry.parent / target).resolve(strict=True)
    common_raw = file(git_directory / "commondir", optional=True)
    common_directory = git_directory if common_raw is None else (
        git_directory / os.fsdecode(common_raw).strip()).resolve(strict=True)
    for directory in {git_directory, common_directory}:
        value = directory.stat()
        require(directory.resolve(strict=True) == directory and directory.is_dir() and value.st_uid == 2618919,
            "Git configuration directory is not the pinned same-user source")
        files[str(directory)] = {"directory": True, "device": value.st_dev, "inode": value.st_ino,
            "uid": value.st_uid, "gid": value.st_gid, "mode": value.st_mode}
    file(common_directory / "config", optional=True)
    file(git_directory / "config.worktree", optional=True)
    return files


def verify_repository_configuration(root, environment):
    """Refuse all effective filter/include configuration before Git can inspect working-tree contents."""
    before = repository_configuration_inputs(root)
    if root in _REPOSITORY_CONFIGURATION:
        require(before == _REPOSITORY_CONFIGURATION[root]["inputs"], "pinned repository configuration changed")
    command = ["git", "-C", root, *REPOSITORY_CONFIGURATION_QUERY]
    outcome = subprocess.run(command, env=environment, capture_output=True, timeout=5, check=False)
    require(len(outcome.stdout) <= LIMIT and len(outcome.stderr) <= LIMIT,
        "repository configuration query exceeded its bound")
    # Git's native no-match return is one with no bytes or errors. Any configured filter/include is refused.
    require(outcome.returncode == 1 and outcome.stdout == b"" and outcome.stderr == b"",
        "repository filters, includes or unverified configuration query forbid source inspection: " + json.dumps({
            "returncode": outcome.returncode, "stdout_sha256": digest(outcome.stdout),
            "stderr_sha256": digest(outcome.stderr), "nonempty_key_count": sum(bool(key) for key in outcome.stdout.split(b"\0"))},
            sort_keys=True))
    after = repository_configuration_inputs(root)
    require(after == before, "repository configuration changed during its read-only query")
    row = {"inputs": before, "query": command, "returncode": outcome.returncode,
        "stdout_sha256": digest(outcome.stdout), "stderr_sha256": digest(outcome.stderr), "matched_key_names": [],
        "configured_filters_or_includes": False}
    _REPOSITORY_CONFIGURATION[root] = row
    return row


def finish_repository_configurations():
    """Rehash every used repository configuration before a successful isolated validation proof."""
    for root in tuple(_REPOSITORY_CONFIGURATION):
        verify_repository_configuration(root, metadata_environment(os.environ))
    return {"kind": "same_user_pinned_repository_configuration_and_mutation_detection",
        "repositories": dict(_REPOSITORY_CONFIGURATION), "filters_and_includes_refused": True,
        "scope": "Pinned same-user source/configuration trust and mutation detection; no kernel-enforced file immutability."}


def validate_repository_configuration_proof(value):
    """Require the isolated child's explicit nonempty repository/configuration trust and mutation proof."""
    require(type(value) is dict and set(value) == {"kind", "repositories", "filters_and_includes_refused", "scope"}
        and value["kind"] == "same_user_pinned_repository_configuration_and_mutation_detection"
        and type(value["repositories"]) is dict and bool(value["repositories"])
        and value["filters_and_includes_refused"] is True
        and value["scope"] == "Pinned same-user source/configuration trust and mutation detection; no kernel-enforced file immutability.",
        "isolated validation omitted its exact pinned repository configuration proof")
    return value


def metadata_event_guard(event, values, roots):
    """Refuse numerical imports, device/library/network access and unlisted Git roots."""
    if event == "import" and values[0].split(".")[0] in NUMERICAL:
        raise RuntimeError("numerical import forbidden in direct correctness child")
    if event in {"socket.connect", "socket.bind", "socket.getaddrinfo", "os.system", "ctypes.dlopen", "ctypes.dlsym"}:
        raise RuntimeError("network or library operation forbidden in direct correctness child")
    if event == "open" and isinstance(values[0], (str, bytes)) and (os.fsdecode(values[0]).startswith(("/dev/nvidia", "/dev/dri"))
            or os.fsdecode(values[0]) == "/tmp/sl5nw_serval02_card1.lock"):
        raise RuntimeError("device access forbidden in direct correctness child")
    if event == "subprocess.Popen":
        require(len(values) == 4 and type(values[0]) is str, "subprocess needs its actual executable and environment")
        command = values[1]
        require(type(command) is list and all(type(value) is str for value in command) and values[2] is None,
            "source-inspection subprocesses require plain arguments and no working-directory override")
        historical = Path(__file__).with_name(HISTORICAL_NAME)
        expected = ["/usr/bin/python3", "-I", "-S", "-B", str(historical), "--read-only-stdin"]
        if command == expected:
            require(values[0] == expected[0] and _ACTIVE_HISTORICAL_SOURCE_SHA256 is not None
                and digest(historical.read_bytes()) == _ACTIVE_HISTORICAL_SOURCE_SHA256,
                "only the exact active read-only historical child may start")
            require_metadata_environment(values[3])
            return
        require(values[0] == "git", "the actual executable must match the declared source-inspection Git")
        source_inspection_git(command, roots)
        require_metadata_environment(values[3])
        if command[3:] != REPOSITORY_CONFIGURATION_QUERY:
            verify_repository_configuration(command[2], metadata_environment(os.environ) if values[3] is None else values[3])


def stdlib_ctypes_bootstrap():
    """Record stdlib's own-process Python API setup before all later load/symbol calls are refused."""
    source = Path(ctypes.__file__).resolve(strict=True)
    expected = Path(sysconfig.get_path("stdlib")).resolve(strict=True) / "ctypes/__init__.py"
    require(source == expected and ctypes.pythonapi._name is None
        and type(ctypes.pythonapi._handle) is int and ctypes.pythonapi._handle > 0,
        "only the actual stdlib ctypes own-process Python API bootstrap is permitted")
    return {"module_path": str(source), "python_api_target": None,
        "own_process_python_api_initialized_before_validation_audit": True,
        "bootstrap_scope": "stdlib ctypes own-process Python API handle only",
        "later_library_loads_and_symbol_calls_refused": True}


def validate_bootstrap_record(value):
    """Require the explicit bootstrap scope in both actual child and saved proof metadata."""
    require(type(value) is dict and set(value) == {"module_path", "python_api_target",
        "own_process_python_api_initialized_before_validation_audit", "bootstrap_scope",
        "later_library_loads_and_symbol_calls_refused"}
        and type(value["module_path"]) is str and Path(value["module_path"]).is_absolute()
        and Path(value["module_path"]).parts[-2:] == ("ctypes", "__init__.py")
        and value["python_api_target"] is None
        and value["own_process_python_api_initialized_before_validation_audit"] is True
        and value["bootstrap_scope"] == "stdlib ctypes own-process Python API handle only"
        and value["later_library_loads_and_symbol_calls_refused"] is True,
        "the actual metadata child omitted or changed its restricted stdlib bootstrap record")
    return value


def main():
    """Run only bounded isolated metadata validation, with no device or scheduler operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-stdin", action="store_true", required=True)
    parser.parse_args()
    environment = metadata_environment(os.environ)
    os.environ.clear()
    os.environ.update(environment)
    bootstrap = stdlib_ctypes_bootstrap()
    validate_bootstrap_record(bootstrap)
    roots = set()
    sys.addaudithook(lambda event, values: metadata_event_guard(event, values, roots))
    require(not NUMERICAL.intersection(sys.modules), "numerical module was already loaded")
    raw = sys.stdin.buffer.read(LIMIT + 1)
    require(len(raw) <= LIMIT, "direct common confirmation plan exceeds its metadata bound")
    plan = json.loads(raw)
    context = metadata_compute_context(plan)
    if "accepted_evidence_snapshot_protocol" not in plan:
        roots.update(declared_git_roots(plan))
    sys.path.insert(0, str(Path(__file__).absolute().parent))
    proof = validate_actual(plan, digest(raw))
    proof["repository_configuration_validation"] = ({
        "kind": "reviewed_historical_cpu_snapshot_no_repository_commands", "repository_commands_executed": 0,
        "current_live_source_validation_performed_elsewhere": True} if "accepted_evidence_snapshot_protocol" in plan
        else finish_repository_configurations())
    proof["metadata_compute_context"] = context
    proof["stdlib_ctypes_bootstrap"] = bootstrap
    print(json.dumps(proof, allow_nan=False))

if __name__ == "__main__":
    main()
