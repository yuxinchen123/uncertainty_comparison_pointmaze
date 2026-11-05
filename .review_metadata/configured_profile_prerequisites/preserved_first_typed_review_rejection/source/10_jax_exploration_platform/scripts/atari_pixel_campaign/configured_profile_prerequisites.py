"""Bind reviewed historical profile prerequisites without replaying old modules.

This metadata path prepares an explicit A4500 profile. It grants no execution,
physical-device, numerical-source, worker, packing or scientific admission.
Each historical consumer remains in its own original staged namespace.
"""

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import time


NAME = "configured_profile_prerequisites.py"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
SOURCE = "d28e6735e79788f618acb0ebfde8609db1f776af"
SNAPSHOT = "88791d3f7e602523c547650d41bb2147d101a3793db0d969d34872ded8388a0e"
SNAPSHOT_REVIEW = "ac50aa8d71e4e1fd7665161486cae76f7b9fa38e045af52559182037a81733db"
PROTOCOL = {"schema_version": 1, "kind": "reviewed_four_context_configured_profile_prerequisites",
    "historical_consumers_replayed_in_current_namespace": False, "operational_admission": False}
CONTEXTS = {(hardware, copies) for hardware in ("h100", "jaguar03_rtx_a4500") for copies in (1, 2)}
HISTORICAL_HELPERS = {"jaguar03_rtx_a4500": ("9fcc43a478b93d731df780805fd7c3c514cdf67f", 21),
    "h100": ("69f1b5075530418adbe02c1e433c5647848a61ba", 29)}
QUALIFICATION_REVIEW = "004fcae1c5f24dec23da5add7452486c2ed5c1fcec89bf853617791a36fee74a"
QUALIFICATION_RESULT = "0c4f94c2d58fdd7277a261481d7a52c4bb5b19e42612a8eae67dff6890c62cd1"
NATIVE_REVIEW = "4264288c209606dccc15c883724ffeccaf5a480c943c5e29dd9b1c409c814da5"
MAXIMUM_METADATA_BYTES = 16 * 1024 * 1024


def require(condition, message):
    """Refuse incomplete metadata rather than infer an approval or physical identity."""
    if not condition:
        raise ValueError(message)


def encoded(value):
    """Keep Boolean values distinct from integers in all configuration joins."""
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def equal(first, second):
    """Compare complete typed metadata without Python's Boolean/integer equivalence."""
    return encoded(first) == encoded(second)


def reference(value):
    """Require one canonical absolute ordinary-file reference with a full content hash."""
    require(type(value) is dict and set(value) == {"path", "sha256"}
        and type(value["path"]) is str and Path(value["path"]).is_absolute()
        and str(Path(value["path"])) == value["path"] and ".." not in Path(value["path"]).parts
        and type(value["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
        "profile prerequisites need exact path and SHA-256 references")
    return value


def file_identity(value):
    """Exclude access timestamps while retaining every content/path identity field."""
    return tuple(getattr(value, name) for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns",
        "st_ctime_ns", "st_mode", "st_uid", "st_gid", "st_nlink"))


class MetadataFiles:
    """Read only bounded owned metadata under one total five-minute deadline."""

    def __init__(self):
        """Start one nonrenewable deadline for this complete validation."""
        self.started = time.monotonic()
        self.records = {}

    def raw(self, value):
        """Hash stable ordinary-file bytes; never open historical binary payloads."""
        reference(value)
        require(time.monotonic() - self.started < 285, "profile prerequisite validation exceeded its bounded metadata window")
        path = Path(value["path"])
        before = path.lstat()
        require(path.resolve(strict=True) == path and stat.S_ISREG(before.st_mode)
            and before.st_uid == os.getuid() and before.st_size <= MAXIMUM_METADATA_BYTES,
            "profile prerequisites must be bounded ordinary campaign-owned metadata")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            identity = os.fstat(descriptor)
            require(file_identity(identity) == file_identity(before), "profile prerequisite pathname changed before reading")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                raw = stream.read(MAXIMUM_METADATA_BYTES + 1)
            require(len(raw) == before.st_size and file_identity(os.fstat(descriptor)) == file_identity(before)
                and file_identity(path.stat(follow_symlinks=False)) == file_identity(before)
                and hashlib.sha256(raw).hexdigest() == value["sha256"],
                "profile prerequisite bytes or pathname changed")
            self.records[value["path"]] = value["sha256"]
            return raw
        finally:
            os.close(descriptor)

    def document(self, value):
        """Reject duplicate keys and nonfinite numbers in every supplied JSON record."""
        def unique(pairs):
            """Keep one explicit value for each metadata key."""
            result = {}
            for key, item in pairs:
                require(key not in result, "duplicate profile prerequisite JSON key")
                result[key] = item
            return result
        return json.loads(self.raw(value), object_pairs_hook=unique,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError("nonfinite prerequisite JSON")))


def without_copies(configuration):
    """Compare only packing counts while preserving every other declared setting."""
    value = copy.deepcopy(configuration)
    require(set(value) == {"learner", "environment", "execution"}
        and type(value["learner"].get("n_copies")) is int and value["learner"]["n_copies"] in (1, 2),
        "the reviewed common baseline requires one or two independent copies")
    # Before: learner contains n_copies=2; after: only that count is omitted.
    value["learner"].pop("n_copies")
    return value


def source_closure(files, plan, unit):
    """Keep the original helper namespace's staged and source-root bytes exact."""
    receipt = plan["helpers"]
    expected_commit, count = HISTORICAL_HELPERS[plan["hardware_class"]]
    require(receipt["source_commit"] == expected_commit and len(receipt["files"]) == count
        and Path(receipt["source_root"]).is_absolute(),
        "the accepted historical plan lacks its exact original helper source")
    names = []
    for relative, expected in receipt["files"].items():
        require(relative.startswith(PREFIX) and Path(relative).suffix == ".py"
            and relative == PREFIX + Path(relative).name and re.fullmatch(r"[0-9a-f]{64}", expected),
            "the original helper closure contains another namespace or an incomplete hash")
        files.raw({"path": str(unit / "code" / Path(relative).name), "sha256": expected})
        files.raw({"path": str(Path(receipt["source_root"]) / relative), "sha256": expected})
        names.append(relative)
    # One read-only Git child binds the same text to its actual historical commit.
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    query = "".join(receipt["source_commit"] + ":" + name + "\n" for name in names).encode()
    completed = subprocess.run(["git", "-C", receipt["source_root"], "cat-file", "--batch"],
        input=query, capture_output=True, check=True, timeout=min(15, max(.001, 285 - (time.monotonic() - files.started))),
        env=environment)
    raw, offset = completed.stdout, 0
    require(len(raw) <= MAXIMUM_METADATA_BYTES and completed.stderr == b"", "historical Git source response is incomplete or oversized")
    for name in names:
        end = raw.find(b"\n", offset)
        require(end >= offset, "historical Git source lacks its object header")
        fields = raw[offset:end].split()
        require(len(fields) == 3 and fields[1] == b"blob" and fields[2].isdigit(), "historical helper is not an actual Git blob")
        size = int(fields[2]); offset = end + 1
        require(hashlib.sha256(raw[offset:offset + size]).hexdigest() == receipt["files"][name]
            and raw[offset + size:offset + size + 1] == b"\n", "the historical staged helper differs from its exact Git object")
        offset += size + 1
    require(offset == len(raw), "historical Git source response contains extra objects")
    return receipt


def confirmation_metadata(proof_path, review_path, hardware, copies):
    """Read the selected reviewed plan as metadata without reopening its large raw result."""
    def actual(path):
        """Describe only an actual existing input, never a future approval."""
        path = Path(path).resolve(strict=True)
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    files = MetadataFiles()
    proof_ref = actual(proof_path)
    proof, review = files.document(proof_ref), files.document(actual(review_path))
    require(review.get("proof") == proof_ref and review.get("approved") is True and review.get("passed") is True
        and type(copies) is int and (hardware, copies) in CONTEXTS,
        "profile preparation requires the actual reviewed prerequisite record")
    entries = [row for row in proof["contexts"] if row.get("hardware_class") == hardware and row.get("copies") == copies]
    require(len(entries) == 1, "the reviewed profile context is missing or repeated")
    accepted = files.document(entries[0]["review"])
    old_plan = files.document(accepted["plan"])
    # This is a metadata summary of the accepted plan, not a rewritten result.
    return {"candidate": old_plan["configuration"], "measurement_mode": "confirmation", "decision": "accepted",
        "configured_isolation": old_plan["configured_isolation"]}, reference(accepted["result"])


def reviewed_context(files, entry, proposed):
    """Join one genuine accepted raw review to its exact original plan and context."""
    require(type(entry) is dict and set(entry) == {"hardware_class", "copies", "review"}
        and type(entry["copies"]) is int and (entry["hardware_class"], entry["copies"]) in CONTEXTS,
        "each common profile prerequisite needs one distinct hardware/count review")
    review = files.document(entry["review"])
    hardware, copies = entry["hardware_class"], entry["copies"]
    require(type(review.get("schema_version")) is int and review["schema_version"] == 1
        and review.get("approved") is True and review.get("passed") is True
        and review.get("source_commit") == SOURCE and type(review.get("copies")) is int
        and review["copies"] == copies and review.get("copy_seeds") == list(range(1, copies + 1))
        and review.get("admission_granted") is False and review.get("worker_admitted") is False,
        "a complete independently accepted common performance review is required")
    plan = files.document(review["plan"])
    unit = Path(reference(review["result"])["path"]).parent
    require(review["plan"]["path"] == str(unit / "measurement_plan.json")
        and unit == Path(proposed["campaign_run_dir"]) / "data" / plan["unit_id"]
        and plan.get("campaign_run_dir") == proposed["campaign_run_dir"]
        and plan.get("hardware_class") == hardware and plan.get("source_commit") == SOURCE
        and plan.get("mode") == "confirmation" and plan.get("copy_seeds") == list(range(1, copies + 1))
        and all(type(plan.get(key)) is int and plan[key] == expected for key, expected in (
            ("rounds", 7), ("iterations", 10), ("minimum_updates_per_round", 10)))
        and type(plan["configuration"]["learner"]["n_copies"]) is int
        and plan["configuration"]["learner"]["n_copies"] == copies,
        "the common review changed its exact source, unit, context, seeds or complete update counts")
    require(equal(without_copies(plan["configuration"]), without_copies(proposed["configuration"]))
        and equal(plan["runtime_environment"], proposed["runtime_environment"])
        and equal(plan["asset_record"], proposed["asset_record"]),
        "all four accepted contexts must share the proposed full configuration, runtime and assets")
    # Keep the only two admitted changes explicit, including their Boolean types.
    candidate, original = plan["configuration"], plan["reference_configuration"]
    changes = [(scope, key, original[scope][key], value) for scope in candidate
        for key, value in candidate[scope].items() if not equal(value, original[scope][key])]
    require(sorted(changes) == [("environment", "cache_room_backgrounds", False, True),
        ("environment", "render_final_two_only", False, True)]
        and all(type(configuration["environment"][key]) is bool for configuration in (candidate, original)
            for key in ("cache_room_backgrounds", "render_final_two_only")),
        "profile prerequisites cannot admit another candidate or reference configuration")
    receipt = source_closure(files, plan, unit)
    require(review.get("helper_commit") == receipt["source_commit"], "the accepted raw review changed its historical helper commit")
    if hardware == "jaguar03_rtx_a4500":
        require(review.get("kind") == "independent_actual_common_configuration_confirmation_review"
            and review.get("actual_terminal_state") == "COMPLETED" and review.get("actual_exit_code") == "0:0"
            and review.get("exact_staged_saved_consumer_passed") is True
            and review.get("current_intermediate_combination_reader_passed") is True
            and review.get("seven_alternating_pairs_ten_full_updates_per_arm") is True
            and review.get("actual_assets_verified") is True
            and review.get("loaded_CUDA_cuDNN_image_hashes_verified") is True
            and review.get("same_supplied_reference_state_origins") is True
            and review.get("packing_or_headroom_credit_granted") is False,
            "the A4500 common performance lacks its genuine original staged-consumer/raw approval")
        require(review.get("all_files_before") == review.get("all_files_after")
            and type(review.get("all_files_before")) is dict and review["all_files_before"],
            "the A4500 raw review did not preserve its complete historical file inventory")
        inventory = review["all_files_before"]
        require(all(inventory.get(value["path"], {}).get("sha256") == value["sha256"]
            for value in (review["plan"], review["result"])),
            "the A4500 raw review did not cover the exact accepted plan and result")
        combination = review["accepted_combination"]
        require(all(equal(combination.get(key), expected) for key, expected in (
            ("configuration", candidate), ("copies", copies), ("hardware_class", hardware),
            ("source_commit", SOURCE), ("runtime_environment", proposed["runtime_environment"]), ("paired_rounds", 7)))
            and type(combination.get("median_speedup")) in (int, float) and math.isfinite(combination["median_speedup"])
            and combination["median_speedup"] > 1
            and type(combination.get("one_sided_sign_probability")) in (int, float)
            and 0 < combination["one_sided_sign_probability"] <= .05,
            "the accepted A4500 combination differs from the reviewed complete paired performance")
    else:
        require(review.get("kind") == "independent_actual_direct_common_configuration_confirmation_review"
            and review.get("independent_of_producer") is True and review.get("decision") == "accepted"
            and review.get("paired_rounds") == 7 and review.get("updates_per_arm_per_round") == 10
            and review.get("raw_window_recomputation_passed") is True
            and type(review.get("normal_actual_child_returncode")) is int and review["normal_actual_child_returncode"] == 0
            and review.get("actual_controller_and_numerical_child_exits_verified") is True
            and review.get("unresolved_anomalies") == [],
            "the H100 common performance lacks its genuine complete raw/terminal approval")
        for name in ("raw_audit", "raw_first_approval", "exit_observation"):
            files.document(review[name])
        consumer = files.document(review["unchanged_staged_consumer"])
        require(consumer.get("kind") == "exact_staged_direct_common_timing_saved_consumer_result"
            and consumer.get("passed") is True and consumer.get("result") == review["result"]
            and consumer.get("plan") == review["plan"] and consumer.get("source_helper_commit") == receipt["source_commit"]
            and type(consumer.get("source_helper_count")) is int and consumer["source_helper_count"] == 29
            and consumer.get("raw_first_approval") == review["raw_first_approval"]
            and equal(review.get("configuration"), candidate) and equal(review.get("reference_configuration"), original),
            "the H100 confirmation must retain its actual exact staged-consumer receipt")
    return {"hardware_class": hardware, "copies": copies, "plan": review["plan"], "result": review["result"],
        "review": entry["review"], "helpers": receipt, "target_qualification_inputs": plan.get("target_qualification_inputs"),
        "common_correctness_inputs": plan["common_correctness_inputs"], "historical_plan": plan}


def initial_snapshot(files, inputs, historical_plan):
    """Consume the genuine snapshot using only its own exact private staged reader."""
    require(set(inputs) == {"proof", "review"} and inputs["proof"]["sha256"] == SNAPSHOT
        and inputs["review"]["sha256"] == SNAPSHOT_REVIEW
        and equal(inputs, historical_plan.get("accepted_evidence_snapshot_inputs")),
        "the existing complete original snapshot and independent review must remain exact")
    path = Path(historical_plan["campaign_run_dir"]) / "data" / historical_plan["unit_id"] / "code/accepted_evidence_snapshot.py"
    source = {"path": str(path), "sha256": historical_plan["helpers"]["files"][PREFIX + path.name]}
    raw = files.raw(source)
    # A private module name keeps this old pure reader out of every current helper namespace.
    specification = importlib.util.spec_from_file_location("configured_profile_historical_snapshot_" + source["sha256"], path)
    module = importlib.util.module_from_spec(specification)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    selected = module.consume(historical_plan)
    return {"inputs": inputs, "reader": source, "comparisons_previously_validated": 120,
        "comparisons_recomputed": 0, "historical_payloads_rehashed": False,
        "historical_registry": selected.files.json(selected.qual["historical_inputs"])["registry"]}


def validate(plan):
    """Require one reviewed complete four-context fact while leaving all operation gates closed."""
    require(equal(plan.get("configured_profile_prerequisite_protocol"), PROTOCOL)
        and plan.get("kind") == "selected_configuration_profile" and plan.get("hardware_class") == "jaguar03_rtx_a4500"
        and plan.get("source_commit") == SOURCE and plan.get("gpu_uuid") is None
        and "configured_isolation" in plan and "direct_binding" not in plan
        and "configured_native_process_control" in plan and "native_process_control" in plan,
        "these prerequisites prepare only an unresolved explicit A4500 profile")
    files = MetadataFiles()
    inputs = plan["configured_profile_prerequisite_inputs"]
    require(set(inputs) == {"proof", "review"}, "actual prerequisite proof and independent review are both required")
    proof, review = files.document(inputs["proof"]), files.document(inputs["review"])
    require(proof.get("kind") == PROTOCOL["kind"] and type(proof.get("schema_version")) is int and proof["schema_version"] == 1
        and proof.get("campaign_run_dir") == plan["campaign_run_dir"] and proof.get("source_commit") == SOURCE
        and proof.get("operational_admission") is False
        and review.get("kind") == "independent_reviewed_four_context_configured_profile_prerequisites"
        and review.get("proof") == inputs["proof"] and review.get("approved") is True and review.get("passed") is True
        and review.get("independent_of_producer") is True and review.get("all_four_raw_reviews_verified") is True
        and review.get("exact_historical_consumer_namespaces_verified") is True
        and review.get("admission_granted") is False and review.get("worker_admitted") is False
        and review.get("unresolved_anomalies") == [], "the complete four-context prerequisite proof needs its genuine independent approval")
    entries = proof["contexts"]
    require(type(entries) is list and len(entries) == 4
        and {(row["hardware_class"], row["copies"]) for row in entries} == CONTEXTS
        and all(type(row["copies"]) is int for row in entries), "all four distinct hardware/count contexts are required")
    native = plan["native_process_control"]["expected_target_contract"]
    require(equal(proof["native_process_qualification"], native)
        and native["independent_review"]["sha256"] == NATIVE_REVIEW,
        "profile preparation must bind the actual accepted process component without granting numerical admission")
    native_review = files.document(native["independent_review"])
    require(native_review.get("kind") == "independent_configured_native_process_review"
        and native_review.get("approved") is True and native_review.get("passed") is True
        and native_review.get("independent_of_producer") is True
        and native_review.get("result") == native["diagnostic_result"]
        and native_review.get("plan") == native["diagnostic_plan"]
        and native_review.get("target_contract") == native["target_contract"]
        and native_review.get("profile_admission_granted") is False,
        "the process component review is incomplete or differs from the prepared native contract")
    contexts = [reviewed_context(files, row, plan) for row in entries]
    selected = next(row for row in contexts if row["hardware_class"] == plan["hardware_class"]
        and row["copies"] == plan["configuration"]["learner"]["n_copies"])
    require(plan["inputs"]["accepted_confirmation"] == selected["result"]
        and plan["copy_seeds"] == list(range(1, selected["copies"] + 1)),
        "the proposed profile must use its exact accepted context and stable copy seeds")
    historical = next(row["historical_plan"] for row in contexts if row["hardware_class"] == "h100" and row["copies"] == 1)
    snapshot = initial_snapshot(files, proof["initial_snapshot_inputs"], historical)
    require(plan["inputs"]["study_registry"]["sha256"] == snapshot["historical_registry"]["sha256"],
        "the profile changed the initial registry covered by the genuine accepted snapshot")
    require(equal(proof["target_qualification_inputs"], selected["target_qualification_inputs"]),
        "the target qualification must be the exact full-original qualification used by accepted A4500 timing")
    qualification_review = files.document(proof["target_qualification_inputs"]["review"])
    require(proof["target_qualification_inputs"]["review"]["sha256"] == QUALIFICATION_REVIEW
        and proof["target_qualification_inputs"]["result"]["sha256"] == QUALIFICATION_RESULT
        and qualification_review.get("kind") == "independent_full_original_target_qualification_review"
        and qualification_review.get("approved") is True
        and qualification_review.get("result") == proof["target_qualification_inputs"]["result"]
        and qualification_review.get("actual_terminal_state") == "COMPLETED" and qualification_review.get("actual_exit_code") == "0:0"
        and qualification_review.get("admission_granted") is False,
        "the original A4500 qualification needs its exact genuine raw acceptance")
    return {"protocol": PROTOCOL, "inputs": inputs, "contexts": [{key: value for key, value in row.items()
        if key != "historical_plan"} for row in contexts], "initial_snapshot": snapshot,
        "target_qualification_inputs": proof["target_qualification_inputs"], "metadata_files": files.records,
        "scientific_credit": 0, "admission_granted": False, "worker_admitted": False,
        "GPU_numerical_qualification": False, "current_physical_identity_verified": False}


def configure_plan(plan, proof_path, review_path):
    """Freeze two existing prerequisite references before staging any profile artifacts."""
    require(proof_path is not None and review_path is not None, "both actual four-context prerequisite records are required")
    def actual(path):
        """Describe actual metadata bytes without inventing a future proof hash."""
        path = Path(path).resolve(strict=True)
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    plan["configured_profile_prerequisite_protocol"] = dict(PROTOCOL)
    plan["configured_profile_prerequisite_inputs"] = {"proof": actual(proof_path), "review": actual(review_path)}
    result = validate(plan)
    plan["target_qualification_inputs"] = result["target_qualification_inputs"]
    return result
