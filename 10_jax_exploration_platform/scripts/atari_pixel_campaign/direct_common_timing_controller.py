"""Start only one separately reviewed common timing unit after its actual predecessor exit.

The default reads local metadata and prints the intended command. It does not
contact serval02, acquire its standing lock, import a numerical runtime, or start
a child. Only the producer owns the GPU lock; this controller records and waits
for its actual child. This file is separate from the committed package-evidence producer.
"""

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time

PACKET = Path(__file__).absolute().parent
INTEGRATION = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl")
RUN = PACKET.parent.parent
UNIT = RUN / "data" / PACKET.name
HELPER_COMMIT = None  # The fresh request must bind the actual reviewed twenty-nine-helper commit.
SOURCE_COMMIT = "d28e6735e79788f618acb0ebfde8609db1f776af"
GPU_UUID = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
UID = 2618919
HOST = "serval02"
BACKEND_EXPECTED_SHA256 = "7d50f9ef96fbe5afa9e892905b0e3e76dde43e9f54835630c69aabf7489d35e4"
BACKEND_PREPARATION_RECORD = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-20-46_direct-backend-package-image-preparation_python-atari-pixels-jax_unique-record-owner_gpu-count-0/direct_backend_image_expected.json")
NUMERICAL_MODULES = {"jax", "jaxlib", "numpy", "torch", "cupy", "flax", "optax", "scipy", "jaxtari", "exploration_platform"}
ATTEMPT_NAMES = (
    "direct_controller_intent.json", "direct_controller_spawn.json",
    "direct_controller_failure.json", "direct_controller_outcome.json", "direct_controller_unidentified_wait.json",
    "direct_controller_stdout.raw", "direct_controller_stderr.raw",
    "direct_controller_output_hashes.json",
    "direct_controller_timeout_observation.json",
    "direct_current_launch.json", "direct_current_launch.json.prepared",
    "direct_current_process_ledger_snapshot.json",
    "direct_process_wait_return.json", "direct_terminal_process.json",
    "direct_process_metadata.json", "direct_device_binding.json",
    "direct_device_binding_attempt.json", "direct_qualification_result.json", "result.json",
    "failure.json", "direct_startup_checks.json", "captured_driver",
    "direct_backend_capture_attempt.json", "direct_backend_capture_begin.json",
    "direct_backend_capture_failure.json", "direct_backend_capture.json", "direct_backend",
)
BOUNDARY = {"scientific_credit": 0, "case_credit": 0, "idea_credit": 0,
            "worker_admitted": False, "admission_granted": False,
            "controller_owns_gpu_standing_descriptor": False,
            "producer_owns_gpu_standing_descriptor": True}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ordinary_bytes(path, bound=16 * 1024 * 1024):
    """Pin every ordinary directory and the file before opening it for reading."""
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "an exact absolute ordinary path is required")
    descriptor = os.open("/", os.O_PATH | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            following = os.open(part, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        pinned = os.open(path.name, os.O_PATH | os.O_NOFOLLOW, dir_fd=descriptor)
        try:
            before = os.fstat(pinned)
            require(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= bound,
                    "an ordinary bounded regular file is required: " + str(path))
            reader = os.open("/proc/self/fd/" + str(pinned), os.O_RDONLY)
            try:
                raw = b""
                while len(raw) <= bound:
                    block = os.read(reader, min(65536, bound + 1 - len(raw)))
                    if not block:
                        break
                    raw += block
                require(len(raw) <= bound and stable_stat(os.fstat(reader)) == stable_stat(before), "file changed during its bounded read")
            finally:
                os.close(reader)
            require(stable_stat(os.stat(path, follow_symlinks=False)) == stable_stat(before), "ordinary input changed its inode or metadata")
            return raw
        finally:
            os.close(pinned)
    finally:
        os.close(descriptor)


def document(path):
    return json.loads(ordinary_bytes(path))


def stable_stat(info):
    return tuple(getattr(info, name) for name in ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_size", "st_mtime_ns", "st_ctime_ns"))


def receipt(path):
    return {"path": str(path), "sha256": digest(ordinary_bytes(path))}


def log_receipt(path):
    """Hash an actual output file in chunks without loading the complete log."""
    path = Path(path)
    require(path.parent.resolve(strict=True) == path.parent, "the actual output parent changed")
    pinned = os.open(path, os.O_PATH | os.O_NOFOLLOW)
    try:
        before = os.fstat(pinned)
        require(stat.S_ISREG(before.st_mode), "actual output is not an ordinary regular file")
        reader = os.open("/proc/self/fd/" + str(pinned), os.O_RDONLY)
        try:
            value = hashlib.sha256()
            count = 0
            while block := os.read(reader, 65536):
                value.update(block)
                count += len(block)
            require(count == before.st_size and stable_stat(os.fstat(reader)) == stable_stat(before)
                    and stable_stat(os.stat(path, follow_symlinks=False)) == stable_stat(before),
                    "actual output changed after the exact child wait")
        finally:
            os.close(reader)
        return {"path": str(path), "sha256": value.hexdigest(), "size_bytes": count}
    finally:
        os.close(pinned)


def publish(path, value):
    """Publish complete durable bytes with no replacement, retaining a partial on failure."""
    path = Path(path)
    require(path.parent.resolve(strict=True) == path.parent, "publication directory must be ordinary")
    temporary = path.with_name(path.name + ".prepared")
    raw = value if isinstance(value, bytes) else encoded(value)
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path, follow_symlinks=False)
    fsync_directory(path.parent)
    temporary.unlink()
    fsync_directory(path.parent)
    return receipt(path)


@contextmanager
def metadata_lock(path, expected_uid):
    """Use the campaign's existing metadata/process lock names, with ordinary paths."""
    path = Path(path)
    require(path.parent.resolve(strict=True) == path.parent, "metadata-lock parent changed")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode) and info.st_uid == expected_uid
                and not info.st_mode & 0o022, "metadata lock is not an ordinary owned protected file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def no_numerical_modules():
    require(not (NUMERICAL_MODULES & {name.split(".", 1)[0] for name in sys.modules}),
            "controller must remain standard-library metadata only")


def check_backend_inputs(plan):
    """Require the unchanged qualified package record for this new twenty-nine-helper common timing plan."""
    relative = "10_jax_exploration_platform/scripts/atari_pixel_campaign/direct_backend_image.py"
    request = document(PACKET / "preparation_request.json")
    backend = plan.get("direct_backend_image")
    require(type(backend) is dict and set(backend) == {"schema_version", "capability", "expected"}
        and type(backend["schema_version"]) is int and backend["schema_version"] == 2
        and backend["capability"] == "prepared_owning_package_image_and_same_process_mapping_version_2",
        "every new direct timing unit requires prepared backend package schema two")
    require(len(plan["helpers"]["files"]) == 29 and relative in plan["helpers"]["files"]
        and plan.get("mode") == "confirmation"
        and plan.get("direct_qualification_inputs") == request["direct_qualification_inputs"]
        and not any(key in plan for key in ("direct_qualification", "target_qualification", "completed_study",
            "final_confirmation_protocol", "device_memory_limit_protocol", "native_process_control")),
        "timing requires the reviewed snapshot namespace and actual qualification inputs")
    expected = UNIT / "direct_backend_image_expected.json"
    require(backend["expected"] == {"path": str(expected), "sha256": BACKEND_EXPECTED_SHA256}
        and receipt(expected) == backend["expected"]
        and plan["inputs"]["direct_backend_image_record"] == {"path": str(BACKEND_PREPARATION_RECORD), "sha256": BACKEND_EXPECTED_SHA256}
        and receipt(BACKEND_PREPARATION_RECORD) == plan["inputs"]["direct_backend_image_record"],
        "the copied or original actual prepared package record changed")
    actual = document(expected)
    require(actual == document(BACKEND_PREPARATION_RECORD) and type(actual["schema_version"]) is int
        and actual["schema_version"] == 2 and actual["kind"] == "prepared_direct_backend_package_image"
        and actual["capability"] == backend["capability"] and actual["selected_python"] == plan["python"]
        and actual["image"]["sha256"] == "ced7988972611b25fbdc7df51453102656c209c05756b528602aa875c10d1f08"
        and type(actual["image"]["size_bytes"]) is int and actual["image"]["size_bytes"] == 473181472
        and actual["numerical_imports"] == [] and type(actual["gpu_operations"]) is int and actual["gpu_operations"] == 0,
        "the actual prepared package image or metadata-only observation changed")
    require(actual["producer_source_sha256"] == plan["helpers"]["files"][relative]
        == digest(ordinary_bytes(UNIT / "code/direct_backend_image.py")),
        "the prepared package record does not use the exact staged producer")
    qualified = document(request["qualification_plan"]["path"])
    versions = qualified["qualification_software"]["packages"]
    normalize = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    normalized = {normalize(name): version for name, version in versions.items()}
    require(len(normalized) == len(versions)
        and normalized.get(normalize(actual["distribution"]["name"])) == actual["distribution"]["version"],
        "the uniquely discovered owner/version differs from the actual accepted original qualification")
    return backend


def packet_metadata_paths(plan):
    """Enumerate the fresh common-timing packet, twenty-nine helpers and reviewed history pointer."""
    fixed = ('preparation_request.json', 'metadata_preparation_result.json',
        'shared_software_assets_staged_verification.json', 'cold_metadata_validation.json',
        'reference_configuration.json', 'candidate_configuration.json', 'runtime_environment.json',
        'asset_record.json', 'direct_binding_profile.json', 'serial_predecessor_exit_inputs.json',
        'manifest.yaml', 'config_resolved.yaml', 'experiment_background.md', 'command.txt',
        'prepare_common_timing.py', 'canonical_controller_source_mapping.json')
    files = [PACKET / name for name in fixed]
    files += [UNIT / name for name in ('measurement_plan.json', 'measurement_prepared.json',
        'manifest.yaml', 'config_resolved.yaml', 'experiment_background.md', 'command.txt',
        'direct_backend_image_expected.json', 'direct_history/inputs.json')]
    names = [Path(path).name for path in plan['helpers']['files']]
    require(len(names) == len(set(names)) == 29, 'only the complete distinct twenty-nine-helper timing closure is admitted')
    files += [UNIT / 'code' / name for name in names]
    return sorted(set(files))


def validate_packet_freeze(plan, frozen):
    """Refuse a reduced metadata/code roster before hashing any of its supplied entries."""
    require(type(frozen.get("schema_version")) is int and frozen["schema_version"] == 1
        and frozen.get("kind") == "actual_direct_common_timing_packet_metadata_freeze"
        and type(frozen.get("files")) is dict, "the packet freeze schema or kind changed")
    expected = {str(path) for path in packet_metadata_paths(plan)}
    require(set(frozen["files"]) == expected, "the frozen packet omitted or added a required metadata/code path")
    for name, checksum in frozen["files"].items():
        require(Path(name).is_relative_to(RUN) and digest(ordinary_bytes(name)) == checksum,
            "an actual required frozen packet input changed: " + name)
    return frozen


def checked_inputs():
    """Validate only frozen current metadata; absent actual preparation or serial inputs must refuse."""
    no_numerical_modules()
    require(PACKET.resolve(strict=True) == PACKET and UNIT.resolve(strict=True) == UNIT,
        "the controller must use its exact ordinary packet and data paths")
    binding = document(PACKET / "controller_bound_metadata.json")
    require(type(binding.get("schema_version")) is int and binding["schema_version"] == 1
        and binding.get("kind") == "actual_prepared_direct_common_timing_controller_inputs"
        and set(binding) == {"schema_version", "kind", "plan", "preparation", "software", "packet_freeze", "request", "serial_predecessor_exit"},
        "only actual complete controller input bindings are supported")
    paths = {"plan": UNIT / "measurement_plan.json", "preparation": PACKET / "metadata_preparation_result.json",
        "software": PACKET / "shared_software_assets_staged_verification.json",
        "packet_freeze": PACKET / "packet_metadata_freeze.json", "request": PACKET / "preparation_request.json",
        "serial_predecessor_exit": PACKET / "serial_predecessor_exit_inputs.json"}
    for name, path in paths.items():
        require(binding[name] == receipt(path), "the bound actual controller input changed: " + name)
    request = document(paths["request"]); frozen = document(paths["packet_freeze"])
    require(request.get("kind") == "direct_common_timing_preparation_request" and type(request.get("schema_version")) is int
        and request["schema_version"] == 1, "old correctness requests cannot admit common timing")
    require(request["packet"] == str(PACKET) and request["data_unit"] == str(UNIT)
        and request["unit_id"] == UNIT.name and request["campaign_run_dir"] == str(RUN),
        "the copied request must name its actual packet and data unit")
    validate_packet_freeze(document(paths["plan"]), frozen)
    prepared, software = document(paths["preparation"]), document(paths["software"])
    require(prepared.get("passed") is True and software.get("passed") is True
        and prepared.get("plan") == binding["plan"] and software.get("plan") == binding["plan"]
        and prepared.get("kind") == "direct_common_timing_metadata_preparation_result"
        and software.get("kind") == "actual_direct_common_timing_shared_metadata_and_staged_verification",
        "the actual reviewed metadata preparation and current software proof must pass for this plan")
    plan = document(paths["plan"])
    require(plan["campaign_run_dir"] == str(RUN) and plan["unit_id"] == UNIT.name
        and plan["helpers"]["source_commit"] == request["helper_source_commit"] and plan["helpers"]["source_root"] == request["helper_source_root"]
        and plan["source_commit"] == SOURCE_COMMIT and plan["source_root"] == request["numerical_source_root"]
        and plan["python"] == request["python"] and plan["asset_record"] == request["asset_record"]
        and plan["gpu_uuid"] == GPU_UUID
        and plan["hardware_class"] == "h100" and plan["direct_binding"]["consumer"] == "selected_measurement"
        and plan["runtime_environment"] == {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
            "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"}
        and plan["configuration"] == request["configuration"] and plan["reference_configuration"] == request["reference_configuration"]
        and plan["copy_seeds"] == request["copy_seeds"] and plan["rounds"] == 7 and plan["minimum_updates_per_round"] == 10,
        "the actual original configuration, two candidate flags, seeds, source or runtime changed")
    qualified = document(request["qualification_plan"]["path"])
    timing_snapshot(plan, request)
    reference = json.loads(json.dumps(qualified["configuration"]))
    copies = plan["configuration"]["learner"]["n_copies"]
    require(type(copies) is int and copies in (1, 2), "only the approved one/two-copy common timing units are supported")
    reference["learner"]["n_copies"] = copies
    candidate = json.loads(json.dumps(reference))
    candidate["environment"]["cache_room_backgrounds"] = True
    candidate["environment"]["render_final_two_only"] = True
    require(reference == plan["reference_configuration"] and candidate == plan["configuration"]
        and plan["copy_seeds"] == list(range(1, copies + 1)), "only the two supplied environment flags may change")
    for value in [request["qualification_plan"], request["qualification_terminal"],
            request["qualification_controller_outcome"], request["qualification_consumer_verification"],
            *request["direct_qualification_inputs"].values()]:
        require(receipt(value["path"]) == value, "an exact actual accepted qualification input changed")
    check_backend_inputs(plan)
    check_serial_predecessor_inputs(request, binding["serial_predecessor_exit"])
    command = [plan["python"], str(UNIT / "code/measure_selected.py"), "--prepared-unit", str(UNIT)]
    require(shlex.split(ordinary_bytes(UNIT / "command.txt").decode()) == command,
        "the intended Popen command differs from the frozen common timing command")
    return plan, command, frozen



def check_serial_predecessor_inputs(request, input_reference):
    """Require exact saved prior normal waits and a real owner-checked exit observation."""
    timing_copy_order(request)
    require(input_reference == receipt(PACKET / "serial_predecessor_exit_inputs.json"),
        "the actual serial predecessor binding changed")
    inputs = document(input_reference["path"])
    names = {"schema_version", "kind", "predecessor_unit", "plan", "result", "terminal", "wait",
        "outcome", "metadata", "controller_spawn", "exit_observation"}
    require(set(inputs) == names and type(inputs["schema_version"]) is int and inputs["schema_version"] == 1
        and inputs["kind"] == "actual_owned_direct_serial_predecessor_exit_inputs"
        and inputs["predecessor_unit"] == request["serial_predecessor_unit"],
        "only this unit's actual predecessor exit evidence is allowed")
    previous = Path(inputs["predecessor_unit"])
    previous_packet = RUN / "remote_setup" / previous.name
    require(previous.parent == RUN / "data" and previous.resolve(strict=True) == previous
        and previous_packet.resolve(strict=True) == previous_packet, "the prior unit/packet must be exact ordinary campaign paths")
    expected_paths = {"plan": previous / "measurement_plan.json", "terminal": previous / "direct_terminal_process.json",
        "wait": previous / "direct_process_wait_return.json", "outcome": previous / "direct_controller_outcome.json",
        "metadata": previous / "direct_process_metadata.json", "controller_spawn": previous_packet / "root_actual_controller_spawn.json"}
    for name, path in expected_paths.items():
        require(inputs[name] == receipt(path), "an actual prior process record changed: " + name)
    for name in ("result", "exit_observation"):
        require(receipt(inputs[name]["path"]) == inputs[name], "an actual prior result or exit observation changed")
    prior_plan, prior_result = document(inputs["plan"]["path"]), document(inputs["result"]["path"])
    metadata = document(inputs["metadata"]["path"])["after"]["identity"]
    terminal, wait = document(inputs["terminal"]["path"]), document(inputs["wait"]["path"])
    outcome, spawn = document(inputs["outcome"]["path"]), document(inputs["controller_spawn"]["path"])
    require(prior_plan["unit_id"] == previous.name and prior_plan["campaign_run_dir"] == str(RUN)
        and prior_plan["source_commit"] == SOURCE_COMMIT and prior_plan["helpers"]["source_commit"] == ("0a880984b4a03a2ff7335337cc18576628bc9fc1" if request["serial_order"] == 1 else request["helper_source_commit"])
        and prior_plan["source_root"] == request["numerical_source_root"]
        and (request["serial_order"] == 1 or prior_plan["helpers"]["source_root"] == request["helper_source_root"])
        and prior_plan["python"] == request["python"]
        and prior_plan["runtime_environment"] == request["runtime_environment"]
        and prior_plan["asset_record"] == request["asset_record"]
        and prior_plan["gpu_uuid"] == GPU_UUID and metadata["hostname"] == HOST and metadata["uid"] == UID,
        "the prior process must be the exact owned direct source and target")
    require(type(spawn.get("actual_pid")) is int and spawn["actual_pid"] > 0
        and re.fullmatch(r"[1-9][0-9]*", spawn["actual_birth_counter"])
        and spawn["kind"] == "actual_controller_transport_popen_return"
        and spawn["host"] == HOST and spawn["uid"] == UID and spawn["automatic_retry_permitted"] is False
        and metadata["parent_pid"] == spawn["actual_pid"], "the actual prior controller or numerical parent changed")
    identity = {key: metadata[key] for key in ("hostname", "pid", "process_start_time")}
    command = [prior_plan["python"], str(previous / "code/measure_selected.py"), "--prepared-unit", str(previous)]
    require(wait.get("kind") == "owned_direct_process_wait_return" and type(wait.get("schema_version")) is int and wait["schema_version"] == 1
        and all(wait.get(key) == value for key, value in identity.items())
        and wait.get("uid") == UID and wait.get("waited_pid") == metadata["pid"] and wait.get("command") == command
        and wait.get("wait_semantics") == "Popen.wait on this exact spawned child"
        and type(wait.get("returncode")) is int and wait["returncode"] == 0
        and wait.get("exited_normally") is True and wait.get("timeouts") == [],
        "the predecessor did not finish with an exact uninterrupted successful child wait")
    for key in ("waiter_before", "waiter_after"):
        fields = wait[key]["stat"].rsplit(")", 1)[1].split()
        require(wait[key]["pid"] == spawn["actual_pid"] and fields[19] == spawn["actual_birth_counter"],
            "the prior exact waiter changed its controller birth")
    require(terminal.get("kind") == "direct_serval02_card1_terminal_process"
        and terminal.get("return_record") == inputs["wait"]
        and terminal.get("launch") == receipt(previous / "direct_current_launch.json")
        and terminal.get("uid") == UID and all(terminal.get(key) == value for key, value in identity.items())
        and outcome.get("kind") == "direct_qualification_controller_outcome" and outcome.get("passed") is True
        and outcome.get("actual_child_started") is True and outcome.get("actual_wait_saved") is True
        and outcome.get("failure") is None and outcome.get("controller_owns_gpu_standing_descriptor") is False,
        "the prior normal terminal proof or exact controller outcome failed")
    observed = document(inputs["exit_observation"]["path"])
    expected = {"controller": (spawn["actual_pid"], spawn["actual_birth_counter"]),
        "numerical_child": (metadata["pid"], metadata["process_start_time"])}
    require(observed.get("kind") == "owned_prior_direct_process_exit_observation"
        and observed.get("hostname") == HOST and type(observed.get("observer_pid")) is int
        and observed["observer_pid"] > 0 and observed.get("observer_uid") == UID
        and observed.get("all_original_processes_exited") is True
        and type(observed.get("gpu_or_standing_lock_operations")) is int
        and observed["gpu_or_standing_lock_operations"] == 0
        and len(observed.get("observations", [])) == 2,
        "a real owner-checked observation of both exact predecessor exits is required")
    roles = {}
    for row in observed["observations"]:
        role = row["role"]
        require(role in expected and role not in roles
            and (row["expected_pid"], row["expected_birth"]) == expected[role]
            and row.get("observation") == "absent" and row.get("original_process_exited") is True,
            "the exit observation must prove each exact original identity absent")
        roles[role] = row
    require(set(roles) == set(expected)
        and datetime.fromisoformat(observed["at_utc"]) >= datetime.fromisoformat(wait["finished_at_utc"]),
        "the two absent identities were not observed after the actual normal wait")
    execution = prior_result.get("execution_identity", {})
    require(all(execution.get(key) == value for key, value in identity.items()),
        "the saved predecessor result does not belong to its actual exited numerical process")
    require(type(request['serial_order']) is int and request['serial_order'] in (1, 2),
        'common timing requires one literal serial position')
    if request['serial_order'] == 1:
        accepted = document(request['accepted_evidence_snapshot_inputs']['proof']['path'])['accepted']['correctness']['2']
        require(inputs['plan'] == accepted['plan'] and inputs['result'] == accepted['result']
            and prior_plan['mode'] == 'correctness' and prior_plan['copy_seeds'] == [1, 2]
            and prior_plan['configuration']['learner']['n_copies'] == 2
            and prior_result.get('correctness_attempted') is True and prior_result.get('correctness_passed') is True
            and prior_result.get('implementation_distinct') is True and prior_result.get('measurement_valid') is False
            and prior_result.get('measurement_mode') == 'correctness'
            and prior_result.get('decision') == 'correctness_passed',
            'first-copy timing must follow the actual accepted second-copy correctness and both normal exits')
    else:
        require(prior_plan['mode'] == 'confirmation' and prior_plan['copy_seeds'] == [1]
            and prior_plan['configuration']['learner']['n_copies'] == 1
            and len(prior_plan['helpers']['files']) == 29
            and inputs['result']['path'] == str(previous / 'result.json')
            and prior_result.get('completed') is True
            and prior_plan['helpers'] == document(UNIT / 'measurement_plan.json')['helpers']
            and prior_plan['accepted_evidence_snapshot_inputs'] == request['accepted_evidence_snapshot_inputs']
            and prior_plan['direct_qualification_inputs'] == request['direct_qualification_inputs'],
            'second-copy timing must follow actual first-copy timing with the same reviewed source and snapshot')
        timing_support(prior_plan, previous, 'direct_common_confirmation').validate_pair_protocol(prior_plan, prior_result)
        reference = json.loads(json.dumps(request['reference_configuration']))
        candidate = json.loads(json.dumps(request['configuration']))
        reference['learner']['n_copies'] = candidate['learner']['n_copies'] = 1
        require(prior_plan['reference_configuration'] == reference and prior_plan['configuration'] == candidate,
            'the actual first-copy timing predecessor changed the original scientific configuration')
    return inputs


def check_unattempted(unit):
    for name in ATTEMPT_NAMES:
        path = Path(unit) / name
        require(not path.exists() and not path.is_symlink()
                and not path.with_name(path.name + ".prepared").exists(),
                "an earlier or partial immutable launch attempt exists: " + name)


def check_publication(commit):
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit),
            "the exact reviewed publication commit is required")
    frozen = document(PACKET / "controller_source_freeze.json")
    require(type(frozen.get("schema_version")) is int and frozen["schema_version"] == 1
            and frozen.get("kind") == "direct_qualification_controller_source_freeze"
            and isinstance(frozen.get("files"), dict)
            and str(PACKET / "direct_qualification_controller.py") in frozen["files"]
            and str(PACKET / "test_direct_qualification_controller.py") in frozen["files"],
            "the controller freeze must bind the actual controller and regression source")
    review_path = PACKET / "controller_root_independent_review.json"
    review = document(review_path)
    require(type(review.get("schema_version")) is int and review["schema_version"] == 1
            and review.get("kind") == "direct_qualification_controller_root_independent_review"
            and review.get("passed") is True and review.get("launch_approved") is True
            and review.get("controller_freeze") == receipt(PACKET / "controller_source_freeze.json")
            and review.get("packet_freeze") == receipt(PACKET / "packet_metadata_freeze.json")
            and review.get("plan") == receipt(UNIT / "measurement_plan.json")
            and review.get("controller_owns_gpu_standing_descriptor") is False,
            "the actual independent root launch review is missing or differs")
    files = dict(frozen["files"])
    files.update(document(PACKET / "packet_metadata_freeze.json")["files"])
    for path in (PACKET / "controller_source_freeze.json", PACKET / "packet_metadata_freeze.json", review_path):
        files[str(path)] = digest(ordinary_bytes(path))
    for name, expected in files.items():
        path = Path(name)
        require(path.is_relative_to(INTEGRATION) and digest(ordinary_bytes(path)) == expected,
                "a reviewed publication input changed")
        raw = subprocess.check_output(["git", "-C", str(INTEGRATION), "cat-file", "blob",
                    commit + ":" + str(path.relative_to(INTEGRATION))], timeout=30)
        require(digest(raw) == expected, "a reviewed input is not the exact published Git byte sequence")
    return receipt(review_path)


def load_core(unit):
    """Import only the exact staged standard-library identity helpers."""
    path = Path(unit) / "code/serval02_card1_binding.py"
    specification = importlib.util.spec_from_file_location("direct_qualification_controller_core", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    no_numerical_modules()
    return module


def read_process(pid, expected_uid):
    """Read only this actual own process; /proc kernel aliases are intentional."""
    require(type(pid) is int and pid > 0, "an actual positive returned PID is required")
    directory = Path("/proc") / str(pid)
    require(directory.stat().st_uid == expected_uid, "the actual process is not owned by this user")
    status = (directory / "status").read_bytes()
    uids = b"".join(line + b"\n" for line in status.splitlines() if line.startswith(b"Uid:"))
    require(re.fullmatch(rb"Uid:\s*" + str(expected_uid).encode() + rb"\s+"
        + str(expected_uid).encode() + rb"\s+" + str(expected_uid).encode() + rb"\s+"
        + str(expected_uid).encode() + rb"\s*", uids), "all four kernel UIDs must be the exact owner")
    raw_stat = (directory / "stat").read_bytes()
    text = raw_stat.decode("utf-8", errors="strict")
    fields = text[text.rfind(")") + 2:].split()
    require(text.split(" ", 1)[0] == str(pid) and len(fields) > 19
            and re.fullmatch(r"[1-9][0-9]*", fields[19]), "the actual process birth is invalid")
    raw_command = (directory / "cmdline").read_bytes()
    command = [part.decode("utf-8", errors="strict") for part in raw_command.rstrip(b"\0").split(b"\0")] if raw_command else []
    affinity = sorted(os.sched_getaffinity(pid))
    repeated_stat = (directory / "stat").read_text()
    repeated_fields = repeated_stat[repeated_stat.rfind(")") + 2:].split()
    require(repeated_fields[19] == fields[19] and directory.stat().st_uid == expected_uid,
            "the actual process identity changed during observation")
    return {"pid": pid, "hostname": socket.gethostname().split(".")[0],
        "process_start_time": fields[19], "parent_pid": int(fields[1]), "uid": expected_uid,
        "stat": text, "stat_sha256": digest(raw_stat), "uid_bytes": uids.decode(),
        "uid_sha256": digest(uids), "command": command,
        "command_bytes_hex": raw_command.hex(), "command_sha256": digest(raw_command),
        "affinity": affinity, "observed_at_utc": now()}


def owner_record(process):
    return {key: process[key] for key in ("pid", "stat", "stat_sha256", "uid_bytes", "uid_sha256")}


def chosen_affinity(available, count):
    require(type(count) is int and count > 0 and isinstance(available, list)
            and all(type(cpu) is int and cpu >= 0 for cpu in available)
            and len(set(available)) == len(available) and len(available) >= count,
            "the actual allowed logical-CPU mask is too small or invalid")
    return sorted(available)[:count]


def child_environment(plan, inherited):
    """Use the frozen numerical controls and remove inherited Slurm/cache settings."""
    # Only these nonsecret path/locale settings may enter the exact child environment.
    environment = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "TZ": "UTC",
        "HOME": inherited.get("HOME", pwd.getpwuid(os.getuid()).pw_dir),
        "USER": "sl5nw", "LOGNAME": "sl5nw", "LD_LIBRARY_PATH": "/usr/lib/nvidia"}
    environment.update(plan["runtime_environment"])
    environment.update(CUDA_VISIBLE_DEVICES=GPU_UUID, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
        JAX_PLATFORMS="cuda", JAX_ENABLE_COMPILATION_CACHE="false", XLA_PYTHON_CLIENT_PREALLOCATE="false",
        JAXTARI_SKIP_SPRITE_UPDATE="1", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    return environment


def append_registration(unit, plan, child, observed_argv, expected_uid):
    """Fsync the actual ledger and copied snapshot before exposing the launch path."""
    unit = Path(unit)
    run = Path(plan["campaign_run_dir"])
    require(child["command"] == observed_argv, "actual kernel argv differs from the exact Popen command")
    row = {"hostname": plan["direct_binding"]["profile"]["hostname"], "gpu_uuid": plan["gpu_uuid"],
           "kind": plan["kind"], "unit": plan["unit_id"], "source_commit": plan["source_commit"],
           "source_root": plan["source_root"], "helper_commit": plan["helpers"]["source_commit"],
           "pid": child["pid"], "start_time": child["process_start_time"]}
    ledger = run / "processes.json"
    with metadata_lock(run / "processes.lock", expected_uid):
        prior_raw = ordinary_bytes(ledger)
        records = json.loads(prior_raw)
        require(isinstance(records, list) and all(isinstance(item, dict) for item in records),
                "the live campaign process ledger is not a record list")
        require(not any(item.get("unit") == plan["unit_id"] or (
            item.get("hostname") == row["hostname"] and item.get("pid") == row["pid"]
            and item.get("start_time") == row["start_time"]) for item in records),
                "the current launch is already recorded or duplicated")
        records.append(row)
        temporary = ledger.with_name("processes.json.direct-" + str(child["pid"]) + ".prepared")
        with temporary.open("xb") as stream:
            stream.write(encoded(records))
            stream.flush()
            os.fsync(stream.fileno())
        require(ordinary_bytes(ledger) == prior_raw, "the live ledger changed outside its exact lock")
        os.replace(temporary, ledger)
        fsync_directory(run)
        publish(unit / "direct_current_process_ledger_snapshot.json", records)
        launch = {"schema_version": 1, "kind": "serval02_card1_current_launch",
            "plan": receipt(unit / "measurement_plan.json"), "process_row": row,
            "process_ledger": {"source_path": str(ledger),
                "snapshot": receipt(unit / "direct_current_process_ledger_snapshot.json")}}
        publish(unit / "direct_current_launch.json", launch)
    return launch


def wait_exact(child, identity, expected_uid, seconds):
    """Bound only our actual spawned PID after its own UID and birth are rechecked."""
    timeouts = []
    try:
        return child.wait(timeout=seconds), timeouts
    except subprocess.TimeoutExpired:
        require(identity is not None, "unknown child birth cannot be signaled; retain the actual child and no-successor rule")
        for action, wait_seconds in ((signal.SIGTERM, 120), (signal.SIGKILL, None)):
            observed = read_process(child.pid, expected_uid)
            require(observed["process_start_time"] == identity["process_start_time"],
                    "timeout child PID was reused; no signal was sent")
            timeouts.append({"signal": int(action), "observed_identity": observed, "at_utc": now()})
            os.kill(child.pid, action)
            try:
                return child.wait(timeout=wait_seconds), timeouts
            except subprocess.TimeoutExpired:
                continue
    raise RuntimeError("the exact child wait produced no result")


def child_identity_observation(observed, command, affinity, parent_pid, popen_pid, *, initial_birth=None):
    """Save each actual predicate without accepting a mismatch or inferring its cause."""
    predicates = {"parent_pid_matches": observed["parent_pid"] == parent_pid,
        "command_matches": observed["command"] == command,
        "inherited_affinity_matches": observed["affinity"] == affinity,
        "popen_pid_matches": observed["pid"] == popen_pid}
    if initial_birth is not None:
        predicates["birth_matches_first_observation"] = observed["process_start_time"] == initial_birth
    return {"schema_version": 1, "kind": "actual_direct_child_identity_observation_before_registration",
        "at_utc": now(), "actual_popen_pid": popen_pid, "expected_parent_pid": parent_pid,
        "expected_command": command, "expected_affinity": affinity, "observed_child": observed,
        "predicate_results": predicates, "all_predicates_match": all(predicates.values()),
        "registration_performed_by_this_observation": False, "historical_failure_cause_established": False,
        **BOUNDARY}


def wait_for_exact_child_startup(child, initial, command, affinity, parent_pid, expected_uid, unit,
                                 *, started_monotonic, maximum_seconds=30., clock=time.monotonic,
                                 sleeper=time.sleep, observer=None):
    """Wait only for empty raw argv, retaining exact identity and a fixed thirty-second bound."""
    require(type(maximum_seconds) in (int, float) and 0 < maximum_seconds <= 30,
        "the empty-command startup allowance must be positive and no more than thirty seconds")
    require(type(started_monotonic) in (int, float) and 0 <= started_monotonic < float("inf"),
        "the actual startup monotonic origin is required")
    require(type(command) is list and command and all(type(value) is str and value for value in command),
        "startup requires one exact nonempty expected command")
    observer = read_process if observer is None else observer
    deadline = started_monotonic + maximum_seconds
    expected_raw = b"\0".join(value.encode("utf-8") for value in command) + b"\0"
    observations, ready, failure = [], None, None
    current = initial
    birth = initial["process_start_time"]
    try:
        while True:
            observed = clock()
            require(type(observed) in (int, float) and started_monotonic <= observed < float("inf"),
                "the startup observation clock is invalid")
            require(len(observations) < 4096, "the bounded startup observation count was exceeded")
            raw = bytes.fromhex(current["command_bytes_hex"])
            decoded = [value.decode("utf-8", errors="strict") for value in raw.rstrip(b"\0").split(b"\0")] if raw else []
            ownership = bool(re.fullmatch(r"Uid:\s*" + str(expected_uid) + r"\s+" + str(expected_uid)
                + r"\s+" + str(expected_uid) + r"\s+" + str(expected_uid) + r"\s*", current["uid_bytes"]))
            predicates = {"all_four_uids_match": ownership and type(current["uid"]) is int and current["uid"] == expected_uid
                    and digest(current["uid_bytes"].encode()) == current["uid_sha256"],
                "popen_pid_matches": type(current["pid"]) is int and current["pid"] == child.pid,
                "birth_matches_first_observation": type(birth) is str and re.fullmatch(r"[1-9][0-9]*", birth) is not None
                    and current["process_start_time"] == birth,
                "parent_pid_matches": type(current["parent_pid"]) is int and current["parent_pid"] == parent_pid,
                "inherited_affinity_matches": current["affinity"] == affinity,
                "raw_command_matches_decoded_record": decoded == current["command"] and digest(raw) == current["command_sha256"]}
            row = {"observed_monotonic": observed, "observed_at_utc": now(), "observed_child": current,
                "identity_predicates": predicates, "raw_command_is_empty": raw == b"",
                "expected_command_matches": current["command"] == command,
                "expected_raw_command_matches": raw == expected_raw, "actual_popen_returncode": None}
            observations.append(row)
            require(observed <= deadline, "the empty-command startup deadline expired")
            require(all(predicates.values()), "actual child identity changed during empty-command startup")
            returned = child.poll()
            row["actual_popen_returncode"] = returned
            require(returned is None, "the exact child exited before positive startup identity")
            require(clock() <= deadline, "the empty-command startup deadline expired during observation")
            if current["command"] == command:
                require(raw == expected_raw, "nonempty raw child command differs from the exact expected bytes")
                ready = current
                break
            require(raw == b"" and current["command"] == [],
                "unexpected nonempty child command cannot wait or register")
            remaining = deadline - clock()
            require(remaining > 0, "the empty-command startup deadline expired")
            sleeper(min(.01, remaining))
            after_poll = child.poll()
            row["after_empty_poll"] = {"observed_monotonic": clock(), "actual_popen_returncode": after_poll}
            require(after_poll is None, "the exact child exited while its raw command was empty")
            current = observer(child.pid, expected_uid)
    except BaseException as error:
        failure = {"type": type(error).__name__, "error": str(error), "at_utc": now()}
        raise
    finally:
        finished = clock()
        if ready is not None and finished > deadline:
            ready = None
            failure = {"type": "TimeoutError", "error": "exact command completed after the startup deadline", "at_utc": now()}
        publish(Path(unit) / "direct_controller_startup_observation_sequence.json", {
            "schema_version": 1, "kind": "bounded_empty_raw_command_startup_observations",
            "started_monotonic": started_monotonic, "deadline_monotonic": deadline,
            "maximum_wait_seconds": maximum_seconds, "finished_monotonic": finished,
            "actual_popen_pid": child.pid, "initial_birth": birth, "expected_parent_pid": parent_pid,
            "expected_command": command, "expected_affinity": affinity, "expected_uid": expected_uid,
            "observations": observations, "exact_command_observed": ready is not None,
            "registration_performed_by_this_observer": False, "failure": failure,
            "historical_failure_cause_for_other_attempts_established": False, **BOUNDARY})
    if ready is None:
        raise TimeoutError("exact command completed after the startup deadline")
    return ready


def supervise(plan, command, unit, environment, *, expected_uid, cpu_count,
              registration_hostname, seconds=10800, after_spawn=None, launch_lock=None, prelaunch_check=None):
    """Common real-process mechanism; tests use only private CPU fixture paths.

    The production entry has fixed paths/identity/publication checks. No command,
    hostname, CPU count, timeout or environment override is exposed by the CLI.
    """
    unit = Path(unit)
    no_numerical_modules()
    check_unattempted(unit)
    require(registration_hostname == plan["direct_binding"]["profile"]["hostname"], "registration host changed")
    parent = read_process(os.getpid(), expected_uid)
    original_affinity = parent["affinity"]
    selected_affinity = chosen_affinity(original_affinity, cpu_count)
    intent = {"schema_version": 1, "kind": "one_direct_qualification_controller_attempt",
        "at_utc": now(), "plan": receipt(unit / "measurement_plan.json"), "command": command,
        "intended_environment": environment, "controller": parent,
        "available_cpu_affinity": original_affinity, "selected_cpu_affinity": selected_affinity,
        "cpu_affinity_is_exclusive": False, "slurm_cpu_allocation_claimed": False,
        "actual_child_pid": None, **BOUNDARY}
    publish(unit / "direct_controller_intent.json", intent)
    child = None
    child_identity = None
    failure = None
    wait_record = None
    stdout = stderr = None
    try:
        with metadata_lock(launch_lock, expected_uid) if launch_lock is not None else nullcontext():
            if prelaunch_check is not None:
                prelaunch_check()
            os.sched_setaffinity(0, selected_affinity)
            require(sorted(os.sched_getaffinity(0)) == selected_affinity, "controller CPU affinity did not take effect")
            stdout = (unit / "direct_controller_stdout.raw").open("xb", buffering=0)
            stderr = (unit / "direct_controller_stderr.raw").open("xb", buffering=0)
            started = now()
            child = subprocess.Popen(command, env=environment, cwd=unit, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, shell=False, close_fds=True)
            # Popen has returned a real PID; nothing below invents a future identity.
            startup_started = time.monotonic()
            child_identity = read_process(child.pid, expected_uid)
            publish(unit / "direct_controller_initial_child_observation.json",
                child_identity_observation(child_identity, command, selected_affinity, os.getpid(), child.pid))
            child_identity = wait_for_exact_child_startup(child, child_identity, command, selected_affinity,
                os.getpid(), expected_uid, unit, started_monotonic=startup_started)
            require(child_identity["parent_pid"] == os.getpid() and child_identity["command"] == command
                    and child_identity["affinity"] == selected_affinity, "actual child parent, argv or inherited CPU affinity differs")
            spawn = {"schema_version": 1, "kind": "actual_direct_qualification_popen_return",
                "returned_at_utc": now(), "started_at_utc": started, "popen_pid": child.pid,
                "command": command, "observed_child": child_identity, "controller": read_process(os.getpid(), expected_uid),
                "cpu_affinity_is_exclusive": False, "slurm_cpu_allocation_claimed": False, **BOUNDARY}
            publish(unit / "direct_controller_spawn.json", spawn)
            if after_spawn is not None:
                after_spawn(child, child_identity)  # CPU tests only; never provided by the fixed CLI.
            current_child = read_process(child.pid, expected_uid)
            publish(unit / "direct_controller_pre_registration_child_observation.json",
                child_identity_observation(current_child, command, selected_affinity, os.getpid(), child.pid,
                    initial_birth=child_identity["process_start_time"]))
            require(current_child["process_start_time"] == child_identity["process_start_time"]
                    and current_child["parent_pid"] == os.getpid() and current_child["command"] == command
                    and current_child["affinity"] == selected_affinity, "actual child exited or changed before durable registration")
            append_registration(unit, plan, child_identity, command, expected_uid)
    except BaseException as error:
        failure = {"type": type(error).__name__, "error": str(error), "at_utc": now()}
    finally:
        try:
            if child is not None:
                try:
                    returncode, timeouts = wait_exact(child, child_identity, expected_uid, seconds)
                except BaseException as error:
                    # An identity-read error must never bypass the actual wait or start a successor.
                    failure = failure or {"type": type(error).__name__, "error": str(error), "at_utc": now()}
                    returncode, timeouts = child.wait(), [{"type": type(error).__name__, "error": str(error)}]
                if timeouts:
                    failure = failure or {"type": "ControllerTimeoutOrWaitFailure",
                        "error": "the child did not complete within an uninterrupted exact controller wait",
                        "actual_returncode": returncode, "at_utc": now()}
                for stream in (stdout, stderr):
                    if stream is not None:
                        os.fsync(stream.fileno())
                if child_identity is not None:
                    wait_record = {"schema_version": 1, "kind": (
                            "owned_direct_process_wait_return_after_controller_intervention" if timeouts
                            else "owned_direct_process_wait_return"),
                        "command": command, "pid": child.pid, "hostname": registration_hostname,
                        "process_start_time": child_identity["process_start_time"], "uid": expected_uid,
                        "waited_pid": child.pid, "wait_semantics": "Popen.wait on this exact spawned child",
                        "returncode": returncode, "exited_normally": returncode >= 0,
                        "waiter_before": owner_record(parent), "waiter_after": owner_record(read_process(os.getpid(), expected_uid)),
                        "finished_at_utc": now(), "timeouts": timeouts,
                        **BOUNDARY}
                    publish(unit / "direct_process_wait_return.json", wait_record)
                    if timeouts:
                        publish(unit / "direct_controller_timeout_observation.json", {
                            "schema_version": 1, "kind": "direct_child_interrupted_controller_wait",
                            "wait": receipt(unit / "direct_process_wait_return.json"),
                            "actual_returncode": returncode, "actual_exited_normally": returncode >= 0,
                            "timeouts": timeouts, "ordinary_successful_terminal_proof_allowed": False,
                            **BOUNDARY})
                    publish(unit / "direct_controller_output_hashes.json", {
                        "schema_version": 1, "kind": "actual_direct_child_output_hashes_after_wait",
                        "wait": receipt(unit / "direct_process_wait_return.json"),
                        "stdout": log_receipt(unit / "direct_controller_stdout.raw"),
                        "stderr": log_receipt(unit / "direct_controller_stderr.raw"), **BOUNDARY})
                    if returncode == 0 and failure is None and not timeouts:
                        require((unit / "direct_current_launch.json").is_file(), "ordinary child success lacks durable launch proof")
                        publish(unit / "direct_terminal_process.json", {
                            "schema_version": 1, "kind": "direct_serval02_card1_terminal_process",
                            "launch": receipt(unit / "direct_current_launch.json"), "pid": child.pid,
                            "hostname": registration_hostname, "process_start_time": child_identity["process_start_time"],
                            "uid": expected_uid, "return_record": receipt(unit / "direct_process_wait_return.json"), **BOUNDARY})
                else:
                    publish(unit / "direct_controller_unidentified_wait.json", {
                        "kind": "actual_popen_wait_without_complete_identity", "pid": child.pid,
                        "command": command, "returncode": returncode, "at_utc": now(), **BOUNDARY})
        finally:
            for stream in (stdout, stderr):
                if stream is not None:
                    stream.close()
            os.sched_setaffinity(0, original_affinity)
            require(sorted(os.sched_getaffinity(0)) == original_affinity, "controller original CPU affinity was not restored")
    if failure is not None:
        publish(unit / "direct_controller_failure.json", {**failure, "schema_version": 1,
            "kind": "direct_qualification_controller_failure", "actual_child_pid": None if child is None else child.pid,
            "actual_wait_saved": wait_record is not None, **BOUNDARY})
    outcome = {"schema_version": 1, "kind": "direct_qualification_controller_outcome",
        "actual_child_started": child is not None, "actual_wait_saved": wait_record is not None,
        "passed": failure is None and wait_record is not None and wait_record["returncode"] == 0
            and not wait_record["timeouts"],
        "failure": failure, "qualification_independently_verified": False, **BOUNDARY}
    publish(unit / "direct_controller_outcome.json", outcome)
    return outcome


def actual_launch(commit):
    """No remote transport or automatic retry is implemented."""
    plan, command, _ = checked_inputs()
    check_publication(commit)
    require(socket.gethostname().split(".")[0] == HOST and pwd.getpwuid(os.getuid()).pw_name == "sl5nw"
            and os.getuid() == UID and os.geteuid() == UID and tuple(os.getresuid()) == (UID, UID, UID),
            "actual mode is restricted to the exact approved direct host and owner")
    require(not any(name.startswith("SLURM_") for name in os.environ),
            "actual direct mode cannot run in a Slurm allocation")
    own = read_process(os.getpid(), UID)
    require(own["hostname"] == HOST, "the actual controller host changed")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU_UUID,
            "actual controller must retain the exact reviewed original UUID selector")
    core = load_core(UNIT)

    def prelaunch_check():
        checked_inputs()
        core.validate_plan(plan, "selected_measurement")
        core.source_identity(plan)
        core.validate_history(plan)
        core.check_live_admission(core.current_admission_claim(plan))

    environment = child_environment(plan, os.environ)
    return supervise(plan, command, UNIT, environment, expected_uid=UID, cpu_count=8,
        registration_hostname=HOST, launch_lock=RUN / "queue/metadata.lock", prelaunch_check=prelaunch_check)


def timing_copy_order(request, plan=None):
    """Bind strict one/two-copy requests to the same serial position and exact seed range."""
    count = request['configuration']['learner']['n_copies']
    require(type(count) is int and count in (1, 2) and type(request['serial_order']) is int
        and request['serial_order'] == count and type(request['copy_seeds']) is list
        and all(type(seed) is int for seed in request['copy_seeds'])
        and request['copy_seeds'] == list(range(1, count + 1)),
        'common timing serial position must equal its strict actual copy count and seed range')
    if plan is not None:
        actual = plan['configuration']['learner']['n_copies']
        require(type(actual) is int and actual == count and plan['copy_seeds'] == request['copy_seeds']
            and all(type(seed) is int for seed in plan['copy_seeds']),
            'the current timing plan changed its actual copy count or serial seeds')
    return count


def timing_support(plan, unit, name):
    """Load only one exact staged metadata helper after verifying its declared source bytes."""
    path = Path(unit) / 'code' / (name + '.py')
    relative = '10_jax_exploration_platform/scripts/atari_pixel_campaign/' + path.name
    require(receipt(path)['sha256'] == plan['helpers']['files'][relative], 'the staged timing metadata helper changed')
    key = 'reviewed_timing_' + name + '_' + plan['helpers']['files'][relative]
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    no_numerical_modules()
    return sys.modules[key]


def timing_snapshot(plan, request):
    """Bind actual reviewed historical inputs to the exact current twenty-nine-helper timing namespace."""
    timing_copy_order(request, plan)
    require(plan.get('mode') == 'confirmation' and type(plan.get('rounds')) is int and plan['rounds'] == 7
        and type(plan.get('iterations')) is int and plan['iterations'] == 10
        and type(plan.get('minimum_updates_per_round')) is int and plan['minimum_updates_per_round'] == 10
        and len(plan['helpers']['files']) == 29 and plan['helpers']['source_commit'] == request['helper_source_commit']
        and re.fullmatch('[0-9a-f]{40}', request['helper_source_commit'])
        and plan.get('accepted_evidence_snapshot_inputs') == request['accepted_evidence_snapshot_inputs']
        and plan.get('common_correctness_inputs') == request['common_correctness_inputs']
        and not any(key in plan for key in ('configured_isolation', 'direct_qualification', 'target_qualification',
            'completed_study', 'device_memory_limit_protocol', 'final_confirmation_protocol', 'native_process_control')),
        'only fresh seven-by-ten common timing with actual reviewed snapshot inputs is supported')
    deadline = time.monotonic() + 180
    root = Path(request['helper_source_root'])
    require(root.resolve(strict=True) == root, 'the actual timing helper checkout must be ordinary')
    for relative, expected in plan['helpers']['files'].items():
        require(relative.startswith('10_jax_exploration_platform/scripts/atari_pixel_campaign/')
            and Path(relative).name == relative.split('/')[-1] and len(Path(relative).parts) == 4
            and re.fullmatch('[0-9a-f]{64}', expected), 'only exact declared timing helper paths and hashes are allowed')
        remaining = deadline - time.monotonic()
        require(remaining > 0, 'timing helper source verification exceeded its metadata deadline')
        raw = subprocess.check_output(['git', '-C', str(root), 'show', request['helper_source_commit'] + ':' + relative],
            timeout=min(30, remaining))
        require(digest(raw) == expected == digest(ordinary_bytes(root / relative))
            == digest(ordinary_bytes(UNIT / 'code' / Path(relative).name)), 'timing helper Git, checkout and staged source differ')
    support = timing_support(plan, UNIT, 'accepted_evidence_snapshot')
    require(set(plan['helpers']['files']) == {'10_jax_exploration_platform/scripts/atari_pixel_campaign/' + name
        for name in support.required_helper_names()}, 'the current timing helper set is not the exact reviewed twenty-nine-file closure')
    selected = support.consume(plan)
    for count in ('1', '2'):
        accepted = selected.proof['accepted']['correctness'][count]
        require(request['accepted_correctness'][count] == {name: accepted[name] for name in ('plan', 'result', 'review')},
            'the request changed a genuinely accepted correctness plan, result or review')
    require(request['qualification_plan'] == selected.qual['plan']
        and request['direct_qualification_inputs'] == {name: selected.qual[name] for name in ('result', 'review')}
        and request['common_correctness_inputs'] == {name: selected.correctness[name] for name in ('result', 'review')},
        'the current timing request lost its exact original qualification or per-count correctness')
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--publication-commit")
    options = parser.parse_args(argv)
    if options.launch:
        result = actual_launch(options.publication_commit)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    plan, command, _ = checked_inputs()
    check_unattempted(UNIT)
    if options.publication_commit is not None:
        check_publication(options.publication_commit)
    print(json.dumps({"kind": "direct_qualification_controller_default_check", "passed": True,
        "action_performed": False, "command": command, "plan": receipt(UNIT / "measurement_plan.json"),
        "actual_child_pid": None, "publication_required_before_launch": True, **BOUNDARY}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

