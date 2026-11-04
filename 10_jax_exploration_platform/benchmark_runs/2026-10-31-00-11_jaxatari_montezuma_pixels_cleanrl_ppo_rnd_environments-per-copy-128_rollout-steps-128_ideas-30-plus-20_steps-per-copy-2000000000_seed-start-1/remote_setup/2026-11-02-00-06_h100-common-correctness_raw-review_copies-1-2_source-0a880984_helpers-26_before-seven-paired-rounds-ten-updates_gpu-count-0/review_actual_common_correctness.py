"""Review actual H100 common correctness only after its real normal wait and exits.

The default reads prepared metadata and source declarations. It never opens a
correctness result, creates an approval, contacts a host, or runs numerical code.
An actual review runs only on this campaign's existing owned CPU allocation.
"""
import argparse
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time
import traceback

P = Path(__file__).resolve(strict=True).parent
INPUT_SHA256 = "51457b4c11f2db8ce16b44987a3255bb917a48b08f9562be9b96faa777ad7bb5"
NUMERICAL = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "cupy", "scipy", "jaxtari", "exploration_platform"}


def require(value, message):
    """Reject an unverified actual input without creating replacement evidence."""
    if not value:
        raise ValueError(message)


def encoded(value):
    """Use the exact stable encoding required by the published correctness guard."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read(path):
    """Read exact ordinary metadata with unique keys and finite numbers."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "an ordinary exact input is required")
    def unique(pairs):
        output = {}
        for key, value in pairs:
            require(key not in output, "duplicate metadata key"); output[key] = value
        return output
    raw = path.read_bytes()
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite metadata")))


def reference(path):
    """Hash an actual ordinary file in bounded chunks."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "an ordinary exact file is required")
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): value.update(chunk)
    return {"path": str(path), "sha256": value.hexdigest()}


def checked(value):
    """Recheck the exact path and hash before reading a declared JSON record."""
    require(type(value) is dict and set(value) == {"path", "sha256"} and reference(value["path"]) == value,
        "a pinned actual input changed")
    return read(value["path"])


def load(path, name):
    """Load only the explicitly supplied ordinary source file."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "an exact source file is required")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def publish(path, value):
    """Write one new durable audit artifact without overwriting an earlier attempt."""
    raw = encoded(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    return reference(path)


def validate_source_freeze():
    """Recheck the closed four-file source set before any actual result or approval can be read."""
    value = read(P / "source_freeze.json")
    names = ("prepared_inputs.json", "review_actual_common_correctness.py",
        "run_raw_review_source_tests.py", "test_review_actual_common_correctness.py")
    require(type(value) is dict and set(value) == {"schema_version", "kind", "files", "scientific_credit",
            "actual_review_written", "actual_terminal_audit_performed", "source_review_required"}
        and type(value["schema_version"]) is int and value["schema_version"] == 1
        and value["kind"] == "h100_common_correctness_raw_review_source_freeze"
        and type(value["scientific_credit"]) is int and value["scientific_credit"] == 0
        and value["actual_review_written"] is False and value["actual_terminal_audit_performed"] is False
        and value["source_review_required"] is True
        and type(value["files"]) is dict and set(value["files"]) == {str(P / name) for name in names},
        "the reviewed source freeze must retain exactly its four required ordinary files")
    require(all(type(checksum) is str and re.fullmatch(r"[0-9a-f]{64}", checksum)
        and reference(path)["sha256"] == checksum for path, checksum in value["files"].items()),
        "an actual approved raw-review source file changed")
    return value


def prepared(copies):
    """Join existing prepared plans and source; do not open any future correctness result."""
    validate_source_freeze()
    require(reference(P / "prepared_inputs.json")["sha256"] == INPUT_SHA256, "prepared review inputs changed")
    inputs = read(P / "prepared_inputs.json")
    require(str(copies) in inputs["copies"] and type(copies) is int and copies in (1, 2), "only the two prepared copy counts are allowed")
    row = inputs["copies"][str(copies)]
    plan = checked(row["plan"])
    require(checked(row["preparation"])["passed"] is True and checked(row["software"])["passed"] is True,
        "actual metadata preparation and software proof must pass")
    qualified = checked(inputs["qualified_plan"])
    require(plan["direct_qualification_inputs"] == {"result": inputs["qualified_result"], "review": inputs["qualified_review"]}
        and plan["source_commit"] == inputs["source_commit"] and plan["helpers"]["source_commit"] == inputs["helper_source_commit"]
        and plan["helpers"] == qualified["helpers"] and len(plan["helpers"]["files"]) == 26
        and plan["mode"] == "correctness" and plan["copy_seeds"] == list(range(1, copies + 1)),
        "actual correctness must use the exact qualified26 source, copy count and seeds")
    reference_configuration = json.loads(json.dumps(qualified["configuration"]))
    reference_configuration["learner"]["n_copies"] = copies
    candidate = json.loads(json.dumps(reference_configuration))
    candidate["environment"]["cache_room_backgrounds"] = True
    candidate["environment"]["render_final_two_only"] = True
    require(plan["reference_configuration"] == reference_configuration and plan["configuration"] == candidate
        and all(plan[key] == qualified[key] for key in ("source_root", "source_commit", "core", "paired_source",
            "python", "hardware_class", "runtime_environment", "asset_record")),
        "only the two declared environment flags may differ from the qualified original")
    guard_ref = inputs["metadata_guard"]
    require(reference(guard_ref["path"]) == guard_ref, "the reviewed metadata-only guard changed")
    guard = load(guard_ref["path"], "reviewed_direct_common_guard")
    return inputs, plan, qualified, guard


def declared_roots(plan, qualified, guard):
    """Allow both exact eight-root manifests only after matching all original names and commits."""
    expected, roots = guard.pinned_history_declarations(plan)
    roots.update(("/p/rlprojects/RND/08_cleanrl_ppo_rnd/cleanrl",
        "/p/rlprojects/RND/reference_repos/jaxtari_pixel_study"))
    for current in (plan, qualified):
        unit = Path(current["campaign_run_dir"]) / "data" / current["unit_id"]
        require(current["campaign_run_dir"] == plan["campaign_run_dir"] and unit.resolve(strict=True) == unit,
            "every source and manifest must belong to the exact ordinary campaign unit")
        for field in ("core", "paired_source", "helpers"):
            value = current[field]; root = Path(value["source_root"])
            require(root.is_absolute() and root.resolve(strict=True) == root and root.is_dir()
                and re.fullmatch(r"[a-f0-9]{40}", value["source_commit"]), "invalid exact source Git root")
            roots.add(str(root))
        history = checked(current["direct_binding"]["frozen_inputs"])
        require(current["direct_binding"]["frozen_inputs"]["path"] == str(unit / "direct_history/inputs.json")
            and history["evidence_manifest"]["path"] == str(unit / "direct_history/evidence_manifest.json"),
            "historical inputs must remain inside this exact prepared unit")
        manifest = checked(history["evidence_manifest"])
        require(set(manifest) == {"schema_version", "kind", "files", "git_roots"}
            and type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
            and manifest["kind"] == "portable_raw_evidence" and len(manifest["git_roots"]) == 8,
            "the complete eight-root original history manifest is required")
        staging = unit / "direct_history"; names, targets = set(), set()
        for row in manifest["git_roots"]:
            require(type(row) is dict and set(row) == {"original_root", "relative_path", "commit"}, "unknown historical mapping")
            relative, original = row["relative_path"], row["original_root"]
            require(type(relative) is str and relative and relative != "." and "\0" not in relative
                and not PurePosixPath(relative).is_absolute() and ".." not in PurePosixPath(relative).parts
                and str(PurePosixPath(relative)) == relative and original in expected and original not in names
                and row["commit"] == expected[original], "unlisted, duplicate or changed original historical identity")
            target = staging / relative
            require(target.resolve(strict=True) == target and target.is_relative_to(staging) and target.is_dir()
                and str(target) not in targets, "the historical Git target is not exact, ordinary and distinct")
            names.add(original); targets.add(str(target))
        require(names == set(expected) and len(targets) == 8, "all eight exact original names/commits must be retained")
        roots.update(targets)
    return roots


def owned_cpu(plan):
    """Prove only this campaign's exact existing CPU allocation and own numeric cgroup."""
    require(os.getuid() == os.geteuid() == 2618919 and os.environ.get("SLURM_JOB_ID") == "6783003"
        and os.environ.get("SLURM_STEP_ID", "").isdecimal(), "actual review requires the exact owned CPU allocation")
    run = Path(plan["campaign_run_dir"]); step = os.environ["SLURM_STEP_ID"]
    for name in ("submitted_jobids.txt", "submitted_jobids_uva.txt"):
        require((run / "slurm" / name).read_text().splitlines().count("6783003") == 1, "owned CPU ledger missing or duplicated")
    rows = [line.split("\t") for line in (run / "slurm/submissions.tsv").read_text().splitlines()
        if line.split("\t", 1)[0] == "6783003"]
    require(len(rows) == 1 and rows[0][1:3] == ["monitor", "79a01e851ff92ddba7d0e12eaf8164cab2028878"],
        "the actual owned CPU source row changed")
    expected = "/system.slice/slurmstepd.scope/job_6783003/step_" + step + "/user/task_0"
    require(Path("/proc/self/cgroup").read_text().splitlines() == ["0::" + expected],
        "the actual reviewer must retain its own numeric CPU step")
    return {"job_id": "6783003", "step_id": step, "pid": os.getpid(), "uid": 2618919,
        "birth": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]}


def actual_raw(directory, plan, qualified, guard, support):
    """Run the unchanged complete saved consumer before writing any raw pass or review."""
    result = read(directory / "result.json")
    previous, binding = support.verify_saved_measurement(directory, result)
    require(previous == plan, "the unchanged saved consumer returned another exact prepared plan")
    validation = support.adjacent("post_comparison_validation")
    training = support.adjacent("train_optimized")
    validation.correctness(result)
    validation.configuration(plan["configuration"]); validation.configuration(plan["reference_configuration"])
    validation.execution_identity(result)
    require(result.get("measurement_mode") == "correctness" and result.get("measurement_valid") is False
        and result.get("correctness_attempted") is True and result.get("correctness_passed") is True
        and result.get("implementation_distinct") is True, "actual correctness completion fields are incomplete")
    metadata = read(directory / "direct_process_metadata.json")
    terminal = guard.controller_evidence(support, directory, metadata)
    expected = guard.expected_review_identity(plan, result, binding, metadata, directory)
    backend = read(directory / "installed_backend.json")
    loaded = result["loaded_gpu_libraries"]
    training.validate_gpu_library_receipt(loaded, {"installed_gpu_libraries": backend["gpu_libraries"]})
    require(loaded["pid"] == metadata["after"]["identity"]["pid"]
        and binding["native"]["driver_library"] in loaded["loaded_shared_libraries"], "loaded images differ from the actual bound process")
    for path, image in loaded["verified_package_libraries"].items():
        require(reference(path)["sha256"] == image["sha256"], "an actual shared CUDA/cuDNN image changed")
    assets = training.verify_assets(plan["asset_record"], {})
    require(read(directory / "asset_receipt.json") == assets, "actual original asset bytes changed")
    qualified_review = checked(plan["direct_qualification_inputs"]["review"])
    source_contract = qualified_review["source_contract"]
    checked(source_contract)
    capture = read(directory / "direct_backend_capture.json")
    raw = {"schema_version": 1, "kind": "independent_actual_direct_common_correctness_raw_validation",
        "passed": True, **expected, **terminal, "execution_identity": result["execution_identity"],
        "unchanged_staged_consumer_passed": True, "asset_receipt": reference(directory / "asset_receipt.json"),
        "installed_backend": reference(directory / "installed_backend.json"),
        "loaded_gpu_libraries_sha256": hashlib.sha256(encoded(loaded)).hexdigest(),
        "captured_backend": reference(directory / "direct_backend_capture.json"),
        "direct_whole_work_memory_sha256": hashlib.sha256(encoded(result["direct_whole_work_memory"])).hexdigest(),
        "backend_expected": plan["direct_backend_image"]["expected"],
        "backend_preparation": plan["inputs"]["direct_backend_image_record"],
        "backend_capture_begin": reference(directory / "direct_backend_capture_begin.json"),
        "post_installed_package_sha256": hashlib.sha256(encoded(capture["post_installed_package"])).hexdigest(),
        "qualification_source_contract": source_contract, "admission_granted": False, "worker_admitted": False,
        **{name: 0 for name in guard.ZERO_FIELDS}}
    require(set(raw) == guard.RAW_FIELDS and not NUMERICAL.intersection(sys.modules),
        "raw proof changed its published exact field contract or imported numerical code")
    return result, expected, raw


def terminal_ready(path, directory, plan, guard):
    """Join both real exited identities and the exact reviewed source before scientific recomputation."""
    validate_source_freeze()
    ready = read(path)
    require(type(ready.get("schema_version")) is int and ready["schema_version"] == 1
        and ready.get("kind") == "actual_owned_direct_correctness_terminal_ready"
        and ready.get("root_confirmed_result_normal_wait_controller_outcome_and_exits") is True
        and ready.get("plan") == reference(directory / "measurement_plan.json")
        and ready.get("result") == reference(directory / "result.json")
        and ready.get("terminal") == reference(directory / "direct_terminal_process.json")
        and ready.get("outcome") == reference(directory / "direct_controller_outcome.json"),
        "actual terminal readiness has not been independently recorded")
    approval = checked(ready["independent_source_review"])
    require(approval.get("kind") == "independent_h100_common_correctness_raw_review_source_review"
        and approval.get("approved") is True and approval.get("passed") is True
        and approval.get("independent_of_author") is True
        and approval.get("source_freeze") == reference(P / "source_freeze.json"),
        "the actual exact raw-review source needs independent approval before use")
    sys.path.insert(0, str(directory / "code"))
    support = load(directory / "code/direct_measurement.py", "direct_measurement")
    metadata = read(directory / "direct_process_metadata.json")
    process = metadata["after"]["identity"]
    refs = guard.controller_evidence(support, directory, metadata)
    wait = checked(refs["wait_record"])
    spawn = read(directory / "direct_controller_spawn.json")
    parent = spawn["controller"]
    observation = checked(ready["exit_observation"])
    expected = {"controller": (parent["pid"], parent["process_start_time"]),
        "numerical_child": (process["pid"], process["process_start_time"])}
    require(all(type(pid) is int and pid > 0 and type(birth) is str and re.fullmatch(r"[1-9][0-9]*", birth)
        for pid, birth in expected.values()) and type(process["uid"]) is int and process["uid"] == 2618919,
        "the real saved controller and numerical child need exact positive integer PIDs and canonical birth counters")
    require(observation.get("kind") == "owned_prior_direct_process_exit_observation"
        and observation.get("hostname") == process["hostname"] == "serval02"
        and type(observation.get("observer_uid")) is int and observation["observer_uid"] == process["uid"] == 2618919
        and type(observation.get("observer_pid")) is int and observation["observer_pid"] > 0
        and observation.get("all_original_processes_exited") is True
        and type(observation.get("gpu_or_standing_lock_operations")) is int
        and observation["gpu_or_standing_lock_operations"] == 0
        and len(observation.get("observations", [])) == 2, "both actual original processes must have exited")
    seen = set()
    for row in observation["observations"]:
        role = row["role"]
        require(role in expected and role not in seen
            and type(row.get("expected_pid")) is int and row["expected_pid"] > 0
            and type(row.get("expected_birth")) is str and re.fullmatch(r"[1-9][0-9]*", row["expected_birth"])
            and (row["expected_pid"], row["expected_birth"]) == expected[role]
            and row.get("observation") == "absent" and row.get("original_process_exited") is True,
            "an exit proof named another identity or a process that had not exited")
        seen.add(role)
    require(seen == set(expected)
        and guard.read(refs["terminal_process"]["path"])[0]["return_record"] == refs["wait_record"],
        "the terminal proof changed its exact normal wait")
    from datetime import datetime
    observed_at = datetime.fromisoformat(observation["at_utc"])
    finished_at = datetime.fromisoformat(wait["finished_at_utc"])
    require(observed_at.tzinfo is not None and finished_at.tzinfo is not None and observed_at >= finished_at,
        "the real two-process exit observation must follow the exact normal wait")
    return ready, support


def main(argv=None):
    """Default to source-only checks; actual auditing remains an explicit owned-CPU terminal action."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copies", required=True, type=int, choices=(1, 2))
    parser.add_argument("--actual-terminal-review", action="store_true")
    parser.add_argument("--terminal-ready-record", type=Path)
    parser.add_argument("--reviewer-identity")
    args = parser.parse_args(argv)
    inputs, plan, qualified, guard = prepared(args.copies)
    bootstrap = guard.stdlib_ctypes_bootstrap()
    roots = declared_roots(plan, qualified, guard)
    sys.addaudithook(lambda event, values: guard.metadata_event_guard(event, values, roots))
    require(not NUMERICAL.intersection(sys.modules), "the fresh metadata reviewer already imported numerical code")
    if not args.actual_terminal_review:
        require(args.terminal_ready_record is None and args.reviewer_identity is None, "terminal options require an actual terminal review")
        print(json.dumps({"kind": "h100_common_correctness_source_only_review_preparation", "passed": True,
            "plan": inputs["copies"][str(args.copies)]["plan"], "declared_git_roots": len(roots),
            "actual_correctness_result_opened": False, "actual_review_written": False, "stdlib_bootstrap": bootstrap,
            "host_or_gpu_actions": 0}, sort_keys=True))
        return 0
    require(args.terminal_ready_record is not None and type(args.reviewer_identity) is str
        and bool(args.reviewer_identity.strip()), "an actual terminal record and real reviewer identity are required")
    context = owned_cpu(plan)
    run = Path(plan["campaign_run_dir"]); directory = run / "data" / plan["unit_id"]
    ready, support = terminal_ready(args.terminal_ready_record, directory, plan, guard)
    lock = run / "queue/cpu_monitor_large_artifact_read.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    info = os.fstat(descriptor)
    require(stat.S_ISREG(info.st_mode) and info.st_uid == 2618919, "ordinary owned CPU read lock required")
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    output = P / ("copies_" + str(args.copies) + "_terminal_review")
    try:
        require(not output.exists() and not output.is_symlink(), "preserve the previous actual audit attempt")
        output.mkdir()
        publish(output / "attempt.json", {"plan": inputs["copies"][str(args.copies)]["plan"],
            "terminal_ready": reference(args.terminal_ready_record), "compute": context,
            "source": reference(Path(__file__)), "at_seconds": time.time(), "scientific_credit": 0})
        try:
            result, expected, raw = actual_raw(directory, plan, qualified, guard, support)
            raw_ref = publish(output / "raw_validation.json", raw)
            review = {"schema_version": 1, "kind": "independent_actual_direct_common_configuration_correctness_review",
                "approved": True, "passed": True, **expected, "raw_audit": raw_ref, "independent_of_producer": True,
                "reviewer_identity": args.reviewer_identity, "reviewed_at_seconds": time.time(),
                "unresolved_anomalies": [], "admission_granted": False, "worker_admitted": False,
                **{name: 0 for name in guard.ZERO_FIELDS}}
            require(set(review) == guard.REVIEW_FIELDS
                and review["reviewed_at_seconds"] >= result["execution_identity"]["finished_at_seconds"],
                "the genuine actual review changed its exact field/time contract")
            review_ref = publish(output / "independent_actual_review.json", review)
            publish(output / "verification.json", {"passed": True, "raw": raw_ref, "review": review_ref,
                "compute": context, "actual_raw_pass_before_review": True, "scientific_credit": 0,
                "actual_future_timing_guard_probe_not_yet_run": True})
            print(json.dumps({"passed": True, "raw": raw_ref, "review": review_ref, "scientific_credit": 0}, sort_keys=True))
        except BaseException as error:
            publish(output / "failure.json", {"error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(), "approved": False, "scientific_credit": 0})
            raise
    finally:
        os.close(descriptor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
