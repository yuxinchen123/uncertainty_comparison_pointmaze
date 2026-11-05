"""Independently test corrected group-reader predicates on owned CPU with no numerical work."""
import ctypes
import copy
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent
SCRIPTS = ROOT / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
FREEZE = ROOT / ".review_metadata/group_runtime_four_predicate_corrections/corrected_source_freeze_for_independent_review.json"


def digest(path):
    """Hash a source or metadata record without loading native code."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(event, args):
    """Allow only private Git fixture commands and refuse numerical or device operations."""
    if event == "subprocess.Popen":
        if args[0] != "git" or not isinstance(args[1], (list, tuple)) or args[1][0] != "git":
            raise RuntimeError("only local Git fixture subprocesses are allowed")
    if event in {"os.system", "os.posix_spawn", "socket.connect", "socket.bind", "socket.getaddrinfo", "ctypes.dlopen", "ctypes.dlsym"}:
        raise RuntimeError("network, shell, GPU, tool and later library operations are forbidden")
    if event == "import" and str(args[0]).split(".")[0] in {
            "jax", "jaxlib", "numpy", "torch", "jaxtari", "cupy", "pynvml", "flax", "optax", "scipy"}:
        raise RuntimeError("numerical imports are forbidden")


def load_file(name, path):
    """Load exact metadata fixture functions without invoking their command-line entry."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    """Check positive fixtures and unrelated false records using actual frozen predicates."""
    if os.environ.get("SLURM_JOB_ID") != "6783003" or not os.environ.get("SLURM_STEP_ID", "").isdigit() or os.getuid() != os.geteuid() or os.getuid() != 2618919:
        raise ValueError("use only this campaign owner's admitted numeric CPU step")
    expected = json.loads(FREEZE.read_text())["files"]
    before = {name: digest(name) for name in expected}
    if before != expected:
        raise ValueError("preserve every frozen source byte")
    sys.addaudithook(audit)
    sys.path.insert(0, str(SCRIPTS)); sys.path.insert(0, str(SCRIPTS / "tests"))
    import group_runtime as runtime
    import group_runtime_evidence as evidence
    import test_group_runtime_predicate_corrections as fixtures
    old = load_file("independent_old_group_boundaries", ROOT / ".review_metadata/group_runtime_independent_review_2026-11-05/check_independent_boundaries_full_configuration.py")
    cases = []

    def refused(name, call, required_text=None):
        """Require the named actual predicate to refuse the changed metadata value."""
        try:
            call()
        except (ValueError, KeyError) as error:
            if required_text is not None and required_text not in str(error):
                raise AssertionError("wrong predicate rejected " + name + ": " + str(error)) from error
            cases.append({"name": name, "passed": True, "refused": True, "error_type": type(error).__name__, "error": str(error)})
        else:
            cases.append({"name": name, "passed": False, "refused": False})

    # Re-run the original empty-report exploit without changing its real target function.
    refused("original_actual_saved_member_empty_reports", lambda: old.make_correctness_case(runtime, evidence), "actual fixed-input")
    refused("original_actual_kernel_job_substitution", lambda: old.current_compute_case(evidence), "kernel cgroup")

    # Before: worker index1 needs global seed2. After: matching2 passes; foreign1 and typed substitutions fail.
    candidate, reference = fixtures.configurations()
    for seed in (2, 1, 2.0, True):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan = {"gpu_count": 8, "configuration": candidate, "reference_configuration": reference,
                "copy_seeds": [[index + 1] for index in range(8)], "source_commit": "0" * 40,
                "helpers": {}, "runtime_environment": {}, "asset_record": {}, "qualification_software": {}}
            previous = {**copy.deepcopy(plan), "gpu_count": 1, "copy_seeds": [[seed]], "mode": "correctness"}
            result, binding = {"copy_seeds": [seed]}, {"gpu_uuid": "GPU-synthetic-index1"}
            (directory / "result.json").write_text(json.dumps(result))
            item = {"result": evidence.group.receipt(directory / "result.json")}
            review = {"schema_version": 2, "kind": evidence.REVIEW_KIND, "approved": True,
                "result": item["result"], "mode": "correctness", "job_id": "123456",
                "actual_terminal_state": "COMPLETED", "actual_exit_code": "0:0", "admission_granted": False, "scientific_credit": 0}
            (directory / "review.json").write_text(json.dumps(review)); item["review"] = evidence.group.receipt(directory / "review.json")
            installed = {name: {} for name in ("selected_python", "site_packages", "image", "distribution")}
            with patch.object(evidence, "verify_saved_member", return_value=(result, previous, binding)), patch.object(evidence.package, "expected_for_plan", return_value=installed), patch.object(evidence.group, "owned_inputs", return_value={"job_id": "123456"}):
                call = lambda: evidence.reviewed_member(item, "correctness", plan, {"index": 1, **binding})
                if type(seed) is int and seed == 2:
                    assert call() == result
                    cases.append({"name": "positive_physical_index1_global_seed2", "passed": True})
                else:
                    refused("physical_index1_different_seed_" + repr(seed), call, "global independent-copy seeds")

    # Keep all positive source/allocation scaffolding, then change one process/custody relationship at a time.
    fixtures.SavedProcessCustodyTests.setUpClass()
    fixture = fixtures.SavedProcessCustodyTests("runTest")
    try:
        fixture.setUp()
        assert fixture.verify() == fixture.controller
        cases.append({"name": "positive_complete_controller_wrapper_numeric_chain", "passed": True})
        for name, field, replacement in [("controller_boolean_schema", "schema_version", True),
                ("controller_nan_start", "started_at_seconds", float("nan")),
                ("controller_changed_numerical_source", "source_commit", "f" * 40)]:
            value = copy.deepcopy(fixture.controller); value[field] = replacement
            with patch.object(runtime.group, "owned_inputs", return_value={"job_id": fixture.job}):
                refused(name, lambda value=value: runtime.validate_saved_controller(fixture.plan, value))
        for name, field, replacement in [("controller_foreign_saved_uid", "uids", [2618919, 2618919, 0, 2618919]),
                ("controller_job_in_kernel_path_changed", "cgroup", fixture.controller["process"]["cgroup"].replace("job_123456", "job_987654")),
                ("controller_batch_role_changed", "cgroup", fixture.controller["process"]["cgroup"].replace("step_batch", "step_0")),
                ("controller_boolean_birth", "process_start_time", True)]:
            value = copy.deepcopy(fixture.controller); value["process"][field] = replacement
            with patch.object(runtime.group, "owned_inputs", return_value={"job_id": fixture.job}):
                refused(name, lambda value=value: runtime.validate_saved_controller(fixture.plan, value))
        row = fixture.rows[1]
        start_path = fixture.unit / "wrapper_records" / (row["name"] + "_started.json")
        wait_path = fixture.unit / "wrapper_records" / (row["name"] + "_wait.json")
        intent_path = fixture.unit / "wrapper_records" / (row["name"] + "_intent.json")
        original_intent = json.loads(intent_path.read_text())
        for name, field, replacement in [("coherent_wrapper_foreign_job", "cgroup", row["start"]["process"]["cgroup"].replace("job_123456", "job_987654")),
                ("coherent_wrapper_numeric_cgroup", "cgroup", row["start"]["process"]["cgroup"].replace("step_batch", "step_1")),
                ("coherent_wrapper_same_as_controller_pid", "pid", fixture.controller["process"]["pid"]),
                ("coherent_wrapper_false_parent", "parent_pid", fixture.controller["process"]["pid"] + 1)]:
            start, wait = copy.deepcopy(row["start"]), copy.deepcopy(row["wait"])
            start["process"][field] = replacement; wait["process"] = copy.deepcopy(start["process"])
            fixture.write(start_path, start); wait["started"] = runtime.group.receipt(start_path); fixture.write(wait_path, wait)
            refused(name, fixture.verify)
        fixture.write(start_path, row["start"]); fixture.write(wait_path, row["wait"])
        for name, field, replacement in [("wrapper_intent_retry_enabled", "automatic_retry_permitted", True),
                ("wrapper_intent_wrong_helpers", "helpers", {}), ("wrapper_intent_reversed_start", "at_seconds", 5.0)]:
            intent = copy.deepcopy(original_intent); intent[field] = replacement; fixture.write(intent_path, intent)
            refused(name, fixture.verify)
        fixture.write(intent_path, original_intent)
        for name, field, replacement in [("wrapper_wait_boolean_return", "returncode", False),
                ("wrapper_wait_wrong_job", "job_id", "987654"),
                ("wrapper_wait_after_intent_before_start", "waited_at_seconds", 2.5)]:
            wait = copy.deepcopy(row["wait"]); wait[field] = replacement; fixture.write(wait_path, wait)
            refused(name, fixture.verify)
        fixture.write(wait_path, row["wait"])
        # A consistent metadata PID change must still fail when it relabels a wrapper as a numerical task.
        numeric_path = row["directory"] / "group_process_identity.json"
        metadata_path = row["directory"] / "group_step_metadata.json"
        numeric, metadata = copy.deepcopy(row["numeric"]), copy.deepcopy(row["metadata"])
        numeric["pid"] = row["start"]["process"]["pid"]; metadata["process"]["pid"] = numeric["pid"]
        fixture.write(numeric_path, numeric); fixture.write(metadata_path, metadata)
        refused("numeric_pid_relabelled_from_its_wrapper", fixture.verify, "relabelled as a numerical process")
    finally:
        fixture.doCleanups(); fixtures.SavedProcessCustodyTests.tearDownClass()
    after = {name: digest(name) for name in expected}
    record = {"kind": "independent_corrected_group_runtime_boundaries", "schema_version": 2,
        "passed": before == after and all(row["passed"] for row in cases), "cases": cases,
        "actual_job": os.environ["SLURM_JOB_ID"], "actual_step": os.environ["SLURM_STEP_ID"], "actual_pid": os.getpid(),
        "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "source_freeze": {"path": str(FREEZE), "sha256": digest(FREEZE)},
        "source_unchanged": before == after, "frozen_file_count": len(expected),
        "unrelated_upstream_gates": "explicit successful metadata placeholders only, actual named predicates unchanged",
        "stdlib_bootstrap_before_audit": "ctypes initializes its own-process Python API before the audit; later library operations refused",
        "actual_numerical_gpu_device_tool_scheduler_operations": 0, "scientific_credit": 0}
    path = DIRECTORY / "independent_corrected_boundary_receipt.json"
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"passed": record["passed"], "cases": len(cases), "step": record["actual_step"], "receipt_sha256": digest(path)}))
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
