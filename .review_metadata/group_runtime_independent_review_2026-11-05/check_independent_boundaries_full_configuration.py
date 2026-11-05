"""Check specific group-reader boundaries using only private metadata fixtures."""
import ctypes
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

WORKTREE = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent
SCRIPTS = WORKTREE / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
FREEZE = WORKTREE / ".review_metadata/group_runtime_completion/source_freeze_for_independent_review.json"


def digest(path):
    """Hash an ordinary source or metadata fixture without executing native code."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    """Write one new reviewer-owned record and retain all previous attempts."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def audit(event, values):
    """Permit private Git fixture commands and forbid tools, schedulers and numerical imports."""
    if event == "subprocess.Popen":
        executable, command = values[:2]
        if executable != "git" or not isinstance(command, (list, tuple)) or command[0] != "git":
            raise RuntimeError("only private local Git fixture commands are permitted")
    if event in {"os.system", "os.posix_spawn", "socket.connect", "socket.bind", "socket.getaddrinfo", "ctypes.dlopen", "ctypes.dlsym"}:
        raise RuntimeError("network, shell, tool, device and later library operations are forbidden")
    if event == "import" and str(values[0]).split(".")[0] in {
            "jax", "jaxlib", "numpy", "torch", "jaxtari", "cupy", "pynvml", "flax", "optax", "scipy"}:
        raise RuntimeError("numerical imports are forbidden in independent metadata cases")


def full_configurations(runtime):
    """Use complete source-derived configurations with one real implementation switch changed."""
    base = runtime.producer.selected.diagnostic.default_configuration(SCRIPTS.parents[2])
    reference = {name: base[name] for name in ("learner", "environment")}
    reference["learner"]["n_copies"] = 1
    reference["execution"] = {"compilation_mode": "whole_iteration", "iterations_per_call": 1,
        "report_readout": "none", "report_every_iterations": 10}
    reference = json.loads(json.dumps(reference))
    reference["environment"]["cache_room_backgrounds"] = False
    candidate = copy.deepcopy(reference)
    candidate["environment"]["cache_room_backgrounds"] = True
    runtime.producer.selected.validation.configuration(reference)
    runtime.producer.selected.validation.configuration(candidate)
    return candidate, reference


def make_correctness_case(runtime, evidence):
    """Call the real member reader with empty correctness reports after valid upstream placeholders."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        unit = root / "data/group"
        worker = unit / "workers/worker_1"
        worker.mkdir(parents=True)
        configuration, reference = full_configurations(runtime)
        plan = {"campaign_run_dir": str(root), "unit_id": "group", "mode": "correctness", "gpu_count": 1,
            "configuration": configuration, "reference_configuration": reference, "copy_seeds": [[1]],
            "source_commit": "0" * 40, "helpers": {}, "runtime_environment": {}, "asset_record": {}}
        metadata = {"process": {"pid": 101, "parent_pid": 100, "process_start_time": "123", "cgroup": "synthetic"}}
        binding = {"gpu_uuid": "GPU-synthetic", "process": metadata["process"]}
        for path, value in ((unit / "scaling_plan.json", plan), (worker / "group_step_metadata.json", metadata),
                (worker / "resolved_device_binding.json", binding), (worker / "group_process_identity.json", {})):
            save(path, value)
        result = {"schema_version": 2, "kind": evidence.package.MODES["correctness"], "completed": True,
            "configuration": configuration, "reference_configuration": reference, "copy_seeds": [1],
            "source_commit": plan["source_commit"], "helpers": {}, "runtime_environment": {}, "asset_record": {},
            "worker_index": 0, "group_metadata": evidence.group.receipt(worker / "group_step_metadata.json"),
            "resolved_device_binding": binding, "scientific_credit": 0, "admission_granted": False,
            "correctness_attempted": True, "correctness_passed": True, "implementation_distinct": True,
            "measurement_valid": False, "correctness": {"initialization": {}, "passed": True,
                "supplied_inputs": {}, "complete_iterations": 1, "complete_iteration": {}}}
        save(worker / "result.json", result)
        # Other gates represent successful source/hardware/package checks; the real comparison predicate is not replaced.
        def adjacent(name):
            """Return only the two declared reader dependencies used by this isolated predicate case."""
            if name == "group_runtime":
                return runtime
            if name == "configured_scaling_evidence":
                return SimpleNamespace(verify_sources=lambda *args: None)
            raise ValueError("unexpected dependency in the independent comparison case: " + name)

        with patch.object(evidence, "require_owned_compute"), patch.object(runtime, "validate_plan"), \
                patch.object(evidence.group, "adjacent", side_effect=adjacent), \
                patch.object(evidence.configured, "verify_qualification_source_receipts"), \
                patch.object(evidence.package, "worker_index", return_value=0), \
                patch.object(evidence.group, "source_identity"), patch.object(evidence.group, "validate_binding"), \
                patch.object(runtime, "numeric_command", return_value=[]), patch.object(runtime, "validate_process"), \
                patch.object(evidence.package, "completed_work"), patch.object(evidence, "verify_library_evidence"), \
                patch.object(evidence, "validate_controlled_window"):
            returned = evidence.verify_saved_member(worker, terminal=False)[0]
        original_refused = False
        try:
            runtime.producer.selected.validation.correctness(returned)
        except ValueError:
            original_refused = True
        return {"new_reader_accepted": returned == result, "unchanged_standard_reader_refused": original_refused,
            "empty_supplied_input_reports": True, "empty_complete_iteration_report": True,
            "scope": "actual correctness predicate; unrelated upstream successful gates are synthetic placeholders"}


def prerequisite_seed_case(runtime, evidence):
    """Show whether a genuinely different prior seed can satisfy the real physical-member predicate."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        configuration, reference = full_configurations(runtime)
        plan = {"configuration": configuration, "reference_configuration": reference,
            "copy_seeds": [[1]], "source_commit": "0" * 40, "helpers": {}, "runtime_environment": {},
            "asset_record": {}, "qualification_software": {}}
        previous = {**copy.deepcopy(plan), "mode": "correctness", "copy_seeds": [[99]]}
        result = {"copy_seeds": [99]}
        binding = {"gpu_uuid": "GPU-synthetic"}
        path = root / "result.json"
        save(path, result)
        item = {"result": evidence.group.receipt(path)}
        review = {"schema_version": 2, "kind": evidence.REVIEW_KIND, "approved": True,
            "result": item["result"], "mode": "correctness", "job_id": "123456",
            "actual_terminal_state": "COMPLETED", "actual_exit_code": "0:0", "admission_granted": False,
            "scientific_credit": 0}
        save(root / "review.json", review)
        item["review"] = evidence.group.receipt(root / "review.json")
        expected_package = {name: {} for name in ("selected_python", "site_packages", "image", "distribution")}
        with patch.object(evidence, "verify_saved_member", return_value=(result, previous, binding)), \
                patch.object(evidence.package, "expected_for_plan", return_value=expected_package), \
                patch.object(evidence.group, "owned_inputs", return_value={"job_id": "123456"}):
            returned = evidence.reviewed_member(item, "correctness", plan, {"index": 0, "gpu_uuid": binding["gpu_uuid"]})
        return {"accepted": returned == result, "current_worker_seeds": [1], "prior_verified_worker_seeds": [99],
            "scope": "actual prerequisite join; full prior-result and package validation represented as successful"}


def current_compute_case(evidence):
    """Compare the real cgroup to a different synthetic claimed job without querying that job."""
    actual = Path("/proc/self/cgroup").read_text()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "slurm").mkdir()
        (root / "slurm/submitted_jobids_uva.txt").write_text("987654\n")
        with patch.dict(os.environ, {"SLURM_JOB_ID": "987654", "SLURM_STEP_ID": "123"}):
            evidence.require_owned_compute({"campaign_run_dir": str(root)})
        return {"accepted": True, "claimed_job": "987654", "actual_cgroup": actual,
            "actual_job": "6783003", "queried_synthetic_job": False,
            "scope": "the real guard read only environment and the synthetic private ledger"}


def wrapper_controller_case(runtime, fixture):
    """Keep wrapper records normal while substituting the saved controller's source and ownership fields."""
    plan = fixture.plan
    directory = runtime.group.unit_directory(plan)
    (directory / "wrapper_records").mkdir()
    controller = {"job_id": "unrelated", "plan": {"path": "/synthetic/unrelated-plan"},
        "helpers": {"source_commit": "f" * 40}, "process": {"pid": 5000, "uids": [0] * 4,
            "argv": ["/synthetic/unrelated-controller"], "cgroup": "unrelated job", "process_start_time": "not-a-birth"}}
    save(directory / "group_job_controller.json", controller)
    with patch.object(runtime.group, "owned_inputs", return_value={"job_id": "123456"}):
        for number, (role, index) in enumerate([("coordinator", None), *[("worker", x) for x in range(plan["gpu_count"])]]):
            name = "coordinator" if role == "coordinator" else "worker_" + str(index + 1)
            command = runtime.producer.step_command(plan, "123456", role, runtime.numeric_command(plan, role, index))
            process = {"pid": 6000 + number, "parent_pid": 5000, "process_start_time": str(1000 + number),
                "uids": [2618919] * 4, "argv": command, "cgroup": "another unrelated job"}
            save(directory / "wrapper_records" / (name + "_started.json"), {"role": role, "index": index,
                "command": command, "process": process, "is_numerical_process": False})
            save(directory / "wrapper_records" / (name + "_wait.json"), {"process": process, "returncode": 0,
                "terminated_for_failure": False, "is_numerical_wait": False})
        returned = runtime.verify_wrapper_waits(plan)
    return {"accepted": returned == controller, "controller_uids": controller["process"]["uids"],
        "controller_plan_and_source_mismatched": True, "wrapper_cgroups_unrelated": True,
        "scope": "actual wrapper reader; literal per-role commands and zero waits remain internally consistent"}


def main():
    """Run bounded source-only cases in the exact owned CPU allocation and preserve all outcomes."""
    if os.environ.get("SLURM_JOB_ID") != "6783003" or not os.environ.get("SLURM_STEP_ID", "").isdigit():
        raise ValueError("the independent cases require owned CPU allocation6783003 and a numeric step")
    if os.getuid() != os.geteuid() or os.getuid() != 2618919:
        raise ValueError("the actual campaign owner is required")
    expected = json.loads(FREEZE.read_text())["files"]
    before = {name: digest(name) for name in expected}
    if before != expected:
        raise ValueError("preserve the frozen source before independent cases")
    sys.addaudithook(audit)
    sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(SCRIPTS / "tests"))
    import group_runtime as runtime
    import group_runtime_evidence as evidence
    import test_group_runtime as tests
    cases = {}
    cases["empty_correctness_reports"] = make_correctness_case(runtime, evidence)
    cases["different_prerequisite_seeds"] = prerequisite_seed_case(runtime, evidence)
    cases["claimed_compute_job_not_actual_cgroup"] = current_compute_case(evidence)
    tests.RuntimePlanTests.setUpClass()
    fixture = tests.RuntimePlanTests("test_runtime_valid_full_measurement_plan")
    try:
        fixture.setUp()
        cases["unrelated_saved_controller"] = wrapper_controller_case(runtime, fixture)
    finally:
        fixture.doCleanups()
        tests.RuntimePlanTests.tearDownClass()
    after = {name: digest(name) for name in expected}
    if before != after:
        raise ValueError("a frozen source changed during independent cases")
    record = {"kind": "independent_group_runtime_boundary_observations", "cases": cases,
        "source_freeze_sha256": digest(FREEZE), "all34frozen_files_unchanged": True,
        "actual_job": "6783003", "actual_step": os.environ["SLURM_STEP_ID"], "actual_pid": os.getpid(),
        "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stdlib_bootstrap_before_audit": "ctypes imports the own-process Python API; later library operations refused",
        "actual_tool_or_gpu_or_numerical_or_scheduler_operations": 0, "scientific_credit": 0}
    save(DIRECTORY / "independent_boundary_observations_full_configuration.json", record)
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
