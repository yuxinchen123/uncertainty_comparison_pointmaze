"""Audit the actual failed metadata startup without repeating or contacting the target."""
import ast
import ctypes
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

P = Path(__file__).resolve().parent
R = P.parent.parent
U = R / "data" / P.name
NUM = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "cupy", "scipy", "jaxtari", "exploration_platform"}
CHECKS = []
START = time.monotonic()
BOOTSTRAP = {"stdlib_ctypes_own_process_python_api_initialized_before_audit": True, "python_api_target": None,
             "scope": "ordinary stdlib import required by the exact metadata binding helper; no external library target"}


def require(condition, statement):
    """Record each successful factual audit assertion."""
    assert condition, statement
    CHECKS.append(statement)


def reference(path):
    """Hash one small ordinary metadata/source input and retain its CPU-local identity."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "ordinary audit input: " + str(path))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def document(path):
    """Read one small actual JSON record."""
    return json.loads(Path(path).read_text())


def guard(event, args):
    """Refuse numerical imports and every remote, subprocess, library-load or GPU-device operation."""
    if event == "import" and args[0].split(".")[0] in NUM:
        raise RuntimeError("numerical import forbidden in failure audit")
    if event in {"socket.connect", "socket.bind", "ctypes.dlopen", "subprocess.Popen"}:
        raise RuntimeError("remote, process or library operation forbidden in failure audit")
    if event == "open" and isinstance(args[0], (str, bytes)) and os.fsdecode(args[0]).startswith("/dev/nvidia"):
        raise RuntimeError("GPU device access forbidden in failure audit")


# Declare the small standard-library bootstrap, then audit only exact saved metadata.
sys.addaudithook(guard)
require(os.environ.get("SLURM_JOB_ID") == "6821437" and os.environ.get("SLURM_STEP_ID", "").isdigit(), "current exact owned CPU step")
require(os.getresuid() == (2618919, 2618919, 2618919), "actual reviewer UID values")
require((R / "slurm/submitted_jobids_uva.txt").read_text().splitlines().count("6821437") == 1, "current parent occurs exactly once in campaign ledger")
plan = document(U / "measurement_plan.json")
spawn = document(P / "root_actual_controller_spawn.json")
remote_intent = document(P / "root_remote_launch_intent.json")
remote_return = document(P / "root_remote_launch_return.json")
intent = document(U / "direct_controller_intent.json")
failure = document(U / "direct_controller_failure.json")
outcome = document(U / "direct_controller_outcome.json")
logs = R / "slurm/direct_jobs" / P.name
require(remote_return["returncode"] == 0 and remote_return["local_wait_timed_out"] is False and remote_return["no_automatic_retry"] is True, "actual transport returned zero separately from the failed benchmark")
require(spawn["kind"] == "actual_controller_transport_popen_return" and spawn["actual_pid"] == 321183 and spawn["actual_birth_counter"] == "83855111" and spawn["host"] == "serval02" and spawn["uid"] == 2618919, "actual saved controller PID/birth/host/owner")
require(spawn["actual_numerical_child_pid"] is None and spawn["metadata_barrier_held_through_record_fsync"] is True, "real root spawn receipt contains no invented numerical child")
controller = intent["controller"]
require(controller["pid"] == spawn["actual_pid"] and controller["process_start_time"] == spawn["actual_birth_counter"] and controller["hostname"] == "serval02" and controller["uid"] == 2618919, "actual controller intent joins real root transport identity")
require(intent["plan"] == reference(U / "measurement_plan.json") and intent["actual_child_pid"] is None, "actual failed startup joins unchanged prepared plan")
require(failure["type"] == outcome["failure"]["type"] == "KeyError" and failure["error"] == outcome["failure"]["error"] == "'admission'", "same exact actual failure in both terminal metadata records")
require(outcome["passed"] is False and outcome["actual_child_started"] is False and outcome["actual_wait_saved"] is False and failure["actual_child_pid"] is None, "actual numerical Popen and numerical wait never occurred")
require(document(logs / "controller.stdout") == outcome and (logs / "controller.stderr").read_bytes() == b"", "actual controller output exactly matches terminal outcome")
chronology = [datetime.datetime.fromisoformat(value) for value in [remote_intent["at_utc"], spawn["at_utc"], remote_return["at_utc"], intent["at_utc"], failure["at_utc"]]]
require(chronology == sorted(chronology), "real transport/spawn/return/controller-intent/failure chronology")
absent = ["direct_controller_spawn.json", "direct_current_launch.json", "direct_current_process_ledger_snapshot.json", "direct_process_metadata.json", "direct_device_binding.json", "direct_process_wait_return.json", "direct_terminal_process.json", "progress.json", "hardware.json", "result.json"]
require(not any((U / name).exists() for name in absent), "no numerical spawn/registration/device/result/progress artifacts")
history_path = U / "direct_history/inputs.json"
history = document(history_path)
require("admission" not in history and "accepted_evidence_snapshot_inputs" in history, "new history is an accepted-snapshot schema, not legacy admission history")
require(plan["helpers"]["source_commit"] == "f7e8f5c8f0a2d1847fff4f93726fce62205c23aa" and plan["source_commit"] == "d28e6735e79788f618acb0ebfde8609db1f776af", "actual failed helper and numerical source identities")
binding_path = U / "code/serval02_card1_binding.py"
relative = "10_jax_exploration_platform/scripts/atari_pixel_campaign/serval02_card1_binding.py"
require(reference(binding_path)["sha256"] == plan["helpers"]["files"][relative], "exact actual staged binding source")
source = binding_path.read_text()
functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
claim_ast = ast.dump(functions["current_admission_claim"], include_attributes=False)
require('inputs["admission"]' in ast.get_source_segment(source, functions["current_admission_claim"]), "actual claim producer unconditionally accesses missing legacy admission key")
require('"accepted_evidence_snapshot_protocol" in plan' in ast.get_source_segment(source, functions["validate_history"]), "actual history validator already supports the new snapshot protocol")

# Reproduce the exact claim-producer error, never the live host authority check or numerical producer.
spec = importlib.util.spec_from_file_location("actual_failed_binding_claim_metadata_audit", binding_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
reproduced = None
try:
    module.current_admission_claim(plan)
except KeyError as error:
    reproduced = str(error)
    actual_traceback = traceback.format_exc()
require(reproduced == "'admission'", "actual unchanged claim producer reproduces exact observed missing key on owned CPU")
require(not NUM.intersection(sys.modules), "no numerical modules imported")
inputs = [P / name for name in ("root_actual_controller_spawn.json", "root_remote_launch_intent.json", "root_remote_launch_return.json")] + [U / name for name in ("measurement_plan.json", "direct_controller_intent.json", "direct_controller_failure.json", "direct_controller_outcome.json")] + [history_path, binding_path, logs / "controller.stdout", logs / "controller.stderr"]
audit = {"schema_version": 1, "kind": "independent_actual_direct_timing_metadata_startup_failure_audit", "passed": True, "benchmark_passed": False,
         "actual_owned_cpu_job_id": os.environ["SLURM_JOB_ID"], "actual_cpu_step_id": os.environ["SLURM_STEP_ID"], "reviewer_hostname": socket.gethostname(),
         "actual_saved_controller_pid": spawn["actual_pid"], "actual_saved_controller_birth": spawn["actual_birth_counter"], "actual_controller_exit_verified_by_this_audit": False,
         "actual_numerical_child_started": False, "environment_transitions": 0, "training_updates": 0, "actual_failure": outcome["failure"],
         "metadata_reproduction": {"error": reproduced, "traceback": actual_traceback, "exact_claim_function_ast_sha256": hashlib.sha256(claim_ast.encode()).hexdigest()},
         "saved_history_keys": sorted(history), "absent_numerical_artifacts": absent, "exact_raw_inputs": [reference(path) for path in inputs],
         "stdlib_bootstrap": BOOTSTRAP, "host_live_admission_check_executed_on_cpu": False, "target_queries": 0, "gpu_or_device_operations": 0,
         "required_fix": ["Add strict new-snapshot claim production at the actual shared current_admission_claim entry used by controller, process metadata and parent readers.", "Retain the exact original admission hash, numerical source, host/owner/UUID/standing-descriptor/current-CUDA and unchanged live check gates.", "Exercise the real current claim producer through CPU preflight so this operational step cannot be missed by a default-only reader.", "Publish reviewed source and use a fresh unattempted unit only after exact prior controller exit; never retry or rewrite the failed unit."],
         "frozen_failed_unit_reexecuted": False, "large_payloads_or_original_numerical_gates_reexecuted": False, "checks": CHECKS,
         "elapsed_seconds": time.monotonic() - START, "scientific_credit": 0, "case_credit": 0, "idea_credit": 0, "worker_admission": False}
out = P / "independent_actual_startup_failure_audit.json"
with out.open("x") as handle:
    json.dump(audit, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps({"passed": True, "benchmark_passed": False, "audit": reference(out), "checks": len(CHECKS), "elapsed_seconds": audit["elapsed_seconds"], "actual_cpu_step_id": os.environ["SLURM_STEP_ID"]}))
