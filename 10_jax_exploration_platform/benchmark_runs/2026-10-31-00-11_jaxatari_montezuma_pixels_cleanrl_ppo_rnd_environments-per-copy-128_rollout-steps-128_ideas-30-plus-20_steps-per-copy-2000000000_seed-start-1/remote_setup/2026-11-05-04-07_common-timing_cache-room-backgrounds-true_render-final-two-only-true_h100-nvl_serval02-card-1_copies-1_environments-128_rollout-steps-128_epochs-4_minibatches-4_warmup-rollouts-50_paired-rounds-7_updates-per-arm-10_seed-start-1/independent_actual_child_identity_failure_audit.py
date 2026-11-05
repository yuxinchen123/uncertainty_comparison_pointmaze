"""Audit the saved initial-child refusal using source text and small actual records only."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import time

P = Path(__file__).resolve().parent
R = P.parent.parent
U = R / "data" / P.name
START = time.monotonic()
CHECKS = []
INPUTS = {}


def require(value, description):
    assert value, description
    CHECKS.append(description)


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "ordinary exact small input: " + path.name)
    raw = path.read_bytes()
    require(len(raw) <= 2**20, "bounded input: " + path.name)
    INPUTS[str(path)] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    return raw


def doc(path):
    return json.loads(read(path))


def function(source, name):
    tree = ast.parse(source)
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def call_lines(node, name):
    return [part.lineno for part in ast.walk(node) if isinstance(part, ast.Call) and (isinstance(part.func, ast.Name) and part.func.id == name or isinstance(part.func, ast.Attribute) and part.func.attr == name)]


require(os.environ.get("SLURM_JOB_ID") == "6821437" and os.environ.get("SLURM_STEP_ID", "").isdigit(), "exact current owned CPU allocation and numeric step")
require(os.getresuid() == (2618919,)*3, "reviewer real/effective/saved owner UID")
owned = (R / "slurm/submitted_jobids_uva.txt").read_text().split()
require("6821437" in owned, "current CPU allocation recorded in matching owned ledger")
plan = doc(U / "measurement_plan.json")
root_spawn = doc(P / "root_actual_controller_spawn.json")
intent = doc(U / "direct_controller_intent.json")
outcome = doc(U / "direct_controller_outcome.json")
refusal = doc(U / "direct_controller_failure.json")
child_failure = doc(U / "failure.json")
wait = doc(U / "direct_process_wait_return.json")
metadata = doc(U / "direct_process_metadata.json")
output = doc(U / "direct_controller_output_hashes.json")
progress = doc(U / "progress.json")
common_return = doc(U / "direct_common_correctness_validation_return.json")
common = doc(U / "direct_common_correctness_validation.json")
require(root_spawn["actual_pid"] == 351185 and root_spawn["actual_birth_counter"] == "84237458", "actual controller PID and birth from root Popen record")
require(outcome["passed"] is False and outcome["actual_child_started"] is True and outcome["actual_wait_saved"] is True, "failed controller did start and wait for its child")
message = "actual child parent, argv or inherited CPU affinity differs"
require(refusal["error"] == outcome["failure"]["error"] == message and refusal["type"] == "ValueError", "actual first combined identity refusal retained without relabeling")
require(refusal["actual_child_pid"] == wait["pid"] == wait["waited_pid"] == 351650 and wait["process_start_time"] == "84242978", "actual returned numerical-producer PID/birth joined to exact wait")
require(wait["returncode"] == 1 and wait["exited_normally"] is True and wait["timeouts"] == [] and wait["kind"] == "owned_direct_process_wait_return", "normal exact child wait returned one without intervention or timeout")
require(wait["waiter_before"]["pid"] == wait["waiter_after"]["pid"] == 351185, "same actual controller saved exact wait")
require(datetime.fromisoformat(refusal["at_utc"]) < datetime.fromisoformat(wait["finished_at_utc"]), "initial controller refusal precedes actual failed-child wait")
for key in ("stdout", "stderr", "wait"):
    expected = output[key]
    read(expected["path"])
    require(INPUTS[expected["path"]]["sha256"] == expected["sha256"], "actual output/wait bytes match controller saved hash: " + key)
require(child_failure["exception_type"] == "FileNotFoundError" and "direct_current_launch.json" in child_failure["traceback"] and "collect_metadata" in child_failure["traceback"] and "launch_inputs" in child_failure["traceback"], "child failed in metadata on missing current launch record")
require(metadata["capture_error"]["type"] == "FileNotFoundError" and metadata["before"]["identity"]["pid"] == 351650 and metadata["before"]["identity"]["process_start_time"] == "84242978", "partial producer metadata belongs to the exact failed child")
require(metadata["before"]["identity"]["parent_pid"] == 351185 and metadata["before"]["identity"]["cpu_affinity"] == intent["selected_cpu_affinity"] == list(range(8)), "later child parent and affinity match; these are not substituted for missing initial values")
require(metadata["lock_before"]["fd_target"] == "/tmp/sl5nw_serval02_card1.lock" and "WRITE 351650 " in metadata["lock_before"]["fdinfo"], "producer actually held the approved standing FLOCK before metadata refusal")
require(common_return["returncode"] == 0 and common["passed"] is True and common["parent_pid"] == 351650 and common["numerical_imports"] == [] and common["gpu_operations"] == 0, "actual isolated prior-correctness metadata check completed without numerical imports or recorded GPU operations")
require(common["stdlib_ctypes_bootstrap"]["own_process_python_api_initialized_before_validation_audit"] is True, "existing own-process standard-library ctypes bootstrap counted explicitly")
require(progress["state"] == "failed" and progress["pid"] == 351650, "producer progress is failed, not completed or training")
absent = ["direct_controller_spawn.json", "direct_current_launch.json", "direct_current_process_ledger_snapshot.json", "direct_terminal_process.json", "result.json", "direct_device_binding.json", "direct_device_binding_attempt.json", "direct_startup_checks.json", "asset_receipt.json", "installed_backend.json", "direct_backend_capture_begin.json", "direct_backend_capture.json"]
for name in absent:
    require(not (U / name).exists() and not (U / (name + ".prepared")).exists(), "no published or partial downstream artifact: " + name)
ledger = doc(R / "processes.json")
require(not any(row.get("unit") == P.name or row.get("hostname") == "serval02" and row.get("pid") == 351650 and row.get("start_time") == "84242978" for row in ledger), "no actual current-process registration for this failed unit")
controller_source = read(P / "direct_qualification_controller.py").decode()
selected_source = read(U / "code/measure_selected.py").decode()
binding_source = read(U / "code/serval02_card1_binding.py").decode()
freeze = doc(P / "controller_source_freeze.json")
require(INPUTS[str(P / "direct_qualification_controller.py")]["sha256"] == freeze["files"][str(P / "direct_qualification_controller.py")], "actual failed controller remains exact frozen source")
for name in ("measure_selected.py", "serval02_card1_binding.py"):
    relative = "10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name
    require(INPUTS[str(U / "code" / name)]["sha256"] == plan["helpers"]["files"][relative], "actual failed staged helper source remains exact plan byte: " + name)
supervise = function(controller_source, "supervise")
text = ast.get_source_segment(controller_source, supervise)
require(text.index("child_identity = read_process") < text.index(message) < text.index('publish(unit / "direct_controller_spawn.json"') < text.index("append_registration"), "source refuses combined condition before saving observed identity/spawn or registration")
require('child_identity["parent_pid"] == os.getpid()' in text and 'child_identity["command"] == command' in text and 'child_identity["affinity"] == selected_affinity' in text, "all three original parent/argv/affinity conditions remain required")
require('child_identity = read_process' in text and 'process_start_time' in ast.get_source_segment(controller_source, function(controller_source,"read_process")), "successful initial read produced birth identity later used for exact wait")
run = function(selected_source, "run_prepared")
require(max(call_lines(run,"collect_metadata")) < min(call_lines(run,"_compute")), "producer collect_metadata occurs before any scientific _compute call")
compute = function(selected_source, "_compute")
require(any(isinstance(node, ast.Import) and any(alias.name == "jax" for alias in node.names) for node in ast.walk(compute)), "explicit JAX import is inside later unentered _compute")
collect = function(binding_source, "collect_metadata")
collect_text = ast.get_source_segment(binding_source, collect)
require(collect_text.index('value["lock_before"] = lock_metadata') < collect_text.index('value["current_launch"] = launch_inputs'), "partial FLOCK observation occurs before missing-launch refusal")
require('direct admission must precede numerical imports' in collect_text and 'publish(directory / "direct_process_metadata.json", value)' in collect_text, "producer requires no numerical imports and preserves partial metadata on failure")
exit_record = doc(P / "root_failed_identity_startup_exact_exit_observation.json")
require(exit_record["all_original_processes_ended"] is True and exit_record["signals_sent"] == 0 and {(row["pid"],row["expected_birth_counter"]) for row in exit_record["observations"]} == {(351185,"84237458"),(351650,"84242978")} and all(row["proc_entry_absent"] and row["original_process_ended"] for row in exit_record["observations"]), "root separately observed both exact original identities ended without signals")
restored = doc(P / "root_metadata_lock_permission_restore.json")
require(restored["restored"] is True and restored["standing_gpu_lock_touched"] is False and restored["matching_process_ledger_rows"] == 0 and all(row["mode"] == 0o660 and row["bytes_unchanged"] for row in restored["after"]), "root separately restored metadata permissions after closure without standing-lock operation")
readonly = doc(P / "root_readonly_current_controller_children_return.json")
require(readonly["returncode"] == 1 and b"/proc/351185" in read(P / "root_readonly_current_controller_children_stderr.raw"), "failed read-only probe retained as failure, not an exit proof")
for path, item in INPUTS.items():
    require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == item["sha256"], "actual bounded input unchanged during independent audit: " + Path(path).name)
report = {
 "schema_version": 1,
 "kind": "independent_actual_initial_child_identity_failure_audit",
 "audit_passed": True,
 "scientific_attempt_passed": False,
 "actual_controller": {"pid":351185,"birth":"84237458"},
 "actual_child": {"pid":351650,"birth":"84242978","normal_wait_status":1,"wait_finished_at_utc":wait["finished_at_utc"]},
 "controller_refusal": refusal,
 "initial_parent_argv_affinity_observation_saved": False,
 "exact_initial_failed_subcondition": "unproved",
 "later_parent_and_affinity_match_expected": True,
 "later_observation_proves_initial_values": False,
 "explicit_scientific_compute_reached": False,
 "operation_accounting": {"actual_scientific_producer_processes_started":1,"actual_isolated_prior_correctness_metadata_check_completed":1,"standing_lock_acquired_by_actual_producer":True,"normal_exact_waits":1,"controller_timeouts_or_signals":0,"environment_transitions":0,"learner_updates":0,"scientific_timing_rounds":0,"scientific_credit":0,"case_credit":0,"idea_credit":0,"admission_granted":False,"explicit_producer_JAX_or_CUDA_stage_reached":False,"whole_process_native_or_device_work":"unquantified; interpreter startup and declared own-process ctypes bootstrap are not represented as zero"},
 "source_recommendation": ["Preserve this failed packet and source; never relaunch it.","In a separate reviewed source, save the actual initial read_process result and the three predicate outcomes before a combined refusal, including intended command, selected affinity, parent PID, all UIDs, birth and time.","Keep exact parent, stable birth, command and affinity checks. Do not replace an unexpected command with an assumed exec-transition explanation.","Use one smallest owned CPU-only selected-private-interpreter child that records its own image/argv/UID/birth/parent/affinity and waits before any numerical/device imports. Parent must save its first actual observation and exact wait. An explicit before-work acknowledgment can distinguish incomplete startup from an actual mismatch, but requires separate source review.","Do not add arbitrary sleeps or accept a mismatched command; no new scientific or GPU qualification is inferred from a CPU-only result."],
 "initial_exec_transition_hypothesis_established":False,
 "inputs":INPUTS,
 "actual_owned_cpu_context":{"job_id":os.environ["SLURM_JOB_ID"],"step_id":os.environ["SLURM_STEP_ID"],"pid":os.getpid(),"hostname":socket.gethostname(),"uids":list(os.getresuid()),"cgroup":Path("/proc/self/cgroup").read_text()},
 "reviewer_actions":{"numerical_or_GPU_imports":False,"device_or_native_library_operations":False,"remote_or_process_queries":False,"frozen_source_edits":False,"raw_historical_payload_replay":False},
 "checks":CHECKS,"elapsed_seconds":time.monotonic()-START,"at_utc":datetime.now(timezone.utc).isoformat()
}
out=P / "independent_actual_child_identity_failure_audit.json"
with out.open("x") as stream:
    json.dump(report,stream,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
print(json.dumps({"audit_passed":True,"checks":len(CHECKS),"step":os.environ["SLURM_STEP_ID"],"elapsed_seconds":report["elapsed_seconds"],"review":{"path":str(out),"sha256":hashlib.sha256(out.read_bytes()).hexdigest()}}))
