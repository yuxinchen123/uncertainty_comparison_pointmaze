"""Add actual normal failed wait, exact exits and metadata restoration to the existing audit."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time

P=Path(__file__).resolve().parent
R=P.parent.parent
U=R/"data"/P.name
START=time.monotonic()
CHECKS=[]
INPUTS={}

def require(value,text):
    assert value,text
    CHECKS.append(text)

def read(path):
    path=Path(path);raw=path.read_bytes()
    require(path.is_file() and not path.is_symlink() and len(raw)<2**20,"ordinary bounded exact artifact: "+path.name)
    INPUTS[str(path)]={"path":str(path),"sha256":hashlib.sha256(raw).hexdigest()}
    return raw

def doc(path):return json.loads(read(path))

def digest(value):return hashlib.sha256(value).hexdigest()

require(os.environ.get("SLURM_JOB_ID")=="6821437" and os.environ.get("SLURM_STEP_ID","").isdigit() and os.getresuid()==(2618919,)*3,"current exact owned CPU job/step/owner")
prior=doc(P/"independent_actual_empty_to_exact_command_audit.json")
require(INPUTS[str(P/"independent_actual_empty_to_exact_command_audit.json")]["sha256"]=="0ba3446e2eea7b761c97937a6ee99de0f3eb26f3078ab0d6d6a9db0086ffb929" and prior["passed"] is True,"existing49-check initial/transition audit remains exact; not rerun")
wait=doc(U/"direct_process_wait_return.json")
outcome=doc(U/"direct_controller_outcome.json")
controller_failure=doc(U/"direct_controller_failure.json")
child_failure=doc(U/"failure.json")
metadata=doc(U/"direct_process_metadata.json")
outputs=doc(U/"direct_controller_output_hashes.json")
exit_proof=doc(P/"root_failed_empty_command_exact_exit_observation.json")
restored=doc(P/"root_metadata_lock_permission_restore.json")
require(wait["pid"]==wait["waited_pid"]==371111 and wait["process_start_time"]=="84596712" and wait["hostname"]=="serval02" and wait["uid"]==2618919,"actual exact child PID/birth/host/owner join wait")
require(type(wait["returncode"]) is int and wait["returncode"]==1 and wait["exited_normally"] is True and wait["timeouts"]==[] and wait["kind"]=="owned_direct_process_wait_return","actual Popen.wait returned normal failure one without intervention")
for name in ("waiter_before","waiter_after"):
    owner=wait[name];fields=owner["stat"].rsplit(")",1)[1].split()
    require(owner["pid"]==370657 and fields[19]=="84591278" and digest(owner["stat"].encode())==owner["stat_sha256"] and digest(owner["uid_bytes"].encode())==owner["uid_sha256"] and re.fullmatch(r"Uid:\s*2618919\s+2618919\s+2618919\s+2618919\s*",owner["uid_bytes"]) is not None,"actual same owner controller birth/stat/allUIDs saved around exact child wait")
require(outcome["passed"] is False and outcome["actual_child_started"] is outcome["actual_wait_saved"] is True and controller_failure["actual_child_pid"]==371111 and controller_failure["actual_wait_saved"] is True,"actual final controller outcome remains failed with real child and saved wait")
require(outcome["failure"]["error"]==controller_failure["error"]=="actual child parent, argv or inherited CPU affinity differs","original first empty-command refusal remains the controller failure")
require(datetime.fromisoformat(outcome["failure"]["at_utc"])<datetime.fromisoformat(wait["finished_at_utc"])<datetime.fromisoformat(exit_proof["at_utc"])<=datetime.fromisoformat(restored["at_utc"]),"actual refusal, normal wait, both exits and permission restore chronology")
for name in ("stdout","stderr","wait"):
    value=outputs[name]
    require(digest(read(value["path"]))==value["sha256"],"actual exact-child output/wait hashes match terminal controller record: "+name)
require(child_failure["exception_type"]=="FileNotFoundError" and "direct_current_launch.json" in child_failure["traceback"] and "collect_metadata" in child_failure["traceback"],"actual producer failed in pre-compute metadata on missing current launch")
require(metadata["capture_error"]["type"]=="FileNotFoundError" and metadata["before"]["identity"]["pid"]==371111 and metadata["before"]["identity"]["process_start_time"]=="84596712" and metadata["before"]["identity"]["parent_pid"]==370657,"actual partial process metadata belongs to same failed child")
lock=metadata["lock_before"]
require(lock["path"]==lock["fd_target"]=="/tmp/sl5nw_serval02_card1.lock" and lock["fd_stat"]==lock["path_stat"] and lock["fd_stat"]["uid"]==2618919 and lock["fd_stat"]["regular_file"] is True and "WRITE 371111 " in lock["fdinfo"] and digest(lock["fdinfo"].encode())==lock["fdinfo_sha256"],"new actual partial metadata now proves actual child standing-FLOCK custody; prior role field alone did not")
require(exit_proof["all_original_processes_ended"] is True and exit_proof["signals_sent"]==exit_proof["gpu_or_standing_lock_operations"]==0 and exit_proof["global_process_enumeration"] is False,"actual separate exact exit observation used no signals, standing-lock operation or global enumeration")
require({(row["pid"],row["expected_birth_counter"]) for row in exit_proof["observations"]}=={(370657,"84591278"),(371111,"84596712")} and all(row["proc_entry_absent"] and row["original_process_ended"] for row in exit_proof["observations"]),"actual root observed both exact original process identities ended")
require(restored["restored"] is True and restored["actual_child_wait_returncode"]==1 and restored["both_exact_processes_ended"] is restored["no_registration"] is True and restored["standing_gpu_lock_touched"] is False,"actual metadata restore follows failed wait/both exits without standing-lock intervention")
require({row["path"] for row in restored["after"]}=={str(R/"queue/metadata.lock"),str(R/"processes.lock")} and all(row["mode"]==0o660 and row["bytes_unchanged"] for row in restored["after"]),"only exact owner metadata paths restored to0660 with unchanged bytes")
before=doc(P/"root_metadata_lock_permission_before.json")
protected=doc(P/"root_metadata_lock_permission_protected.json")
require(before["observation_host"]==protected["observation_host"]=="portal05" and before["files"]==protected["files"] and protected["actual_permission_change_performed"] is True,"same-host original/protected metadata identities preserved by actual root records")
for old in before["files"]:
    new=next(row for row in restored["after"] if row["path"]==old["path"])
    require(new["device"]==old["device"] and new["inode"]==old["inode"] and old["uid"]==2618919 and new["mode"]==old["old_mode"]==0o660 and old["protected_mode"]==0o640 and new["bytes_unchanged"] is True,"root same-host actual before/after metadata device/inode/owner/mode and bytes join")
ledger=doc(R/"processes.json")
require(not any(row.get("unit")==P.name or row.get("hostname")=="serval02" and row.get("pid")==371111 and row.get("start_time")=="84596712" for row in ledger),"failed unit has no live registration row")
for name in ("direct_controller_spawn.json","direct_controller_pre_registration_child_observation.json","direct_current_launch.json","direct_current_process_ledger_snapshot.json","direct_terminal_process.json","result.json","direct_device_binding.json","direct_backend_capture_begin.json"):
    require(not (U/name).exists(),"no downstream successful or numerical evidence: "+name)
for path,value in INPUTS.items():
    if path!=str(R/"processes.json"):
        require(digest(Path(path).read_bytes())==value["sha256"],"actual terminal artifact unchanged during addendum: "+Path(path).name)
report={"schema_version":1,"kind":"independent_initial_empty_command_actual_terminal_wait_exit_addendum","passed":True,"attempt_remains_failed":True,"initial_transition_audit":INPUTS[str(P/"independent_actual_empty_to_exact_command_audit.json")],"actual_controller":{"pid":370657,"birth":"84591278"},"actual_child":{"pid":371111,"birth":"84596712","normal_exact_wait_returncode":1,"finished_at_utc":wait["finished_at_utc"]},"both_exact_process_exits_verified":True,"metadata_restore_after_actual_wait_and_both_exits_verified":True,"actual_child_standing_FLOCK_proof_now_available":True,"original_earlier04_07_cause_established":False,"actual_command_ready_time_known":False,"scientific_or_timing_credit":0,"GPU_or_worker_admission":False,"environment_transitions":0,"learner_updates":0,"whole_interpreter_native_work":"unquantified","reviewer_target_or_process_or_GPU_or_NUM_action":False,"actual_owned_CPU_context":{"job_id":os.environ["SLURM_JOB_ID"],"step_id":os.environ["SLURM_STEP_ID"],"hostname":socket.gethostname(),"pid":os.getpid()},"inputs":INPUTS,"checks":CHECKS,"elapsed_seconds":time.monotonic()-START}
out=P/"independent_actual_empty_command_terminal_addendum.json"
with out.open("x") as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
print(json.dumps({"passed":True,"checks":len(CHECKS),"step":os.environ["SLURM_STEP_ID"],"elapsed_seconds":report["elapsed_seconds"],"review":{"path":str(out),"sha256":digest(out.read_bytes())}}))
