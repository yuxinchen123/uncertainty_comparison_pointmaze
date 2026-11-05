"""Verify actual bounded startup and registration only, without querying a GPU or running a process test."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

P=Path(__file__).resolve().parent
R=P.parent.parent
I=R.parents[2]
U=R/"data"/P.name
INNER="c88f4b1a4111919066f80cb81701041b00b43bad"
OUTER="ba07a961eeb00f368db70416fd69a9a9313df9b9"
START=time.monotonic()
CHECKS=[]
INPUTS={}


def require(value,message):
    assert value,message
    CHECKS.append(message)


def read(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<2**20,"ordinary bounded exact input: "+path.name)
    raw=path.read_bytes();INPUTS[str(path)]={"path":str(path),"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)}
    return raw


def doc(path):return json.loads(read(path))
def sha(raw):return hashlib.sha256(raw).hexdigest()


def identity(value,pid,birth,parent=None,expected_command=None):
    require(value["pid"]==pid and value["process_start_time"]==birth and value["hostname"]=="serval02" and value["uid"]==2618919,"exact actual direct process PID/birth/host/owner")
    fields=value["stat"].rsplit(")",1)[1].split()
    require(value["stat"].split(" ",1)[0]==str(pid) and fields[19]==birth and fields[1]==str(value["parent_pid"]) and sha(value["stat"].encode())==value["stat_sha256"],"actual raw stat/hash matches PID/birth/PPID")
    require(re.fullmatch(r"Uid:\s*2618919\s+2618919\s+2618919\s+2618919\s*",value["uid_bytes"]) is not None and sha(value["uid_bytes"].encode())==value["uid_sha256"],"all four raw kernel UIDs and their hash match owner")
    raw=bytes.fromhex(value["command_bytes_hex"])
    decoded=[part.decode() for part in raw.rstrip(b"\0").split(b"\0")] if raw else []
    require(decoded==value["command"] and sha(raw)==value["command_sha256"],"actual raw command bytes/hash agree with decoded record")
    require(parent is None or value["parent_pid"]==parent,"exact actual child PPID")
    if expected_command is not None:require(value["command"]==expected_command,"actual recorded decoded command equals expected command")
    return raw


require(os.environ.get("SLURM_JOB_ID")=="6821437" and os.environ.get("SLURM_STEP_ID","").isdigit() and os.getresuid()==(2618919,)*3,"current exact owned CPU audit allocation/step/owner")
plan=doc(U/"measurement_plan.json")
root_spawn=doc(P/"root_actual_controller_spawn.json")
intent=doc(U/"direct_controller_intent.json")
initial=doc(U/"direct_controller_initial_child_observation.json")
sequence=doc(U/"direct_controller_startup_observation_sequence.json")
spawn=doc(U/"direct_controller_spawn.json")
prereg=doc(U/"direct_controller_pre_registration_child_observation.json")
launch=doc(U/"direct_current_launch.json")
publication=doc(P/"root_actual_transport_publication_verification.json")
proposal=doc(P/"proposed_root_once_ssh_command.json")
remote_intent=doc(P/"root_remote_launch_intent.json")
remote_return=doc(P/"root_remote_launch_return.json")
restore=doc(P/"root_metadata_lock_permission_restore.json")
before=doc(P/"root_metadata_lock_permission_before.json")
protected=doc(P/"root_metadata_lock_permission_protected.json")
require(INPUTS[str(U/"measurement_plan.json")]["sha256"]=="4dd7ea9dcc7f70ebc85170780b36c9564790017cd06ce7d3fbf8e1b45a7d952c","exact bound actual scientific plan")
require(publication["outer_publication"]==publication["remote_head"]==remote_intent["outer_publication"]==OUTER and publication["inner_publication"]==remote_intent["inner_publication"]==INNER and publication["all_nine_source_and_log_records_exact_published_bytes"] is True and publication["actual_transport_review_verified"] is True,"actual published outer/inner source bytes verified before once-only root transport")
require(remote_intent["argv"]==proposal["one_shot_ssh_argv"] and remote_intent["one_attempt"] is True and remote_intent["standing_gpu_lock_operations_by_root"]==0 and remote_return["returncode"]==0 and remote_return["timed_out"] is remote_return["retry_permitted"] is False,"actual single reviewed root SSH returned zero without timeout/retry or standing lock")
read(P/"root_remote_launch_stdout.raw");read(P/"root_remote_launch_stderr.raw")
transport_freeze=doc(P/"root_transport_source_freeze.json")
for path,expected in transport_freeze["files"].items():
    require(sha(read(path))==expected["sha256"],"actual held transport file unchanged: "+Path(path).name)
    committed=subprocess.check_output(["git","-C",str(I),"cat-file","blob",OUTER+":"+str(Path(path).relative_to(I))],timeout=20)
    require(sha(committed)==expected["sha256"],"outer publication contains exact actual transport/log record: "+Path(path).name)
controller=read(P/"direct_qualification_controller.py")
require(sha(controller)=="8e567998fc5209f5ff4370338fe62bf78391b7c5637adf989572b42486adcb0e","actual executed strict empty-only controller source remains exact")
committed=subprocess.check_output(["git","-C",str(I),"cat-file","blob",INNER+":"+str((P/"direct_qualification_controller.py").relative_to(I))],timeout=20)
require(committed==controller,"inner publication contains exact executed bounded-startup controller")
require(root_spawn["actual_pid"]==intent["controller"]["pid"]==386202 and root_spawn["actual_birth_counter"]==intent["controller"]["process_start_time"]=="84810007","actual returned controller PID/birth joins own controller intent")
identity(root_spawn["actual_controller_identity"],386202,"84810007")
identity(intent["controller"],386202,"84810007")
command=intent["command"]
expected_raw=b"\0".join(part.encode() for part in command)+b"\0"
require(initial["actual_popen_pid"]==spawn["popen_pid"]==sequence["actual_popen_pid"]==prereg["actual_popen_pid"]==386669 and sequence["initial_birth"]=="84815419" and sequence["expected_parent_pid"]==386202,"actual Popen child and stable first birth join all startup records")
require(sequence["expected_command"]==initial["expected_command"]==prereg["expected_command"]==spawn["command"]==command and sequence["expected_affinity"]==intent["selected_cpu_affinity"]==list(range(8)) and sequence["expected_uid"]==2618919,"actual startup expected argv, selected eight CPUs and owner are consistent")
require(sequence["exact_command_observed"] is True and sequence["failure"] is None and sequence["registration_performed_by_this_observer"] is False and sequence["maximum_wait_seconds"]==30 and sequence["deadline_monotonic"]==sequence["started_monotonic"]+30,"actual startup observer completed exact-command readiness under one fixed30second deadline without registering itself")
rows=sequence["observations"]
require(len(rows)==2 and rows[0]["observed_child"]==initial["observed_child"] and rows[-1]["observed_child"]==spawn["observed_child"],"actual initial empty snapshot and later ready snapshot are the exact saved raw observations")
last_time=sequence["started_monotonic"]
for index,row in enumerate(rows):
    raw=identity(row["observed_child"],386669,"84815419",386202)
    observed=row["observed_child"]
    require(observed["affinity"]==list(range(8)),"same exact selected affinity at every startup observation")
    predicates={"all_four_uids_match":re.fullmatch(r"Uid:\s*2618919\s+2618919\s+2618919\s+2618919\s*",observed["uid_bytes"]) is not None and type(observed["uid"]) is int and observed["uid"]==2618919 and sha(observed["uid_bytes"].encode())==observed["uid_sha256"],"popen_pid_matches":type(observed["pid"]) is int and observed["pid"]==386669,"birth_matches_first_observation":observed["process_start_time"]=="84815419","parent_pid_matches":type(observed["parent_pid"]) is int and observed["parent_pid"]==386202,"inherited_affinity_matches":observed["affinity"]==list(range(8)),"raw_command_matches_decoded_record":sha(raw)==observed["command_sha256"]}
    require(row["identity_predicates"]==predicates and all(predicates.values()),"all actual raw UID/PID/birth/PPID/affinity/command-hash predicates independently recomputed")
    require(row["expected_command_matches"]==(observed["command"]==command) and row["expected_raw_command_matches"]==(raw==expected_raw) and row["raw_command_is_empty"]==(raw==b""),"actual raw and decoded command predicates independently recomputed")
    require(last_time<=row["observed_monotonic"]<=sequence["deadline_monotonic"] and row["actual_popen_returncode"] is None,"same fixed monotonic deadline and exact-child-live poll at each observation")
    last_time=row["observed_monotonic"]
    if index==0:
        require(raw==b"" and observed["command"]==[] and row["after_empty_poll"]["actual_popen_returncode"] is None,"actual zero-byte first command waited while the exact child remained present")
    else:require(raw==expected_raw and observed["command"]==command,"ready observation requires complete exact expected raw argv")
require(last_time<=sequence["finished_monotonic"]<=sequence["deadline_monotonic"] and 0<=sequence["finished_monotonic"]-sequence["started_monotonic"]<30,"actual bounded startup observation completed within30seconds")
ready_raw=identity(spawn["observed_child"],386669,"84815419",386202,command)
identity(spawn["controller"],386202,"84810007")
prereg_raw=identity(prereg["observed_child"],386669,"84815419",386202,command)
require(prereg["all_predicates_match"] is True and all(prereg["predicate_results"].values()) and prereg["observed_child"]["affinity"]==list(range(8)),"actual unchanged pre-registration decoded-command/birth/parent/affinity gate passed")
require(ready_raw==expected_raw and prereg_raw==expected_raw,"actual recorded ready and pre-registration raw argv both match in this run; production second gate remains decoded-only")
require(launch["plan"]=={"path":str(U/"measurement_plan.json"),"sha256":INPUTS[str(U/"measurement_plan.json")]["sha256"]},"actual immutable current launch joins exact plan")
row=launch["process_row"]
expected_row={"hostname":"serval02","gpu_uuid":"GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3","kind":plan["kind"],"unit":P.name,"source_commit":"d28e6735e79788f618acb0ebfde8609db1f776af","source_root":plan["source_root"],"helper_commit":"69f1b5075530418adbe02c1e433c5647848a61ba","pid":386669,"start_time":"84815419"}
require(row==expected_row and row["unit"]==plan["unit_id"] and row["unit"]!=str(U),"actual registration row uses immutable unit name and exact PID/birth/source/helper/UUID")
ledger=launch["process_ledger"]
require(ledger["source_path"]==str(R/"processes.json") and ledger["snapshot"]["path"]==str(U/"direct_current_process_ledger_snapshot.json"),"actual current registration names exact live process ledger and immutable snapshot")
snapshot=doc(ledger["snapshot"]["path"])
require(INPUTS[ledger["snapshot"]["path"]]["sha256"]==ledger["snapshot"]["sha256"],"actual ledger snapshot hash matches current launch")
live=doc(R/"processes.json")
for records in (snapshot,live):
    require(sum(item==row for item in records)==1 and sum(item.get("unit")==P.name for item in records)==1 and sum(item.get("hostname")=="serval02" and item.get("pid")==386669 and item.get("start_time")=="84815419" for item in records)==1,"exactly one actual process row matches unit name/PID/birth in snapshot and live ledger")
require(before["observation_host"]==protected["observation_host"]=="portal05" and before["files"]==protected["files"] and protected["actual_permission_change_performed"] is True and protected["standing_gpu_lock_touched"] is False,"root protected same owned metadata files before actual transport, without standing descriptor")
require(datetime.fromisoformat(publication["at_utc"])<=datetime.fromisoformat(protected["at_utc"])<=datetime.fromisoformat(remote_intent["at_utc"])<datetime.fromisoformat(prereg["at_utc"])<=datetime.fromisoformat(restore["at_utc"]),"actual source verification, metadata protection, launch, pre-registration and restore chronology")
require(restore["restored"] is True and restore["exact_registered_controller_pid"]==386202 and restore["exact_registered_child_pid"]==386669 and restore["matching_actual_process_rows"]==1 and restore["standing_gpu_lock_touched"] is False,"actual metadata permission restoration followed exact controller/child registration")
for old in before["files"]:
    new=next(value for value in restore["after"] if value["path"]==old["path"])
    require(new["device"]==old["device"] and new["inode"]==old["inode"] and new["mode"]==old["old_mode"]==0o660 and old["protected_mode"]==0o640 and new["bytes_unchanged"] is True,"root same-host owned metadata device/inode/mode/bytes preserved through restore")
status={name:(U/name).exists() for name in ("direct_process_metadata.json","direct_device_binding.json","direct_startup_checks.json","progress.json","direct_process_wait_return.json","direct_controller_outcome.json","failure.json","result.json")}
for path,item in INPUTS.items():
    if path!=str(R/"processes.json"):require(sha(Path(path).read_bytes())==item["sha256"],"immutable actual startup evidence unchanged during audit: "+Path(path).name)
report={"schema_version":1,"kind":"independent_actual_bounded_empty_startup_and_single_registration_review","passed":True,"startup_and_registration_accepted":True,"scientific_result_evaluated":False,"scientific_or_timing_credit":0,"new_GPU_or_worker_or_hardware_qualification":False,"actual_outer_publication":OUTER,"actual_inner_publication":INNER,"actual_controller":{"pid":386202,"birth":"84810007"},"actual_child":{"pid":386669,"birth":"84815419","parent_pid":386202},"actual_empty_to_exact_ready_observation_seconds":sequence["finished_monotonic"]-sequence["started_monotonic"],"ready_sequence_rows":len(rows),"complete_raw_argv_verified_at_startup_ready":True,"actual_recorded_prereg_raw_argv_also_matches":True,"production_prereg_gate_contract":"unchanged decoded-command/birth/parent/affinity; no second raw-command equality gate claimed","actual_registration_row":row,"matching_unit_name_and_PID_rows_in_live_and_snapshot":1,"actual_metadata_restore_after_registration_verified":True,"actual_GPU_setup_or_CUDA_binding_evaluated":False,"saved_artifact_presence_at_audit":status,"actual_final_child_wait_or_end_evaluated":False,"initial_declared_standing_descriptor_role_is_actual_custody_proof":False,"earlier_failed_attempt_causes_inferred":False,"reviewer_actions":{"source_or_process_tests_repeated":False,"remote_or_GPU_query_or_launch":False,"NUM_or_device_or_native_library_loads":False},"actual_owned_CPU_audit_context":{"job_id":os.environ["SLURM_JOB_ID"],"step_id":os.environ["SLURM_STEP_ID"],"hostname":socket.gethostname(),"pid":os.getpid()},"inputs":INPUTS,"checks":CHECKS,"elapsed_seconds":time.monotonic()-START,"at_utc":datetime.now(timezone.utc).isoformat()}
out=P/"independent_actual_startup_registration_review.json"
with out.open("x") as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
print(json.dumps({"passed":True,"checks":len(CHECKS),"step":os.environ["SLURM_STEP_ID"],"elapsed_seconds":report["elapsed_seconds"],"saved_artifact_presence":status,"review":{"path":str(out),"sha256":sha(out.read_bytes())}}))
