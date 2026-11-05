"""Audit the actual initial empty command and later exact command of the same child."""
import ast
from datetime import datetime, timezone
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


def require(value,message):
    assert value,message
    CHECKS.append(message)


def read(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<2**20,"exact ordinary bounded artifact: "+path.name)
    raw=path.read_bytes()
    INPUTS[str(path)]={"path":str(path),"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)}
    return raw


def doc(path):
    return json.loads(read(path))


def sha(value):
    return hashlib.sha256(value).hexdigest()


def validate_identity(value,empty):
    require(value["pid"]==371111 and value["process_start_time"]=="84596712" and value["parent_pid"]==370657 and value["hostname"]=="serval02" and value["uid"]==2618919,"actual same child PID/birth/PPID/host/UID")
    require(sha(value["stat"].encode())==value["stat_sha256"],"exact raw child stat hash")
    fields=value["stat"].rsplit(")",1)[1].split()
    require(value["stat"].split(" ",1)[0]=="371111" and fields[19]=="84596712" and fields[1]=="370657","raw stat PID/birth/PPID match typed identity")
    require(re.fullmatch(r"Uid:\s*2618919\s+2618919\s+2618919\s+2618919\s*",value["uid_bytes"]) is not None and sha(value["uid_bytes"].encode())==value["uid_sha256"],"all four raw kernel UIDs match exact owner")
    raw=bytes.fromhex(value["command_bytes_hex"])
    require(sha(raw)==value["command_sha256"],"actual raw command bytes match their hash")
    require(value["affinity"]==list(range(8)),"exact actual eight selected CPUs")
    if empty:
        require(raw==b"" and value["command"]==[] and value["command_sha256"]=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","initial command is genuinely zero bytes and empty decoded argv")
    else:
        require(raw==b"\0".join(part.encode() for part in command)+b"\0" and value["command"]==command,"later actual command is exact expected argv and raw bytes")
    return fields


require(os.environ.get("SLURM_JOB_ID")=="6821437" and os.environ.get("SLURM_STEP_ID","").isdigit(),"current exact owned CPU audit job/step")
require(os.getresuid()==(2618919,)*3,"actual reviewer real/effective/saved UIDs")
first=doc(U/"direct_controller_initial_child_observation.json")
intent=doc(U/"direct_controller_intent.json")
spawn=doc(P/"root_actual_controller_spawn.json")
later=doc(P/"root_actual_same_child_command_observation.json")
command=intent["command"]
require(first["expected_command"]==later["expected_command"]==command and first["expected_affinity"]==intent["selected_cpu_affinity"]==list(range(8)),"actual first and later records bind same expected command and selected mask")
require(first["actual_popen_pid"]==371111 and first["expected_parent_pid"]==intent["controller"]["pid"]==spawn["actual_pid"]==370657 and intent["controller"]["process_start_time"]==spawn["actual_birth_counter"]=="84591278","actual root controller return and selected child snapshot identities join")
initial_fields=validate_identity(first["observed_child"],True)
final_fields=validate_identity(later["observed"],False)
recomputed={"parent_pid_matches":first["observed_child"]["parent_pid"]==370657,"command_matches":first["observed_child"]["command"]==command,"inherited_affinity_matches":first["observed_child"]["affinity"]==list(range(8)),"popen_pid_matches":first["observed_child"]["pid"]==371111}
require(first["predicate_results"]==recomputed and recomputed=={"parent_pid_matches":True,"command_matches":False,"inherited_affinity_matches":True,"popen_pid_matches":True} and first["all_predicates_match"] is False,"independently recomputed first refusal is exactly the empty command comparison")
require(first["registration_performed_by_this_observation"] is first["admission_granted"] is first["worker_admitted"] is False and first["scientific_credit"]==first["case_credit"]==first["idea_credit"]==0,"initial mismatch grants no registration/admission/credit")
require(later["original_process_present"] is True and later["pid"]==371111 and later["expected_birth"]=="84596712" and later["signals_sent"]==later["gpu_or_standing_lock_operations"]==0 and later["global_process_enumeration"] is False,"later root observation is exact owned same PID/birth without signal/GPU/global enumeration")
require(later["exact_command_now_matches"] is later["parent_still_matches"] is later["affinity_still_matches"] is True,"later exact raw identity agrees with all three saved typed matches")
require(datetime.fromisoformat(first["observed_child"]["observed_at_utc"])<datetime.fromisoformat(later["observed"]["observed_at_utc"]),"actual empty observation precedes later exact command of same birth identity")
require(initial_fields[0]=="D" and initial_fields[23]==initial_fields[24]=="0" and initial_fields[45]=="140733453932894" and initial_fields[46]=="0","raw first stat state/code fields and argument positions preserved literally; argument start is nonzero")
later_raw=doc(P/"root_actual_same_child_command_observation_stdout.raw")
require(later_raw==later and doc(P/"root_actual_same_child_command_observation_return.json")["returncode"]==0,"later saved root observation equals actual successful raw stdout")
read(P/"root_actual_same_child_command_observation_intent.json")
read(P/"root_actual_same_child_command_observation_stderr.raw")
source=read(P/"direct_qualification_controller.py").decode()
require(sha(source.encode())=="09a1c944116541b631e12e465f5e782641429e305e9a135da4dc8e5379f9a5b6","actual failed controller remains exact frozen observation source")
functions={node.name:node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)}
body=ast.get_source_segment(source,functions["supervise"])
require(body.index('publish(unit / "direct_controller_initial_child_observation.json"')<body.index('"actual child parent, argv or inherited CPU affinity differs"')<body.index('publish(unit / "direct_controller_spawn.json"')<body.index('append_registration'),"source records first observation then refuses empty command before any successful spawn/registration publication")
require('child_identity["command"] == command' in body and 'child_identity["parent_pid"] == os.getpid()' in body and 'child_identity["affinity"] == selected_affinity' in body,"original exact scientific parent/argv/affinity conditions remain required")
read_process=ast.get_source_segment(source,functions["read_process"])
require('if raw_command else []' in read_process and 'raw_command = (directory / "cmdline").read_bytes()' in read_process,"empty decoded command is the actual zero-byte proc read, not filtering or substituted argv")
require('wait_exact(child, child_identity' in body and body.index('wait_exact(child, child_identity')<body.index('publish(unit / "direct_controller_outcome.json"'),"actual controller waits for its exact child before publishing final outcome")
status_time=datetime.now(timezone.utc).isoformat()
current_status={name:(U/name).exists() for name in ("direct_controller_spawn.json","direct_controller_pre_registration_child_observation.json","direct_current_launch.json","direct_current_process_ledger_snapshot.json","direct_process_metadata.json","direct_process_wait_return.json","direct_controller_outcome.json","failure.json","result.json")}
require(not current_status["direct_controller_spawn.json"] and not current_status["direct_controller_pre_registration_child_observation.json"] and not current_status["direct_current_launch.json"] and not current_status["direct_current_process_ledger_snapshot.json"] and not current_status["result.json"],"no successful spawn, pre-registration, current launch, registration snapshot or scientific result")
for item in list(INPUTS.values()):
    require(sha(Path(item["path"]).read_bytes())==item["sha256"],"immutable actual input unchanged during audit: "+Path(item["path"]).name)
report={"schema_version":1,"kind":"independent_actual_initial_empty_command_then_same_child_exact_command_audit","passed":True,"current_attempt_passed":False,"actual_controller":{"pid":370657,"birth":"84591278"},"actual_child":{"pid":371111,"birth":"84596712","parent_pid":370657},"first_observed_at_utc":first["observed_child"]["observed_at_utc"],"later_observed_at_utc":later["observed"]["observed_at_utc"],"first_raw_command_hex":"","first_predicate_results_recomputed":recomputed,"later_exact_command_verified":True,"actual_time_command_became_ready":"unknown between the two actual observations; observation gap is not a measured startup duration","initial_raw_stat_details":{"state":initial_fields[0],"field26_start_code":initial_fields[23],"field27_end_code":initial_fields[24],"field48_argument_start":initial_fields[45],"field49_argument_end":initial_fields[46]},"original_earlier04_07_failure_cause_established":False,"actual_kernel_or_exec_mechanism_established":False,"current_status_at_utc":status_time,"current_saved_artifact_status":current_status,"completed_wait_or_both_exit_proof_claimed_by_this_audit":False,"initial_declared_producer_descriptor_role_is_actual_FLOCK_proof":False,"scientific_credit":0,"timing_credit":0,"worker_or_GPU_admission":False,"root_must_wait_for_exact_child_wait_and_both_process_exits_before_successor":True,"proposed_correction_review_requirements":["Separate fresh reviewed source only; preserve frozen05-10.","Bounded waiting may apply only to truly empty raw cmdline, under exact all-four-UID/PID/birth/PPID/affinity revalidation at every observation.","Unexpected nonempty argv, changed identity or affinity, child exit, read error or deadline must refuse.","The complete expected argv is still mandatory before any registration or admission.","Retain every initial/intermediate/final observation and actual wait; do not infer the cause of the older unrecorded04-07 snapshot."],"actual_owned_CPU_audit_context":{"job_id":os.environ["SLURM_JOB_ID"],"step_id":os.environ["SLURM_STEP_ID"],"hostname":socket.gethostname(),"pid":os.getpid()},"reviewer_actions":{"target_query_or_launch":False,"frozen_source_edit":False,"numerical_or_GPU_or_native_library_load":False},"inputs":INPUTS,"checks":CHECKS,"elapsed_seconds":time.monotonic()-START}
out=P/"independent_actual_empty_to_exact_command_audit.json"
with out.open("x") as stream:
    json.dump(report,stream,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
print(json.dumps({"passed":True,"checks":len(CHECKS),"step":os.environ["SLURM_STEP_ID"],"elapsed_seconds":report["elapsed_seconds"],"current_status":current_status,"review":{"path":str(out),"sha256":sha(out.read_bytes())}}))
