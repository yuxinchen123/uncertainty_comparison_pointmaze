"""Audit only actual saved eight-CPU diagnostic evidence, source and publication bytes."""
from datetime import datetime, timezone
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
PUB="4f42f257553865008a2a2f6b0ce9dd235a895ca6"
COMMIT="e16c015ac7c9ca674f768240cad35a372eb18e87"
START=time.monotonic()
CHECKS=[]
INPUTS={}


def require(value,message):
    assert value,message
    CHECKS.append(message)


def read(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<2**20,"ordinary bounded exact artifact: "+path.name)
    raw=path.read_bytes()
    INPUTS[str(path)]={"path":str(path),"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw)}
    return raw


def doc(path):
    return json.loads(read(path))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(value,*,pid,parent=None,birth=None,command=None,affinity=None):
    require(value["pid"]==pid and value["hostname"]=="serval02" and value["uid"]==2618919,"actual owned direct-host identity PID/host/UID")
    stat_raw=value["stat"].encode()
    require(sha(stat_raw)==value["stat_sha256"] and stat_raw.decode().split(" ",1)[0]==str(pid),"actual stat bytes/hash and PID join")
    fields=value["stat"].rsplit(")",1)[1].split()
    require(fields[19]==value["process_start_time"] and fields[1]==str(value["parent_pid"]) and value["process_start_time"].isdigit(),"actual birth and parent join raw kernel stat")
    require(re.fullmatch(r"Uid:\s*2618919\s+2618919\s+2618919\s+2618919\s*",value["uid_bytes"]) is not None and sha(value["uid_bytes"].encode())==value["uid_sha256"],"all four actual kernel owner UIDs and raw hash")
    raw=bytes.fromhex(value["command_bytes_hex"])
    require(raw==b"\0".join(part.encode() for part in value["command"])+b"\0" and sha(raw)==value["command_sha256"],"actual full command raw bytes and hash join decoded argv")
    require(parent is None or value["parent_pid"]==parent,"exact actual parent PID matches")
    require(birth is None or value["process_start_time"]==birth,"exact actual stable birth matches")
    require(command is None or value["command"]==command,"exact actual observed command matches expected")
    require(affinity is None or value["affinity"]==affinity,"exact actual observed CPU affinity matches selected mask")


require(os.environ.get("SLURM_JOB_ID")=="6821437" and os.environ.get("SLURM_STEP_ID","").isdigit(),"current exact owned CPU audit allocation/step")
require(os.getresuid()==(2618919,)*3 and "6821437" in (R/"slurm/submitted_jobids_uva.txt").read_text().split(),"actual owner and matching campaign CPU ledger")
origin=doc(P/"diagnostic_origin.json")
result=doc(P/"diagnostic_result.json")
first=doc(P/"immediate_child_observation.json")
last=doc(P/"acknowledged_child_observation.json")
intent=doc(P/"root_actual_execution_intent.json")
returned=doc(P/"root_actual_execution_return.json")
verification=doc(P/"root_actual_source_publication_verification.json")
proposal=doc(P/"proposed_one_shot_command.json")
final=doc(P/"final_standard_record_freeze.json")
require(INPUTS[str(P/"final_standard_record_freeze.json")]["sha256"]=="f4627ddc3f82ec7afeb1463d0522b3d8ca58d8f23e53d66f8cd33788dd812c4b","actual pre-execution final standard-record freeze")
addendum=doc(P/"independent_final_standard_record_addendum.json")
require(INPUTS[str(P/"independent_final_standard_record_addendum.json")]["sha256"]=="27d174373729fb6601c2181ff0773492218c1385a20c3a6b7aae5309e50ecb32" and addendum["passed"] is True,"genuine final standard-record independent approval")
require(verification["publication_commit"]==verification["remote_head"]==returned["publication_commit"]==intent["publication_commit"]==PUB and verification["actual_source_commit"]==returned["actual_source_commit"]==final["actual_source_commit"]==COMMIT,"actual publication/source receipts join exact executed records")
require(verification["source_and_all_final_records_exact_published_bytes"] is True and verification["source_worktree_tracked_files_clean"] is True and verification["approved_command_and_addendum_verified"] is True and verification["target_attempted"] is False,"root verified published source/record bytes and clean checkout before its one attempt")
require(intent["argv"]==proposal["one_shot_ssh_argv"] and intent["one_attempt"] is True and returned["returncode"]==0,"actual once-only SSH argv equals reviewed command and returned zero")
require(datetime.fromisoformat(verification["at_utc"])<=datetime.fromisoformat(intent["at_utc"])<datetime.fromisoformat(returned["finished_at_utc"]),"actual publication verification precedes intent and successful SSH return")
require(0<returned["wall_seconds"]<300 and result["elapsed_seconds"]<returned["wall_seconds"],"actual complete SSH and diagnostic durations are bounded")
require(json.loads(read(P/"root_actual_execution_stdout.raw"))==result,"actual SSH stdout exactly equals saved diagnostic result")
stderr=read(P/"root_actual_execution_stderr.raw").decode()
require(stderr=="/u/sl5nw/.bashrc: line 55: conda: command not found\n","actual remote shell startup warning retained separately; child diagnostic stderr remains empty")
subprocess.run(["git","-C",str(I),"merge-base","--is-ancestor",COMMIT,PUB],check=True,timeout=20)
for name,value in final["records"].items():
    require(sha(read(value["path"]))==value["sha256"],"current final standard record unchanged: "+name)
    path=Path(value["path"])
    committed=subprocess.check_output(["git","-C",str(I),"cat-file","blob",PUB+":"+str(path.relative_to(I))],timeout=20)
    require(sha(committed)==value["sha256"],"actual publication contains exact final standard record: "+name)
for relative,value in final["source_blobs"].items():
    committed=subprocess.check_output(["git","-C",str(I),"cat-file","blob",PUB+":"+relative],timeout=20)
    require(sha(committed)==value["sha256"] and len(committed)==value["bytes"],"actual publication contains exact reviewed source/test blob: "+Path(relative).name)
require(result["passed"] is True and result["failure"] is None and result["requested_cpu_count"]==origin["requested_cpu_count"]==8 and origin["selected_affinity"]==list(range(8)),"actual completed diagnostic used eight selected nonexclusive CPUs")
require(result["first_observation"]==first and result["acknowledged_observation"]==last,"separate actual observation files equal saved result fields")
require(result["actual_popen_pid"]==first["actual_popen_pid"]==last["actual_popen_pid"]==365739 and first["expected_parent_pid"]==last["expected_parent_pid"]==origin["parent"]["pid"]==365721,"actual parent and returned child PIDs join all observations")
command=origin["command"]
image="c889543c0e008c1362b69c711330cac598833a790dc0c48c5ee4a8567d3f3b51"
source="25876e545a9791b15c0ec649882da3a4accc9bcc56788e4f7a28911f0ad763ba"
controller="09a1c944116541b631e12e465f5e782641429e305e9a135da4dc8e5379f9a5b6"
require(command[:4]==["/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python","-I","-S","-B"] and command[5]=="--ack-child" and command[7:9]==["--expected-controller-sha256",controller] and command[9:]==["--work-end",str(origin["work_end_monotonic"])],"actual selected-interpreter acknowledgment command has exact fixed image, flags and work deadline")
require(origin["source_sha256"]==result["source_sha256"]==source and origin["controller_sha256"]==result["controller_sha256"]==controller and origin["selected_executable_sha256"]==result["selected_executable_sha256"]==image,"actual source/controller/image hashes join frozen executed arguments")
identity(origin["parent"],pid=365721,birth="84478339",command=proposal["actual_diagnostic_argv"])
identity(first["observed_child"],pid=365739,parent=365721,birth="84478390",command=command,affinity=list(range(8)))
identity(last["observed_child"],pid=365739,parent=365721,birth="84478390",command=command,affinity=list(range(8)))
for observation in (first,last):
    observed=observation["observed_child"]
    predicates={"parent_pid_matches":observed["parent_pid"]==365721,"command_matches":observed["command"]==command,"inherited_affinity_matches":observed["affinity"]==list(range(8)),"popen_pid_matches":observed["pid"]==365739}
    if observation is last:predicates["birth_matches_first_observation"]=observed["process_start_time"]==first["observed_child"]["process_start_time"]
    require(observation["predicate_results"]==predicates and all(predicates.values()) and observation["all_predicates_match"] is True,"actual raw parent/argv/affinity/birth predicates independently recomputed")
    require(observation["controller_owns_gpu_standing_descriptor"] is observation["producer_owns_gpu_standing_descriptor"] is False and observation["scientific_credit"]==observation["case_credit"]==observation["idea_credit"]==0 and observation["admission_granted"] is observation["worker_admitted"] is False,"actual CPU-only snapshots claim no GPU descriptor or admission/credit")
ack_raw=bytes.fromhex(result["acknowledgment_raw_hex"])
require(ack_raw.endswith(b"\n") and sha(ack_raw)==result["acknowledgment_sha256"] and json.loads(ack_raw)==result["actual_child_acknowledgment"],"actual complete raw child acknowledgment joins its hash and decoded fields")
ack=result["actual_child_acknowledgment"]
require(ack["nonce"]==command[6] and ack["source_sha256"]==source and ack["actual_image_sha256"]==last["actual_executable_sha256"]==image,"actual child nonce/source and mapped executable hash match challenge and prepared image")
identity(ack["identity"],pid=365739,parent=365721,birth="84478390",command=command,affinity=list(range(8)))
require(last["actual_executable_path"]==origin["resolved_selected_executable"]=="/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python3.12","actual acknowledged executable image path joins selected resolved interpreter")
require(datetime.fromisoformat(first["at_utc"])<datetime.fromisoformat(ack["identity"]["observed_at_utc"])<datetime.fromisoformat(last["at_utc"]),"actual immediate, child-self and parent-acknowledged observation chronology")
require(type(result["actual_wait_returncode"]) is int and result["actual_wait_returncode"]==0 and result["actual_wait_verified"] is True and result["release_sent_after_both_positive_observations"] is True,"actual exact Popen child was released only after both positive observations and waited to normal zero exit")
require(result["CPU_affinity_restored"] is True and result["within_total_300_seconds"] is True and origin["work_end_monotonic"]-origin["started_monotonic"]==240 and origin["total_end_monotonic"]-origin["started_monotonic"]==300,"reviewed actual producer observed restored original affinity within fixed240/300second limits")
require(result["stderr_hex"]==result["stdout_tail_hex"]=="","actual diagnostic child produced no trailing output or stderr")
require(origin["numerical_modules_or_devices_requested"] is False and result["scientific_credit"]==result["environment_transitions"]==0 and result["worker_admission"] is False,"actual diagnostic performs zero scientific transitions and grants no credit/admission")
require(result["isolated_python_startup_flags"]==origin["isolated_python_startup_flags"]==["-I","-S","-B"] and result["matches_original_scientific_startup_flags"] is origin["matches_original_scientific_startup_flags"] is False and result["historical_failure_cause_established"] is result["full_original_launch_reproduced"] is result["interpreter_startup_native_work_quantified"] is False,"isolated-startup/native-work/original-failure limits remain explicit")
for item in list(INPUTS.values()):
    require(sha(Path(item["path"]).read_bytes())==item["sha256"],"actual bounded evidence unchanged during independent audit: "+Path(item["path"]).name)
report={"schema_version":1,"kind":"independent_actual_eight_CPU_selected_interpreter_diagnostic_review","passed":True,"actual_diagnostic_accepted":True,"actual_source_commit":COMMIT,"actual_publication":PUB,"actual_parent":{"pid":365721,"birth":"84478339"},"actual_child":{"pid":365739,"birth":"84478390","parent_pid":365721,"normal_exact_wait_returncode":0},"selected_CPU_affinity":list(range(8)),"actual_elapsed_seconds":result["elapsed_seconds"],"immediate_and_acknowledged_predicates_recomputed":True,"actual_wait_source_and_saved_evidence_verified":True,"affinity_restored_as_observed_by_reviewed_producer":True,"separate_raw_post_restore_mask_recorded":False,"actual_explicit_numerical_or_GPU_operations":0,"whole_interpreter_native_startup_work":"unquantified","scientific_credit":0,"worker_or_GPU_or_numerical_admission":False,"historical_failure_cause_established":False,"full_original_scientific_startup_reproduced":False,"remaining_action":"A fresh separately prepared timing attempt may use the reviewed observation-only controller with unchanged exact identity checks. Preserve both earlier failed units; this diagnostic does not establish their initial mismatch cause or numerical success.","remote_shell_startup_warning":stderr,"actual_owned_CPU_audit_context":{"job_id":os.environ["SLURM_JOB_ID"],"step_id":os.environ["SLURM_STEP_ID"],"hostname":socket.gethostname(),"pid":os.getpid()},"reviewer_actions":{"process_cases_repeated":False,"target_queries_or_launches":False,"numerical_or_device_or_native_library_loads":False,"large_historical_payload_replay":False},"inputs":INPUTS,"checks":CHECKS,"elapsed_seconds":time.monotonic()-START,"at_utc":datetime.now(timezone.utc).isoformat()}
out=P/"independent_actual_eight_CPU_diagnostic_review.json"
with out.open("x") as stream:
    json.dump(report,stream,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
print(json.dumps({"passed":True,"checks":len(CHECKS),"step":os.environ["SLURM_STEP_ID"],"elapsed_seconds":report["elapsed_seconds"],"review":{"path":str(out),"sha256":sha(out.read_bytes())}}))
