"""Read and approve the exact CPU-only diagnostic command without executing it."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import time

P = Path(__file__).resolve().parent
START = time.monotonic()
INPUTS = {}
CHECKS = []


def require(value, statement):
    assert value, statement
    CHECKS.append(statement)


def ref(path):
    path = Path(path)
    raw = path.read_bytes()
    require(path.is_file() and not path.is_symlink() and len(raw)<2**20,"ordinary bounded exact input: " + path.name)
    item = {"path":str(path),"sha256":hashlib.sha256(raw).hexdigest()}
    INPUTS[str(path)] = item
    return item


def doc(path):
    ref(path)
    return json.loads(Path(path).read_bytes())


require(os.environ.get("SLURM_JOB_ID") == "6821437" and os.environ.get("SLURM_STEP_ID","").isdigit(),"exact current owned CPU review step")
require(os.getresuid() == (2618919,)*3,"actual independent reviewer UID")
proposal = doc(P/"proposed_one_shot_command.json")
require(ref(P/"proposed_one_shot_command.json")["sha256"] == "cb60c54dc8695da888650dc87874ff92965909d489484d021e2b5540f6bbcc16","exact root-held corrected command proposal")
freeze_ref = proposal["source_freeze"]
require(ref(freeze_ref["path"]) == freeze_ref and freeze_ref["sha256"] == "63c3186658242633d913b8302c1b0dae38adb8ee72faf311c64b693d324665cc","proposal binds corrected freeze path and bytes, preserving earlier freeze")
freeze = doc(freeze_ref["path"])
source_review_path = Path(freeze_ref["path"]).parent / "independent_source_review.json"
require(ref(source_review_path)["sha256"] == "db15bd9b3ce07c9913473aebc3e3756e7d027f5e25ca02be1eb903cd5825702f","genuine final-byte independent source review")
source_review = doc(source_review_path)
require(source_review["approved"] is True and source_review["passed"] is True and source_review["freeze"] == freeze_ref and source_review["actual_target_eight_CPU_diagnostic_run"] is False,"source review passed without claiming actual target execution")
for item in freeze["files"].values():
    require(ref(item["path"])["sha256"] == item["sha256"],"exact held source/test before root command admission: " + Path(item["path"]).name)
source = Path(freeze["source_root"])/"10_jax_exploration_platform/scripts/atari_pixel_campaign/diagnose_initial_child_identity.py"
expected = ["/usr/bin/python3","-I","-S","-B",str(source),"--execute","--run-dir",str(P),"--expected-source-sha256","25876e545a9791b15c0ec649882da3a4accc9bcc56788e4f7a28911f0ad763ba","--expected-controller-sha256","09a1c944116541b631e12e465f5e782641429e305e9a135da4dc8e5379f9a5b6","--expected-python-sha256","c889543c0e008c1362b69c711330cac598833a790dc0c48c5ee4a8567d3f3b51"]
require(proposal["actual_diagnostic_argv"] == expected,"exact corrected source/controller/selected-image hashes and one diagnostic output path")
argv = proposal["one_shot_ssh_argv"]
require(len(argv)==9 and argv[:8]==["ssh","-o","BatchMode=yes","-o","ControlMaster=no","-o","ConnectTimeout=15","serval02"],"exact one direct serval02 SSH without retries or other host")
remote = shlex.split(argv[8])
environment = {"PATH":"/usr/local/bin:/usr/bin:/bin","LANG":"C.UTF-8","TZ":"UTC","PYTHONNOUSERSITE":"1","PYTHONDONTWRITEBYTECODE":"1"}
require(remote[:-1] == ["/usr/bin/env","-i",*[key+"="+value for key,value in environment.items()],"/usr/bin/python3","-I","-S","-B","-c"],"exact fixed clean environment and isolated system-Python host/owner precheck")
expected_code = 'import os,socket; assert socket.gethostname().split(".")[0]=="serval02" and os.getresuid()==(2618919,2618919,2618919); os.execve('+repr('/usr/bin/python3')+','+repr(expected)+','+repr(environment)+')'
require(ast.dump(ast.parse(remote[-1]),include_attributes=False)==ast.dump(ast.parse(expected_code),include_attributes=False),"remote code only checks exact host and owner then execs exact diagnostic argv/environment")
require(proposal["executed"] is False and proposal["repeat_after_attempt_forbidden"] is True,"proposal is unattempted and forbids retry")
require(shlex.split((P/"command.txt").read_text()) == argv,"standard command record is exact reviewed single SSH argv")
for name in ("manifest.yaml","config_resolved.yaml","experiment_background.md","command.txt"):
    ref(P/name)
config = doc(P/"config_resolved.yaml")
require(config["host"]=="serval02" and config["cpu_count"]==8 and config["cpu_affinity_exclusive"] is False and config["gpu_count"]==0 and config["maximum_wall_seconds"]==300,"standard exact eight-CPU nonexclusive zero-GPU resource declaration")
require(config["source_sha256"]==expected[9] and config["controller_sha256"]==expected[11] and config["selected_executable_sha256"]==expected[13] and config["source_freeze_sha256"]==freeze_ref["sha256"],"config matches every exact proposed source hash")
require(config["environment_transitions"]==config["learner_updates"]==config["scientific_credit"]==config["timing_credit"]==config["standing_gpu_lock_operations"]==config["root_metadata_permission_changes"]==0,"zero transitions, scientific/timing credit, standing-lock or metadata-permission operations")
require(config["executed"] is False and config["isolated_startup_flags"]==["-I","-S","-B"] and config["original_scientific_startup_flags"]==[] and config["full_original_launch_reproduced"] is False and config["historical_failure_cause_established"] is False,"isolated diagnostic differs from historical scientific startup and grants no retrospective cause")
for name in ("diagnostic_origin.json","immediate_child_observation.json","acknowledged_child_observation.json","diagnostic_result.json","root_actual_execution_return.json"):
    require(not (P/name).exists(),"diagnostic remains unattempted: " + name)
for item in list(INPUTS.values()):
    require(ref(item["path"])==item,"command/source review input unchanged: " + Path(item["path"]).name)
report = {"schema_version":1,"kind":"independent_cpu_selected_interpreter_diagnostic_one_shot_command_review","passed":True,"approved_for_single_root_execution_after_publication":True,"source_review":ref(source_review_path),"corrected_source_freeze":freeze_ref,"actual_ssh_argv":argv,"actual_diagnostic_argv":expected,"inputs":INPUTS,"actual_source_publication":None,"actual_target_execution":False,"scientific_or_operational_worker_admission":False,"scientific_credit":0,"standing_gpu_lock_or_metadata_permission_changes":False,"original_failure_cause_established":False,"actual_owned_cpu_context":{"job_id":os.environ["SLURM_JOB_ID"],"step_id":os.environ["SLURM_STEP_ID"],"hostname":socket.gethostname(),"pid":os.getpid()},"required_before_actual_once_SSH":["Publish exact reviewed source, standard records, source review and this command review.","Verify actual clean source commit and exact proposed hashes immediately before one attempt; record actual intent, stdout/stderr and return.","Do not repeat this packet after an attempt; preserve all actual observations and waits, including failure.","This isolated-startup CPU diagnostic does not qualify numerical timing, a GPU or the historical failure cause."],"checks":CHECKS,"elapsed_seconds":time.monotonic()-START}
out=P/"independent_command_review.json"
with out.open("x") as stream:
    json.dump(report,stream,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
print(json.dumps({"passed":True,"checks":len(CHECKS),"step":os.environ["SLURM_STEP_ID"],"elapsed_seconds":report["elapsed_seconds"],"review":{"path":str(out),"sha256":hashlib.sha256(out.read_bytes()).hexdigest()}}))
