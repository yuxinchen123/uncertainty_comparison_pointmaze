"""Verify actual source Git bytes and final standard-record changes without executing the command."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

P = Path(__file__).resolve().parent
COMMIT = "e16c015ac7c9ca674f768240cad35a372eb18e87"
START = time.monotonic()
CHECKS = []


def require(value, text):
    assert value,text
    CHECKS.append(text)


def ref(path):
    path=Path(path)
    raw=path.read_bytes()
    require(path.is_file() and not path.is_symlink() and len(raw)<2**20,"ordinary bounded exact file: "+path.name)
    return {"path":str(path),"sha256":hashlib.sha256(raw).hexdigest()}


def document(path):
    ref(path)
    return json.loads(Path(path).read_bytes())


def publish(path,value):
    with path.open("x") as stream:
        json.dump(value,stream,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())


require(os.environ.get("SLURM_JOB_ID")=="6821437" and os.environ.get("SLURM_STEP_ID","").isdigit(),"current exact owned CPU allocation and numeric step")
require(os.getresuid()==(2618919,)*3,"actual independent reviewer owner")
command_review_ref=ref(P/"independent_command_review.json")
require(command_review_ref["sha256"]=="8c3477d718e0e51aeeeecb278c8bbf51c4693754bc06b6764ea1f883330c1cff","genuine existing command review remains unchanged")
review=document(command_review_ref["path"])
source_review=ref(review["source_review"]["path"])
require(source_review==review["source_review"] and source_review["sha256"]=="db15bd9b3ce07c9913473aebc3e3756e7d027f5e25ca02be1eb903cd5825702f","genuine source review remains exact")
freeze=document(review["corrected_source_freeze"]["path"])
root=Path(freeze["source_root"])
head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],timeout=20,text=True).strip()
require(head==COMMIT,"actual reviewed source commit exists and is checkout HEAD")
changed=subprocess.check_output(["git","-C",str(root),"diff-tree","--no-commit-id","--name-only","-r",COMMIT],timeout=20,text=True).splitlines()
require(set(changed)==set(freeze["files"]) and len(changed)==3,"actual source commit changes only three reviewed source/test paths")
source_blobs={}
for relative,item in freeze["files"].items():
    require(ref(item["path"])["sha256"]==item["sha256"],"source still matches corrected frozen bytes: "+Path(relative).name)
    blob=subprocess.check_output(["git","-C",str(root),"cat-file","blob",COMMIT+":"+relative],timeout=20)
    require(hashlib.sha256(blob).hexdigest()==item["sha256"] and len(blob)==item["bytes"],"actual source Git blob matches reviewed freeze: "+Path(relative).name)
    source_blobs[relative]={"git_commit":COMMIT,"sha256":item["sha256"],"bytes":len(blob)}
    require(subprocess.check_output(["git","-C",str(root),"diff","--",relative],timeout=20)==b"","reviewed source path has no unstaged changes: "+Path(relative).name)
old=P/"standard_records_before_actual_source_commit"
for name in ("manifest.yaml","experiment_background.md"):
    require(ref(old/name)["sha256"]==review["inputs"][str(P/name)]["sha256"],"prior standard record bytes preserved from reviewed command: "+name)
for name in ("config_resolved.yaml","command.txt","proposed_one_shot_command.json"):
    require(ref(P/name)==review["inputs"][str(P/name)],"actual command/config pins remain unchanged: "+name)
before=document(old/"manifest.yaml")
after=document(P/"manifest.yaml")
expected=dict(before)
expected.pop("source_publication_pending")
expected["state"]="reviewed_prepared_single_root_execution_after_publication"
expected["git"]={"commit":COMMIT,"source_root":str(root),"reviewed_source_files_unchanged":True}
expected["source_publication_receipt"]="root_actual_source_publication_verification.json"
require(after==expected,"manifest changes only actual source Git state, prepared state and future receipt path; no invented publication")
old_background=(old/"experiment_background.md").read_text()
new_background=(P/"experiment_background.md").read_text()
require(old_background.split("## Git state",1)[0]==new_background.split("## Git state",1)[0],"purpose, resource scope and code-change sections unchanged")
new_git=new_background.split("## Git state",1)[1]
require(COMMIT in new_git and "publish the source and these records before the single invocation" in new_git and "does not reproduce the original scientific startup" in new_git and "failure’s cause" in new_git,"updated background names actual local source commit and keeps publication/diagnostic limitations explicit")
records={name:ref(P/name) for name in ("manifest.yaml","config_resolved.yaml","experiment_background.md","command.txt","proposed_one_shot_command.json")}
require(not (P/"diagnostic_origin.json").exists() and not (P/"diagnostic_result.json").exists(),"actual target diagnostic remains unattempted")
final_freeze={"schema_version":1,"kind":"final_standard_records_and_actual_reviewed_source_commit_freeze","actual_source_commit":COMMIT,"source_root":str(root),"source_blobs":source_blobs,"records":records,"command_review":command_review_ref,"source_review":source_review,"actual_source_publication_receipt_available":False,"actual_target_execution":False,"scientific_credit":0}
freeze_path=P/"final_standard_record_freeze.json"
publish(freeze_path,final_freeze)
for value in records.values():
    require(ref(value["path"])==value,"final standard record unchanged during addendum: "+Path(value["path"]).name)
addendum={"schema_version":1,"kind":"independent_final_standard_records_and_actual_source_commit_addendum","passed":True,"approved_for_single_root_execution_after_publication":True,"actual_source_commit":COMMIT,"source_blobs":source_blobs,"prior_command_review":command_review_ref,"prior_source_review":source_review,"final_standard_record_freeze":ref(freeze_path),"only_manifest_and_background_changed_since_prior_review":True,"CPU_process_cases_repeated":False,"actual_target_execution":False,"actual_source_publication_verified_by_this_addendum":False,"required_before_actual_once_SSH":"Root publishes exact source and final standard records/reviews, verifies real publication bytes and clean source, then uses the exact unchanged reviewed command once.","actual_owned_CPU_context":{"job_id":os.environ["SLURM_JOB_ID"],"step_id":os.environ["SLURM_STEP_ID"],"hostname":socket.gethostname(),"pid":os.getpid()},"checks":CHECKS,"elapsed_seconds":time.monotonic()-START}
out=P/"independent_final_standard_record_addendum.json"
publish(out,addendum)
print(json.dumps({"passed":True,"checks":len(CHECKS),"step":os.environ["SLURM_STEP_ID"],"elapsed_seconds":addendum["elapsed_seconds"],"review":ref(out),"final_standard_record_freeze":ref(freeze_path)}))
