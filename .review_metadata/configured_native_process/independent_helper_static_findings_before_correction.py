"""Record independent static findings before the author corrects the held helper slice."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import time

D=Path(__file__).resolve().parent
W=D.parents[1]
S=W/"10_jax_exploration_platform/scripts/atari_pixel_campaign"
START=time.monotonic()
EXPECTED="b124b587b9a3e3f56a4150be0d9d11c43233a6e6e34f2a438c2920668108c4de"
CHECKS=[]

def require(value,text):
    assert value,text
    CHECKS.append(text)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

require(os.environ.get("SLURM_JOB_ID")=="6821437" and os.environ.get("SLURM_STEP_ID","").isdigit() and os.getresuid()==(2618919,)*3,"current exact owned CPU job/step/owner")
freeze_path=D/"author_helper_source_freeze_2.json"
require(sha(freeze_path)==EXPECTED,"exact held helper freeze before corrections")
freeze=json.loads(freeze_path.read_bytes())
for relative,item in freeze["files"].items():require(sha(W/relative)==item["sha256"],"actual held source/test bytes: "+Path(relative).name)
path=S/"configured_native_process.py"
spec=importlib.util.spec_from_file_location("independent_pure_configured_native_policy",path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
profile_command=module.producer_command(Path("/synthetic/nonoperational/unit"),{"kind":module.PROFILE,"python":"/synthetic/selected/python"})
require(profile_command[1:4]==["-I","-S","-B"],"concrete metadata fixture reproduces formal producer startup flags that differ from original prepared command")
require("nsight_process_identity.py" not in module.HELPERS,"critical shared ownership reader missing from qualification helper set")
source=(S/"native_process_evidence.py").read_text()
require("require(all(event['identity']['pid'] == event['opened_pid']" in source and "expected_opened.append(owners['capture']['pid'])" in source,"configured waiter gained capture handle but retains live-only descriptor-event condition after child wait")
diag=(S/"diagnose_configured_native_process.py").read_text()
require("child.wait(timeout=deadline.timeout())" in diag and "observer.close_process_handle(child_handle)" in diag,"new diagnostic closes capture handle after actual child wait")
require("signal.signal" not in diag and "guard_termination" not in diag,"new diagnostic has no explicit signal-to-scope-cleanup context")
report={"schema_version":1,"kind":"independent_held_configured_native_helper_static_findings","approved":False,"source_freeze":{"path":str(freeze_path),"sha256":EXPECTED},"scope":"static source and nonoperational command fixture; profile author slice excluded; no native syscall or target work","findings":[{"problem":"Future profile producer command adds isolated flags despite original selected profile package-discovery command.","required_correction":"Keep isolated flags for diagnostic only; preserve original formal profile producer command and numerical child startup."},{"problem":"Configured shared qualification helper hashes omit nsight_process_identity.py, the live ownership/ancestry reader.","required_correction":"Bind that shared critical reader in configured declaration and diagnostic/profile qualification comparison; stage unique helper names."},{"problem":"New waiter closes capture pidfd after successful child wait, while its saved-role validator requires all descriptor events to retain a live PID.","required_correction":"Allow only the configured capture-handle close to show exited PID after its actual successful wait timestamp. Self/producer handles remain live-only; other native failures remain failures."},{"problem":"Diagnostic default SIGTERM/SIGINT bypass Python scope cleanup.","required_correction":"Establish the existing reviewed signal-to-cleanup behavior before tool scope, preserving exact waits/ECHILD and truthful interrupted failure."},{"problem":"New diagnostic preparation emits only a paragraph for the experiment background.","required_correction":"Generate the standard Purpose, Key hyperparameters, Code and config changes, and Git state sections with actual source and zero-credit scope."}],"raw_formal_profile_command_fixture":profile_command,"no_tool_or_GPU_or_NUM_or_native_operation":True,"scientific_credit":0,"actual_owned_CPU_context":{"job_id":os.environ["SLURM_JOB_ID"],"step_id":os.environ["SLURM_STEP_ID"],"hostname":socket.gethostname(),"pid":os.getpid()},"checks":CHECKS,"elapsed_seconds":time.monotonic()-START}
out=D/"independent_helper_static_findings_before_correction.json"
with out.open("x") as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
print(json.dumps({"approved":False,"findings":len(report["findings"]),"checks":len(CHECKS),"step":os.environ["SLURM_STEP_ID"],"elapsed_seconds":report["elapsed_seconds"],"record":{"path":str(out),"sha256":sha(out)}}))
