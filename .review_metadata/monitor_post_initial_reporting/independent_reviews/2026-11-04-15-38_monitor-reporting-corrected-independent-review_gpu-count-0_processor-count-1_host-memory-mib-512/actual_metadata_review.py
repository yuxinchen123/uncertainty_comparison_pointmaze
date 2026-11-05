"""Join existing approved metadata only; never invoke scientific validators."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import sys

W=Path("/p/rlprojects/RND_worktrees/2026-11-04-15-04_monitor-post-initial-validation-reporting_common-correctness-confirmation-kernel-profiles-and-packing")
R=Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
OUT=Path(__file__).resolve().parent
D=W/".review_metadata/monitor_post_initial_reporting"
freeze_path=D/"source_freeze_after_independent_review_corrections.json"
freeze_sha="0e4f4a967b282c4e2f5acd541fdec74ec5175410b0d46831b0d3fe5f7d9b07bd"
assert os.environ.get("SLURM_JOB_ID")=="6783003" and os.getuid()==2618919
step=os.environ.get("SLURM_STEP_ID","")
assert re.fullmatch(r"[0-9]+",step) and socket.gethostname().split(".")[0]=="heartpiece"
assert re.search(r"/job_6783003/step_"+step+r"(?:/|$)",Path("/proc/self/cgroup").read_text())
assert hashlib.sha256(freeze_path.read_bytes()).hexdigest()==freeze_sha
freeze=json.loads(freeze_path.read_text())
assert all(hashlib.sha256((W/name).read_bytes()).hexdigest()==row["sha256"] for name,row in freeze["files"].items())
denied=[]
def audit(event,args):
    bad=event in {"subprocess.Popen","os.system","os.posix_spawn","ctypes.dlopen","ctypes.dlsym","socket.connect","socket.bind"}
    if event=="import" and args and str(args[0]).split(".")[0] in {"jax","jaxlib","numpy","torch","flax","optax","cupy"}:
        bad=True
    if event=="open" and args and isinstance(args[0],(str,bytes,os.PathLike)):
        name=os.fsdecode(args[0]);bad|=name.startswith(("/dev/nvidia","/dev/dri")) or name.endswith((".so",".so.1"))
        if name.startswith(str(R)+"/") and not name.startswith(str(OUT)+"/"):
            bad|=not name.endswith(".json")
            mode=args[1] if len(args)>1 else None
            flags=args[2] if len(args)>2 else 0
            bad|=isinstance(mode,str) and any(value in mode for value in "wax+")
            bad|=isinstance(flags,int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
    if bad:
        denied.append({"event":event,"arguments":repr(args)[:400]})
        raise RuntimeError("outside actual metadata join scope: "+event)
sys.addaudithook(audit)
scripts=W/"10_jax_exploration_platform/scripts/atari_pixel_campaign"
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location("independent_actual_metadata_controller",scripts/"controller.py")
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
scientific_calls=[]
def no_scientific(*args,**kwargs):
    scientific_calls.append(True);raise RuntimeError("scientific validator forbidden in reporting review")
for name in ("validate_attempt_completion","scientific_case_complete","final_requirements_met","verify_source","ready_specification"):
    setattr(c,name,no_scientific)
n=json.loads((R/"next_actions.json").read_text())
reference=n["new_common_correctness_submissions"]["independent_raw_review"]
aggregate=json.loads(Path(reference["path"]).read_text())
results=[]
for copies in (1,2):
    result_ref=aggregate["records"][str(copies)]["result"]
    plan_path=Path(result_ref["path"]).with_name("measurement_plan.json")
    plan_ref={"path":str(plan_path),"sha256":hashlib.sha256(plan_path.read_bytes()).hexdigest()}
    plan=json.loads(plan_path.read_text())
    entry={"plan":plan_ref,"result":result_ref,"review":reference}
    output=c.reporting_reviewed_unit(R,entry,"common_correctness",plan["source_commit"],c.reporting_configuration(plan["configuration"]),1)
    assert output==("jaguar03_rtx_a4500",copies)
    results.append({"requirement":"common_correctness","hardware_class":output[0],"copies":copies,"matched":True})
    review_ref=n["common_a4500_confirmation_preparation"]["independent_reviews"][str(copies)]
    review=json.loads(Path(review_ref["path"]).read_text())
    entry={"plan":review["plan"],"result":review["result"],"review":review_ref}
    plan=json.loads(Path(entry["plan"]["path"]).read_text())
    output=c.reporting_reviewed_unit(R,entry,"common_confirmation",plan["source_commit"],c.reporting_configuration(plan["configuration"]),1)
    assert output==("jaguar03_rtx_a4500",copies)
    results.append({"requirement":"common_confirmation","hardware_class":output[0],"copies":copies,"matched":True})
# The newly accepted direct one-copy review has a different existing raw-review shape.
unit=R/"data/2026-11-01-23-06_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_h100-nvl_serval02-card-1_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1_hours-3"
review_path=R/"remote_setup/2026-11-02-00-06_h100-common-correctness_raw-review_copies-1-2_source-0a880984_helpers-26_before-seven-paired-rounds-ten-updates_gpu-count-0/copies_1_terminal_review/independent_actual_review.json"
assert hashlib.sha256(review_path.read_bytes()).hexdigest()=="22f4ec55120c987fe8d44b55e6427ab75e25db61785e29d7cb6bdcbc55df8818"
entry={key:{"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for key,path in
       [("plan",unit/"measurement_plan.json"),("result",unit/"result.json"),("review",review_path)]}
plan=json.loads((unit/"measurement_plan.json").read_text())
output=c.reporting_reviewed_unit(R,entry,"common_correctness",plan["source_commit"],c.reporting_configuration(plan["configuration"]),1)
assert output==("h100",1)
results.append({"requirement":"common_correctness","hardware_class":"h100","copies":1,"matched":True})
preview=c.post_initial_reporting_requirements(R,"d28e6735e79788f618acb0ebfde8609db1f776af",[1,2])
assert all(row["scientific_credit"]==0 and row["admission_authority"] is False for row in preview)
assert not denied and not scientific_calls
assert all(hashlib.sha256((W/name).read_bytes()).hexdigest()==row["sha256"] for name,row in freeze["files"].items())
report={"kind":"independent_actual_existing_reporting_metadata_joins","passed":True,"job_id":"6783003","step_id":step,"uid":os.getuid(),"host":socket.gethostname(),"at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"source_freeze":{"path":str(freeze_path),"sha256":freeze_sha},"results":results,"actual_metadata_joins":len(results),"live_campaign_writes":0,"raw_scientific_validators_invoked":0,"denied_events":denied,"scientific_credit":0,"gpu_or_device_or_library_or_network_or_process_actions":0,"source_files_unchanged":True,"reporting_preview":preview}
(OUT/"actual_metadata_review.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({key:value for key,value in report.items() if key not in {"reporting_preview","results"}}))
