"""Check display-only joins against the four existing A4500 metadata chains."""
import hashlib, importlib.util, json, os
from pathlib import Path
import re, socket, sys
D=Path(__file__).resolve(strict=True).parent
W=D.parents[1]
R=Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
def main():
    """Use only exact owned CPU metadata reads; never run a scientific validator."""
    job,step=os.environ.get("SLURM_JOB_ID"),os.environ.get("SLURM_STEP_ID")
    assert job=="6783003" and re.fullmatch(r"[0-9]+",step or "")
    assert not socket.gethostname().split(".")[0].startswith(("portal","login"))
    assert re.search(r"/job_"+job+r"/step_"+step+r"(?:/|$)",Path("/proc/self/cgroup").read_text())
    f=json.loads((D/"source_freeze_after_metadata_size_correction.json").read_text())
    assert all(hashlib.sha256((W/p).read_bytes()).hexdigest()==v["sha256"] for p,v in f["files"].items())
    denied=[]
    def audit(event,args):
        """Refuse numerical, device, library and process actions during metadata joins."""
        bad=event in {"subprocess.Popen","os.system","os.posix_spawn","ctypes.dlopen","ctypes.dlsym","socket.connect"}
        if event=="import" and args and str(args[0]).split(".")[0] in {"jax","jaxlib","numpy","torch"}:bad=True
        if event=="open" and args and isinstance(args[0],(str,bytes,os.PathLike)):
            name=os.fsdecode(args[0]);bad|=name.startswith(("/dev/nvidia","/dev/dri")) or name.endswith((".so",".so.1"))
        if bad:denied.append(event);raise RuntimeError("outside reporting metadata join: "+event)
    sys.addaudithook(audit)
    root=W/"10_jax_exploration_platform/scripts/atari_pixel_campaign";sys.path.insert(0,str(root))
    spec=importlib.util.spec_from_file_location("reporting_join_controller",root/"controller.py");c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    n=json.loads((R/"next_actions.json").read_text())
    aggregate_ref=n["new_common_correctness_submissions"]["independent_raw_review"]
    aggregate=json.loads(Path(aggregate_ref["path"]).read_text())
    results=[]
    for copies in (1,2):
        result_ref=aggregate["records"][str(copies)]["result"];plan_path=Path(result_ref["path"]).with_name("measurement_plan.json")
        plan_ref={"path":str(plan_path),"sha256":hashlib.sha256(plan_path.read_bytes()).hexdigest()}
        plan=json.loads(plan_path.read_text());entry={"plan":plan_ref,"result":result_ref,"review":aggregate_ref}
        output=c.reporting_reviewed_unit(R,entry,"common_correctness",plan["source_commit"],c.reporting_configuration(plan["configuration"]),1)
        assert output==("jaguar03_rtx_a4500",copies);results.append({"requirement":"common_correctness","copies":copies,"matched":True})
        review_ref=n["common_a4500_confirmation_preparation"]["independent_reviews"][str(copies)];review=json.loads(Path(review_ref["path"]).read_text())
        entry={"plan":review["plan"],"result":review["result"],"review":review_ref};plan=json.loads(Path(entry["plan"]["path"]).read_text())
        output=c.reporting_reviewed_unit(R,entry,"common_confirmation",plan["source_commit"],c.reporting_configuration(plan["configuration"]),1)
        assert output==("jaguar03_rtx_a4500",copies);results.append({"requirement":"common_confirmation","copies":copies,"matched":True})
    preview=c.post_initial_reporting_requirements(R,"d28e6735e79788f618acb0ebfde8609db1f776af",[1,2])
    assert all(row["state"]=="awaiting_reviewed_evidence" for row in preview)
    v={"kind":"actual_existing_a4500_reporting_metadata_join","passed":True,"job_id":job,"step_id":step,"results":results,"actual_campaign_writes":0,"scientific_recomputation":False,"scientific_credit":0,"raw_scientific_validators_invoked":0,"GPU_library_device_network_process_actions":0,"denied_events":denied,"missing_index_current_preview":preview}
    p=D/("actual_metadata_"+job+"_"+step+".json")
    with p.open("x") as stream:stream.write(json.dumps(v,indent=2)+"\n")
    print(json.dumps({"passed":True,"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"actual_existing_metadata_joins":len(results)}))
    return 0
if __name__=="__main__":raise SystemExit(main())
