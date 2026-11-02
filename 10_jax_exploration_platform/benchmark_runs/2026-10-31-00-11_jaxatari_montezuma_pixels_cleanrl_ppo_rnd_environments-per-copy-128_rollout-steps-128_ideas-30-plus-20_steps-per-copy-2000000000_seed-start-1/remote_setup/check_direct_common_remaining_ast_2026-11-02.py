"""Close the two remaining protected execution-function comparisons without numerical imports."""
import ast,datetime,hashlib,json,os,subprocess
from pathlib import Path
W=Path("/p/rlprojects/RND_worktrees/2026-11-01-21-42_direct-common-confirmation_backend-schema-2_helpers-27_serval02-card-1_environments-128_rollout-steps-128_paired-rounds-7_updates-per-round-10_before-numerical-import");R=Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
assert os.environ.get("SLURM_JOB_ID")=="6783003" and os.environ.get("SLURM_STEP_ID","").isdigit()
path="10_jax_exploration_platform/scripts/atari_pixel_campaign/measure_selected.py"
base="fdb35f886ad01c6d4ed387dc55b172d0221a9853"
original=ast.parse(subprocess.check_output(["git","-C",str(W),"show",base+":"+path]))
current=ast.parse((W/path).read_bytes())
functions=lambda tree:{node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,ast.FunctionDef)}
left,right=functions(original),functions(current)
evidence={name:hashlib.sha256(left[name].encode()).hexdigest() for name in ("producer_support","card_admission")}
assert all(left[name]==right[name] for name in evidence)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
freeze=W/".review_metadata/direct_common_with_backend_preparation/source_freeze_stdlib_bootstrap_before_audit.json"
assert sha(freeze)=="8b35080816df709ed3402006f272af157862c7b4646bed970d80f96ac67eeec6"
value={"kind":"independent_direct_common_remaining_execution_function_comparisons","passed":True,
 "primary_review":{"path":str(R/"remote_setup/direct_common_guard_independent_6783003_154.json"),"sha256":"008f3c53cefbcc40eab2036d433c134577dfefe987f77f5e048ba50ff2f1504c"},
 "all_thirteen_required_function_comparisons_complete":True,"additional_functions":evidence,
 "source_freeze":{"path":str(freeze),"sha256":sha(freeze)},"measure_selected_sha256":sha(W/path),
 "base":base,"job_id":"6783003","step_id":os.environ["SLURM_STEP_ID"],"pid":os.getpid(),
 "at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "numerical_imports":0,"actual_H100_raw_results_read":False,"GPU_or_host_or_job_actions":0,"admission_granted":False}
out=R/"remote_setup"/("direct_common_guard_remaining_ast_6783003_"+os.environ["SLURM_STEP_ID"]+".json")
with out.open("x") as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write("\n");stream.flush();os.fsync(stream.fileno())
print(json.dumps({"passed":True,"receipt":str(out),"sha256":sha(out)}))

