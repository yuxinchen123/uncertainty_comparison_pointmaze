"""Freeze the actual common-confirmation packets without running their producer."""
import ast,datetime,hashlib,json,os,stat,subprocess
from pathlib import Path
W=Path("/p/rlprojects/RND_worktrees/2026-11-01-17-55_a4500-common-confirmation_prior-correctness-required_rounds-7_complete-updates-per-arm-10_copies-1-2");R=Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1");H=Path("/p/rlprojects/RND_worktrees/2026-11-01-18-47_common-confirmation-source_commit-9fcc43a478b93d731df780805fd7c3c514cdf67f")
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p)}
def put(p,v):
 raw=(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return ref(p)
records=[];files={}
for copies,hours in ((1,2),(2,3)):
 names=list((R/'remote_setup').glob('2026-11-01-18-54_common*confirmation_paired-rounds-7*copies-'+str(copies)+'_*'))
 assert len(names)==1;P=names[0];U=R/'data'/P.name;plan=json.loads((U/'measurement_plan.json').read_bytes())
 assert len(P.name.encode())==242 and plan['mode']=='confirmation' and plan['rounds']==7 and plan['iterations']==10
 assert plan['copy_seeds']==list(range(1,copies+1)) and plan['gpu_uuid'] is None and plan['helpers']['source_commit']=='9fcc43a478b93d731df780805fd7c3c514cdf67f'
 assert len(plan['helpers']['files'])==21
 progress=U/'progress.json'
 if not progress.exists():
  put(progress,{'schema_version':1,'kind':'prepared_intermediate_common_confirmation_progress','state':'prepared_not_submitted','mode':'confirmation','copy_seeds':plan['copy_seeds'],'paired_rounds_planned':7,'complete_updates_per_arm_per_round':10,'completed_rounds':0,'actual_job_id':None,'gpu_uuid':None,'submitted':False,'correctness_prerequisite':plan['common_correctness_inputs'],'scientific_credit':0,'case_credit':0,'idea_credit':0,'final_training_credit':0,'admission_granted':False})
 for directory in (P,U):
  for name in ('experiment_background.md','manifest.yaml','config_resolved.yaml','command.txt'):
   assert (directory/name).is_file()
 for name in ('experiment_background.md','manifest.yaml','config_resolved.yaml','command.txt','progress.json'):assert (U/name).is_file()
 assert not (U/'owned_allocation_inputs.json').exists() and not (P/'submission_intent.json').exists()
 tests=json.loads((P/'barrier_tests_final_receipt.json').read_bytes());assert tests['tests_run']==22 and tests['returncode']==0
 assert not any((P/n).exists() for n in ('submission_receipt.json','held_binding.json','release_receipt.json'))
 for directory in (P,U):
  for p in sorted(directory.rglob('*')):
   assert len(p.name.encode())<=255 and not p.is_symlink()
   if p.is_file():files[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size,'mode':stat.S_IMODE(p.stat().st_mode)}
 records.append({'copies':copies,'hours':hours,'packet':str(P),'unit':str(U),'plan':ref(U/'measurement_plan.json'),'script':ref(P/'job.sbatch'),'artifacts':ref(P/'immutable_artifacts.json'),'barrier_tests':ref(P/'barrier_tests_final_receipt.json'),'standard_data_records':[ref(U/n) for n in ('experiment_background.md','manifest.yaml','config_resolved.yaml','command.txt','progress.json')]})
index=R/'remote_setup/2026-11-01-19-00_common-confirmation_prepared-packets_copies-1-2_rounds-7_updates-per-arm-10.json'
value={'schema_version':1,'kind':'common_a4500_confirmation_packets_static_freeze','passed':True,'cpu_job_id':'6783003','cpu_step_id':os.environ['SLURM_STEP_ID'],'helper_root':str(H),'helper_commit':'9fcc43a478b93d731df780805fd7c3c514cdf67f','packets':records,'files':files,'count':len(files),'preparation_guard':ref(W/'.review_metadata/common_confirmation_packet_preparation_6783003_42.json'),'qualification_scope':ref(W/'.review_metadata/qualification_reuse_scope_6783003_26.json'),'independent_source_review':ref(W/'.review_metadata/common_confirmation_independent_source_review.json'),'time_budget':ref(W/'.review_metadata/common_confirmation_walltime_budget_6783003_35.json'),'canonical_environment_registry':ref(Path('/p/rlprojects/RND/.venvs/ENVS.md')),'current_monitor_scientific_acceptance':'progress display only; use each exact new staged consumer','submitted_jobs':0,'GPU_operations':0,'scientific_credit':0,'admission_granted':False}
assert subprocess.check_output(['git','-C',str(H),'status','--porcelain=v1','--untracked-files=all'])==b''
print(json.dumps(put(index,value)))

