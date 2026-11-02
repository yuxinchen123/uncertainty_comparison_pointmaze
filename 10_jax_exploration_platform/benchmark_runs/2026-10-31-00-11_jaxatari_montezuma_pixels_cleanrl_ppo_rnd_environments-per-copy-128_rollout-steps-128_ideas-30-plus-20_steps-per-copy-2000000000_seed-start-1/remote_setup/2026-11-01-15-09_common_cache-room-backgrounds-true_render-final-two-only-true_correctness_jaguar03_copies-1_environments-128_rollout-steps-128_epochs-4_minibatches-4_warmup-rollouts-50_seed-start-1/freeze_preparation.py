"""Freeze exact new prepared artifacts without updating source, branches or campaign records."""
from datetime import datetime,timezone
import hashlib,json,os,subprocess
from pathlib import Path

INDEX=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-15-09_unconfirmed-common-a4500-preparation_index.json');v=json.loads(INDEX.read_bytes());run=Path(v['packets'][0]['packet']).parent.parent;repo=run.parents[2]
assert os.environ.get('SLURM_JOB_ID')=='6782816' and os.environ.get('SLURM_STEP_ID','').isdecimal()
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'sha256':digest(p)}
def put(p,x):
 with Path(p).open('x') as f:json.dump(x,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return ref(p)
resources=json.loads((Path(v['packets'][0]['packet'])/'current_resource_observation.json').read_bytes())
assert 'State=ACTIVE' in resources['queries']['reservation']['stdout'] and 'Reason=' in resources['queries']['node']['stdout']
index=[];allpaths=[INDEX]
for item in v['packets']:
 packet=Path(item['packet']);unit=Path(item['data_unit'])
 if item['mode']=='correctness':
  result=subprocess.run(['sha256sum','--check','--strict',str(packet/'source_checksums.sha256')],capture_output=True,check=False);assert result.returncode==0,result.stderr
  put(packet/'source_checksums_verification.json',{'schema_version':1,'passed':True,'cpu_job':'6782816','cpu_step':os.environ['SLURM_STEP_ID'],'returncode':result.returncode,'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(result.stderr).hexdigest(),'checked_files':len(result.stdout.splitlines()),'gpu_operations':0})
  plan=json.loads((unit/'measurement_plan.json').read_text());tests=json.loads((packet/'barrier_tests_final_receipt.json').read_text());assert tests['tests_run']==20 and tests['failures']==tests['errors']==0
  assert plan['configuration']['learner']['n_copies']==item['copies'] and plan['configuration']['learner']['n_envs']==128 and plan['configuration']['learner']['num_steps']==128 and plan['configuration']['learner']['observation_warmup_iterations']==50 and plan['configuration']['learner']['total_timesteps']==2000000000
  assert plan['mode']=='correctness' and plan['rounds']==7 and plan['iterations']==10 and 'final_confirmation_protocol' not in plan and 'completed_study' not in plan
  info={'mode':'correctness','copies':item['copies'],'formal_plan':ref(unit/'measurement_plan.json'),'script':ref(packet/'job.sbatch'),'barrier':ref(packet/'wait_for_owned_inputs.py'),'barrier_tests':ref(packet/'barrier_tests_final_receipt.json'),'actual_target_qualification_inputs':plan['target_qualification_inputs'],'resources':'one A4500/eight CPUs/64GiB/three hours/gpu/sl5nw_base/csresnolim/ownedreservation sl5nw_168','unknown_frozen_job_and_physical_identity':True}
 else:
  assert not any((unit/name).exists() for name in ['measurement_plan.json','measurement_prepared.json','result.json','failure.json'])
  assert not any((packet/name).exists() for name in ['job.sbatch','held_submission_command.json'])
  info={'mode':'generic_confirmation_request_only','copies':item['copies'],'request':ref(packet/'confirmation_request.json'),'timing_budget_supplement':ref(packet/'historical_timing_budget.json'),'formal_plan_present':False,'actual_correctness_and_review_still_required':True,'no_future_correctness_proof_or_hash_invented':True}
 for root in [packet,unit]:
  assert all(len(part.encode())<=255 for path in root.rglob('*') for part in path.parts)
  allpaths.extend(path for path in root.rglob('*') if path.is_file())
 files=[{'path':str(path.relative_to(repo)),'absolute_path':str(path),'bytes':path.stat().st_size,'sha256':digest(path)} for root in [packet,unit] for path in sorted(root.rglob('*')) if path.is_file()]
 freeze={'schema_version':1,'kind':'common_a4500_prepared_packet_exact_artifact_roster','packet':str(packet),'unit':str(unit),'prepared_at_utc':datetime.now(timezone.utc).isoformat(),'cpu_job':'6782816','cpu_step':os.environ['SLURM_STEP_ID'],'details':info,'files':files,'file_count':len(files),'scientific_credit':0,'case_credit':0,'idea_credit':0,'admission_granted':False,'job_submitted':False}
 f=packet/'immutable_artifacts.json';roster=put(f,freeze);allpaths.append(f);index.append({'packet':str(packet),'unit':str(unit),'copies':item['copies'],'mode':item['mode'],'immutable_roster':roster,'files':len(files),'details':info})
owner=Path(v['packets'][0]['packet']);summary={'schema_version':1,'kind':'frozen_unconfirmed_common_a4500_correctness_and_planned_confirmation_preparation','passed':True,'at_utc':datetime.now(timezone.utc).isoformat(),'cpu_job':'6782816','cpu_step':os.environ['SLURM_STEP_ID'],'records':index,'source_core_and_helpers_unchanged':'exact published E5AF and D28; no integration tracked source/index/canonical edits','confirmation_formal_units_prepared':0,'correctness_formal_units_prepared':2,'new_jobs_submitted':0,'new_gpu_operations':0,'correctness_credit':0,'confirmation_credit':0,'case_credit':0,'idea_credit':0,'admission_granted':False,'independent_review_required_before_any_submission_or_release':True}
summary_path=owner/'frozen_preparation_summary.json';put(summary_path,summary);allpaths.append(summary_path)
files=[{'path':str(path.relative_to(repo)),'absolute_path':str(path),'bytes':path.stat().st_size,'sha256':digest(path)} for path in sorted(set(allpaths))]
roster=owner/'publication_roster.json';put(roster,{'schema_version':1,'kind':'unconfirmed_common_a4500_preparation_publication_roster','files':files,'file_count':len(files),'summary':ref(summary_path),'only_new_disjoint_packet_and_data_paths':True,'no_actual_submit_release_or_gpu':True})
print(json.dumps({'passed':True,'summary':ref(summary_path),'publication_roster':ref(roster),'files':len(files),'cpu_step':os.environ['SLURM_STEP_ID'],'packets':index}))
