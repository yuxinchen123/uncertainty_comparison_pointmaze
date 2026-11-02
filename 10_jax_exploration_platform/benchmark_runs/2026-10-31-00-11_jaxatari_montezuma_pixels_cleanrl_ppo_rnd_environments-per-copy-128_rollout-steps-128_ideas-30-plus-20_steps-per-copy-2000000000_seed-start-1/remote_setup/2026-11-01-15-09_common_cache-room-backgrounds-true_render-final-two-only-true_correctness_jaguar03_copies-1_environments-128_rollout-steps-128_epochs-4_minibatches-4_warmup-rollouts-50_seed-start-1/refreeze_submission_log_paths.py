"""Verify only the authorized submission-log-path change and preserve the original freeze."""
from datetime import datetime,timezone
import hashlib,json,os,subprocess
from pathlib import Path

INDEX=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-15-09_unconfirmed-common-a4500-preparation_index.json');v=json.loads(INDEX.read_bytes());first=Path(v['packets'][0]['packet']);run=first.parent.parent;repo=run.parents[2]
assert os.environ.get('SLURM_JOB_ID')=='6782816' and os.environ.get('SLURM_STEP_ID','').isdecimal()
def image(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'sha256':image(p)}
def doc(p):return json.loads(Path(p).read_bytes())
def put(p,x):
 with Path(p).open('x') as f:json.dump(x,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return ref(p)
original=doc(first/'publication_roster.json');preserved=doc(first/'first_150_file_freeze_preservation_after_log_path_correction.json')
assert original['file_count']==150 and len(preserved['files'])==150
for logical,entry in preserved['files'].items():assert image(entry['preserved_path'])==entry['sha256']
new_records=[]
for item in v['packets']:
 packet=Path(item['packet']);unit=Path(item['data_unit'])
 if item['mode']!='correctness':continue
 before=(packet/'job_before_campaign_submission_log_paths.sbatch').read_text().splitlines();after=(packet/'job.sbatch').read_text().splitlines();assert len(before)==len(after)
 difference=[{'before':a,'after':b} for a,b in zip(before,after) if a!=b];assert len(difference)==2
 destinations={}
 for stream in ['output','error']:
  old=[line for line in before if line.startswith('#SBATCH --'+stream+'=')];new=[line for line in after if line.startswith('#SBATCH --'+stream+'=')]
  assert len(old)==len(new)==1 and old[0]!=new[0]
  destination=Path(new[0].split('=',1)[1]);assert destination.parent==run/'slurm/jobs' and '%j' in destination.name and len(destination.name.encode())<=255
  destinations[stream]=str(destination)
 assert all(row['before'].startswith(('#SBATCH --output=','#SBATCH --error=')) for row in difference)
 assert subprocess.run(['bash','-n',str(packet/'job.sbatch')],capture_output=True,check=False).returncode==0
 checksum=subprocess.run(['sha256sum','--check','--strict',str(packet/'source_checksums.sha256')],capture_output=True,check=False);assert checksum.returncode==0
 plan=unit/'measurement_plan.json';original_plan=next(row for row in original['files'] if row['absolute_path']==str(plan));assert image(plan)==original_plan['sha256']
 barrier=packet/'wait_for_owned_inputs.py';original_barrier=next(row for row in original['files'] if row['absolute_path']==str(barrier));assert image(barrier)==original_barrier['sha256']
 old_verification=doc(packet/'packet_verification.json');verified={**old_verification,'kind':'common_correctness_launch_packet_verification_after_submission_log_correction','at_utc':datetime.now(timezone.utc).isoformat(),'cpu_job':'6782816','cpu_step':os.environ['SLURM_STEP_ID'],'script':ref(packet/'job.sbatch'),'previous_script':ref(packet/'job_before_campaign_submission_log_paths.sbatch'),'previous_packet_verification':ref(packet/'packet_verification.json'),'submission_output_paths':destinations,'only_two_submission_log_directives_changed':True,'startup_barrier_and20test_results_unchanged':True,'frozen_measurement_plan_and_all_scientific_inputs_unchanged':True,'runtime_input_checksums_passed_after_log_change':True,'node_availability_scope':'Slurm scheduler IDLE/CPUAlloc/AllocMem/AllocTRES only; not proof that a physical GPU is free. Actual owned binding and own-only samples remain required.','physical_gpu_free_verified':False,'actual_job_attempt':False}
 verification=put(packet/'packet_verification_after_submission_log_correction.json',verified)
 files=[{'path':str(path.relative_to(repo)),'absolute_path':str(path),'bytes':path.stat().st_size,'sha256':image(path)} for root in [packet,unit] for path in sorted(root.rglob('*')) if path.is_file()]
 roster=put(packet/'immutable_artifacts_after_submission_log_correction.json',{'schema_version':1,'kind':'common_correctness_current_artifacts_after_submission_log_correction','packet':str(packet),'unit':str(unit),'files':files,'file_count':len(files),'current_verification':verification,'previous_freeze_preservation':ref(first/'first_150_file_freeze_preservation_after_log_path_correction.json'),'plan_unchanged':True,'source_or_barrier_unchanged':True,'gpu_operations':0,'jobs_submitted':0})
 new_records.append({'packet':str(packet),'unit':str(unit),'copies':item['copies'],'current_artifact_roster':roster,'current_verification':verification,'script':ref(packet/'job.sbatch'),'plan':ref(plan),'barrier':ref(barrier),'log_paths':destinations})
summary={'schema_version':1,'kind':'common_a4500_preparation_submission_logs_corrected_refreeze','passed':True,'at_utc':datetime.now(timezone.utc).isoformat(),'cpu_job':'6782816','cpu_step':os.environ['SLURM_STEP_ID'],'previous_summary':ref(first/'frozen_preparation_summary.json'),'previous_150_file_roster':ref(first/'publication_roster.json'),'previous_150_payload_preservation':ref(first/'first_150_file_freeze_preservation_after_log_path_correction.json'),'correctness_packets':new_records,'confirmation_requests_unchanged_and_no_formal_plans':True,'only_executable_changes':'two output/error Slurm directives per correctness script','scientific_configs_barriers_qualified_core_helpers_and_measurement_plans_changed':False,'physical_gpu_free_claimed':False,'gpu_operations':0,'actual_job_attempts':0,'canonical_files_edited':False,'scientific_credit':0,'case_credit':0,'idea_credit':0,'admission_granted':False}
summary_ref=put(first/'refrozen_preparation_summary_after_submission_log_correction.json',summary)
paths=[INDEX]
for item in v['packets']:
 for root in [Path(item['packet']),Path(item['data_unit'])]:paths.extend(p for p in root.rglob('*') if p.is_file())
files=[{'path':str(path.relative_to(repo)),'absolute_path':str(path),'bytes':path.stat().st_size,'sha256':image(path)} for path in sorted(set(paths))]
roster_ref=put(first/'publication_roster_after_submission_log_correction.json',{'schema_version':1,'kind':'common_a4500_preparation_current_publication_roster_after_log_path_fix','files':files,'file_count':len(files),'summary':summary_ref,'previous_freeze_preserved':True,'gpu_operations':0,'jobs_submitted':0})
print(json.dumps({'passed':True,'summary':summary_ref,'publication_roster':roster_ref,'file_count':len(files),'records':new_records,'cpu_step':os.environ['SLURM_STEP_ID']}))
