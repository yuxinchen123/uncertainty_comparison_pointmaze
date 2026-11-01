"""Join actual terminal result labels and time bounds to its frozen plan and recorded process."""
from datetime import datetime,timezone
import hashlib,json,math,os
from pathlib import Path

AUDIT=Path(__file__).absolute().parent
PACKET=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-13-50_original-qualification_explicit-pytest-root_jaguar03_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-3_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-1')
UNIT=PACKET.parent.parent/'data'/PACKET.name
assert os.environ.get('SLURM_JOB_ID')=='6781231' and os.environ.get('SLURM_STEP_ID','').isdecimal()
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def doc(p):return json.loads(Path(p).read_bytes())
def ref(p):return {'path':str(p),'sha256':digest(p)}
checks=[]
def check(v,n):assert v,n;checks.append(n)
def finite(v):return type(v) in (int,float) and math.isfinite(v)
result_path=UNIT/'target_qualification_result.json';before=digest(result_path)
r=doc(result_path);p=doc(UNIT/'measurement_plan.json');m=doc(UNIT/'configured_isolation_metadata.json');b=doc(UNIT/'resolved_device_binding.json')
check(r['copy_seeds']==p['copy_seeds'] and len(r['copy_seeds'])==p['configuration']['learner']['n_copies'] and all(type(x) is int and 0<=x<2**32 for x in r['copy_seeds']),'actual ordered independent copy seeds equal the frozen plan')
check(p['configuration']==p['reference_configuration'],'qualified original configuration and reference equal exactly')
config_fields={}
for name,expected in [('configuration',p['configuration']),('candidate',p['configuration']),('reference',p['reference_configuration'])]:
 config_fields[name]={'present':name in r}
 if name in r:check(r[name]==expected,'actual result '+name+' equals its frozen configuration');config_fields[name]['exact_match']=True
for name in ['source_commit','hardware_class','runtime_environment','asset_record','helpers','configured_isolation']:
 check(r[name]==p[name],'actual result '+name+' label equals its frozen plan')
check(r['measurement_mode']==p['mode']=='correctness','actual result mode equals original qualification correctness')
check(r['protocol']=={'rounds':p['rounds'],'requested_minimum_updates_per_round':p['minimum_updates_per_round'],'actual_updates_per_round':p['iterations']},'generic prepared protocol labels equal plan; dedicated qualification controlled counters remain the work authority')
identity=r['execution_identity'];process=m['process']
for name in ['pid','hostname','process_start_time']:check(identity[name]==process[name],'actual execution '+name+' equals collected current-process metadata')
check(type(identity['pid']) is int and identity['pid']>0 and type(identity['process_start_time']) is str and identity['process_start_time'].isdecimal() and int(identity['process_start_time'])>0,'actual execution PID and birth have canonical positive types')
check(b['process']==process and b['native']['pid']==identity['pid'] and r['loaded_gpu_libraries']['pid']==identity['pid'],'binding/native/loaded-library records belong to the same actual process')
for name in ['pid','hostname','process_start_time']:
 if name in r['hardware']:check(r['hardware'][name]==process[name],'actual hardware '+name+' matches current-process metadata')
check(r['hardware']['gpu_uuid']==b['gpu_uuid'],'actual hardware UUID equals same-process native physical binding')
start,end=identity['started_at_seconds'],identity['finished_at_seconds']
check(finite(start) and finite(end) and 0<start<end,'actual execution timestamps are finite positive and ordered')
w=r['qualification_memory']['assigned_gpu_process_observation'];ws,we=w['window_started_at_seconds'],w['window_finished_at_seconds']
check(finite(ws) and finite(we) and start<=ws<we<=end,'complete observed qualification window lies inside actual execution interval')
check(w['pid']==identity['pid'] and w['gpu_uuid']==b['gpu_uuid'],'qualification sampler uses actual current PID and physical UUID')
for sample in w['samples']:check(finite(sample['timestamp_seconds']) and ws<=sample['timestamp_seconds']<=we,'actual sample time lies inside recorded qualification window')
check(all(r[k] is False and b[k] is False for k in ['physical_gres_mapping_verified','kernel_device_enforcement_verified','installed_binary_matches_upstream_verified','worker_admitted','admission_granted']),'all physical trust and worker/admission false flags remain false')
check(digest(result_path)==before,'actual terminal result unchanged during extra independent joins')
value={'schema_version':1,'kind':'actual_full_original_qualification_extra_label_process_and_time_review','passed':True,'at_utc':datetime.now(timezone.utc).isoformat(),'target_job_id':'6782778','audit_job_id':'6781231','audit_step_id':os.environ['SLURM_STEP_ID'],'actual_result':ref(result_path),'actual_plan':ref(UNIT/'measurement_plan.json'),'actual_metadata':ref(UNIT/'configured_isolation_metadata.json'),'actual_binding':ref(UNIT/'resolved_device_binding.json'),'checks':checks,'check_count':len(checks),'configuration_field_presence':config_fields,'copy_seeds':r['copy_seeds'],'process_identity':identity,'qualification_window':{'started_at_seconds':ws,'finished_at_seconds':we,'pid':w['pid'],'gpu_uuid':w['gpu_uuid']},'unmodified_full_raw_reader_still_required':True,'no_validator_or_tolerance_changed':True,'scientific_credit':0,'case_credit':0,'idea_credit':0,'final_training_credit':0,'site_qualification_credit':0,'admission_granted':False,'worker_admitted':False,'gpu_operations':0,'scheduler_queries':0,'network_calls':0,'numerical_imports':0}
out=AUDIT/'terminal_result_extra_joins.json'
with out.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'passed':True,'receipt':ref(out),'audit_step':os.environ['SLURM_STEP_ID'],'checks':len(checks)}))
