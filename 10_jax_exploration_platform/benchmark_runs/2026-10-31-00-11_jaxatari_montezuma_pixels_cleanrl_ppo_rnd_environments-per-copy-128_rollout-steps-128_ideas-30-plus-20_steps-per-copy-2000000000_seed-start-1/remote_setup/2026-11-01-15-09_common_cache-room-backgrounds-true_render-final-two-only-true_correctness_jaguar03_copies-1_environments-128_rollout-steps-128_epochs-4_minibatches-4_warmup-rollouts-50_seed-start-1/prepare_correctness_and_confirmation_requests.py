"""Prepare exact full correctness units and generic later timing requests on owned CPU only."""
import ctypes
from datetime import datetime,timezone
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

INDEX=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-15-09_unconfirmed-common-a4500-preparation_index.json')
RECORDS=json.loads(INDEX.read_bytes())
PACKETS=RECORDS['packets']
RUN=Path(PACKETS[0]['packet']).parent.parent
OWNER=Path(__file__).absolute().parent
BLOCKED=[]
assert os.environ.get('SLURM_JOB_ID')=='6782816' and os.environ.get('SLURM_STEP_ID','').isdecimal()
os.environ['GIT_OPTIONAL_LOCKS']='0'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return {'path':str(path),'sha256':sha(path)}
def doc(path):return json.loads(Path(path).read_bytes())
def put(path,value):
 raw=value if isinstance(value,bytes) else (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return ref(path)
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy'}:BLOCKED.append(name);raise RuntimeError('numerical import forbidden in preparation')
sys.meta_path.insert(0,NoNumerical())
sys.addaudithook(lambda event,args:(_ for _ in ()).throw(RuntimeError('library or network operation forbidden')) if event in {'socket.connect','socket.bind','socket.getaddrinfo'} or event=='ctypes.dlopen' and args[0] is not None else None)

try:
 common=Path(RECORDS['common_inputs']['directory'])
 assert sha(common/'readiness_receipt.json')==RECORDS['common_inputs']['readiness_receipt_sha256']=='deaa0cfe7c26b2dece2fd6c509ca5de7974a79cc03882a7be6d39a1aeaf1aaf7'
 source=Path(PACKETS[0]['helper_root']);S=source/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
 assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==PACKETS[0]['helper_commit']
 before=subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=all'],text=True);assert before.strip()==''
 sys.path.insert(0,str(S));import measure_selected as selected
 sys.path.insert(0,str(S));import configured_measurement as support
 qualified=RECORDS['actual_qualification'];qresult=Path(qualified['target_qualification_result']);qreview=Path(qualified['target_qualification_actual_independent_review']);qplan=doc(qresult.parent/'measurement_plan.json')
 assert sha(qresult)==qualified['target_qualification_result_sha256']=='0c4f94c2d58fdd7277a261481d7a52c4bb5b19e42612a8eae67dff6890c62cd1'
 assert sha(qreview)==qualified['target_qualification_actual_independent_review_sha256']=='004fcae1c5f24dec23da5add7452486c2ed5c1fcec89bf853617791a36fee74a'
 exitproof=RECORDS['initial_completion']
 assert sha(exitproof['actual_120_raw_audit'])==exitproof['raw_audit_sha256'] and sha(exitproof['worker_stop_receipt'])==exitproof['worker_stop_sha256'] and sha(exitproof['exit_plan'])==exitproof['exit_plan_sha256']
 assert sha(exitproof['physical_evidence_capture_manifest'])==exitproof['physical_evidence_capture_manifest_sha256']
 resources={}
 for label,command in [('node',['scontrol','show','node','jaguar03','-o']),('partition',['scontrol','show','partition','gpu','-o']),('reservation',['scontrol','show','reservation','sl5nw_168','-o'])]:
  query=subprocess.run(command,capture_output=True,text=True,timeout=30,check=False)
  resources[label]={'command':command,'returncode':query.returncode,'stdout':query.stdout,'stderr':query.stderr};assert query.returncode==0
 assert 'State=UP' in resources['partition']['stdout'] and 'jaguar03' in resources['reservation']['stdout'] and 'sl5nw' in resources['reservation']['stdout']
 resource_receipt=put(OWNER/'current_resource_observation.json',{'schema_version':1,'at_utc':datetime.now(timezone.utc).isoformat(),'actual_cpu_job':'6782816','actual_cpu_step':os.environ['SLURM_STEP_ID'],'queries':resources,'GPU_operations':0})
 protocol=doc(common/'comparison_protocol.json')
 completed=[]
 for item in PACKETS:
  packet=Path(item['packet']);copies=item['copies'];unit=Path(item['data_unit']);entry=next(v for v in protocol['configuration_inputs'] if v['policy_copies']==copies)
  candidate=doc(entry['unconfirmed_candidate']['path']);original=doc(entry['reference']['path'])
  assert sha(entry['unconfirmed_candidate']['path'])==entry['unconfirmed_candidate']['sha256'] and sha(entry['reference']['path'])==entry['reference']['sha256']
  assert candidate['learner']==original['learner'] and candidate['execution']==original['execution']
  assert {k:(original['environment'][k],candidate['environment'][k]) for k in original['environment'] if original['environment'][k]!=candidate['environment'][k]}=={'cache_room_backgrounds':(False,True),'render_final_two_only':(False,True)}
  for name,value in [('candidate.json',candidate),('reference.json',original),('runtime.json',doc(common/'runtime_environment.json')),('assets.json',qplan['asset_record']),('profile.json',qplan['configured_isolation']['profile'])]:put(packet/name,value)
  inputs={'initial_raw120_audit':ref(exitproof['actual_120_raw_audit']),'initial_worker_stop':ref(exitproof['worker_stop_receipt']),'initial_exit_plan':ref(exitproof['exit_plan']),'captured_original_evidence_manifest':ref(exitproof['physical_evidence_capture_manifest']),'common_input_readiness':ref(common/'readiness_receipt.json'),'actual_target_qualification_result':ref(qresult),'actual_target_qualification_independent_review':ref(qreview),'source_commit':item['numerical_commit'],'helper_commit':item['helper_commit'],'no_new_hypothesis_selection':True,'new_correctness_credit':0,'new_confirmation_credit':0,'case_credit':0,'idea_credit':0,'admission_granted':False}
  put(packet/'launch_prerequisites.json',inputs)
  shape={'nodes':1,'ntasks':1,'cpus_per_task':8,'host_memory_gib':64,'typed_gpu_request':'gpu:nvidia_rtx_a4500:1','node':'jaguar03','partition':'gpu','account':'sl5nw_base','qos':'csresnolim','reservation_at_preparation':'sl5nw_168','requeue':False,'walltime':'03:00:00' if item['mode']=='correctness' else None}
  if item['mode']=='correctness':
   args=selected.parse_arguments(['--prepare-only','--mode','correctness','--source-root',item['numerical_root'],'--source-commit',item['numerical_commit'],'--configuration',str(packet/'candidate.json'),'--reference-configuration',str(packet/'reference.json'),'--runtime-environment',str(packet/'runtime.json'),'--asset-record',str(packet/'assets.json'),'--jaxtari-source-root',qplan['jaxtari_source_root'],'--run-dir',str(RUN),'--unit-id',unit.name,'--hardware-class','jaguar03_rtx_a4500','--seed-start','1','--rounds','7','--minimum-updates-per-round','10','--configured-isolation-profile',str(packet/'profile.json'),'--target-qualification-record',str(qresult),'--target-qualification-review',str(qreview),'--python','/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python'])
   actual=selected.prepare(args);assert actual==unit
   plan=doc(unit/'measurement_plan.json');assert plan['mode']=='correctness' and plan['rounds']==7 and plan['iterations']==10 and plan['minimum_updates_per_round']==10 and plan['gpu_uuid'] is None and plan['copy_seeds']==list(range(1,copies+1))
   assert 'target_qualification' not in plan and 'completed_study' not in plan and 'final_confirmation_protocol' not in plan and 'device_memory_limit_protocol' not in plan
   assert plan['configuration']==candidate and plan['reference_configuration']==original and plan['helpers']['source_commit']==item['helper_commit'] and len(plan['helpers']['files'])==20
   put(packet/'resolved_measurement_plan.json',plan)
   manifest={**item,'formal_plan_prepared':True,'measurement_plan':ref(unit/'measurement_plan.json'),'configuration_inputs':{k:ref(packet/(k+'.json')) for k in ['candidate','reference','runtime','assets','profile']},'resource_shape':shape,'actual_resource_observation':resource_receipt,'launch_prerequisites':ref(packet/'launch_prerequisites.json'),'actual_target_qualification_matching_scope':'same D28 numerical source,hardware class,runtime,assets and configured-isolation machine profile; distinct fresh owned process and physical binding remain mandatory','new_correctness_credit':0,'jobs_submitted':False}
   put(packet/'manifest.yaml',manifest);put(packet/'config_resolved.yaml',{'candidate':candidate,'reference':original,'runtime_environment':doc(packet/'runtime.json'),'resource_shape':shape})
   completed.append({'mode':item['mode'],'copies':copies,'packet':str(packet),'unit':str(unit),'plan':ref(unit/'measurement_plan.json'),'helper_files':len(plan['helpers']['files'])})
  else:
   request={**item,'resource_shape':shape,'walltime_state':'unresolved until actual correctness plus initial full-round/warmup/compile evidence supports a budget within actual partition and reservation','reference':original,'candidate':candidate,'timing_protocol':{'rounds':7,'full_updates_per_arm_per_round':10,'alternating_arm_order':protocol['variant_orders'],'normalization_warmup_rollouts':50,'original_total_timesteps':2000000000,'measurement_scope':'full training iteration execution; whole execution/report windows','requires_separate_actual_correctness_and_independent_review':True},'actual_qualification_inputs':{'result':ref(qresult),'review':ref(qreview)},'formal_confirmation_plan_present':False,'correctness_receipt_not_yet_available':True,'no_future_hash_or_placeholder_is_a_proof':True}
   put(packet/'confirmation_request.json',request)
   put(packet/'config_resolved.yaml',request)
   put(unit/'config_resolved.yaml',request)
   completed.append({'mode':item['mode'],'copies':copies,'packet':str(packet),'unit':str(unit),'formal_plan_prepared':False})
 assert subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=all'],text=True)==before and not BLOCKED
 receipt={'schema_version':1,'kind':'unconfirmed_common_a4500_formal_correctness_and_generic_confirmation_preparation','passed':True,'at_utc':datetime.now(timezone.utc).isoformat(),'actual_cpu_job':'6782816','actual_cpu_step':os.environ['SLURM_STEP_ID'],'source_root':str(source),'source_commit':PACKETS[0]['helper_commit'],'source_clean_before_after':True,'records':completed,'source_numeric_operations':0,'numerical_imports':0,'GPU_operations':0,'jobs_submitted':0,'canonical_files_edited':False,'correctness_credit':0,'confirmation_credit':0,'case_credit':0,'idea_credit':0,'admission_granted':False}
 print(json.dumps({'passed':True,'receipt':put(OWNER/'formal_correctness_preparation_receipt.json',receipt),'step':os.environ['SLURM_STEP_ID']}))
except BaseException:
 put(OWNER/'preparation_failure.json',{'traceback':traceback.format_exc(),'actual_cpu_job':'6782816','actual_cpu_step':os.environ['SLURM_STEP_ID'],'GPU_operations':0,'canonical_files_edited':False});raise
