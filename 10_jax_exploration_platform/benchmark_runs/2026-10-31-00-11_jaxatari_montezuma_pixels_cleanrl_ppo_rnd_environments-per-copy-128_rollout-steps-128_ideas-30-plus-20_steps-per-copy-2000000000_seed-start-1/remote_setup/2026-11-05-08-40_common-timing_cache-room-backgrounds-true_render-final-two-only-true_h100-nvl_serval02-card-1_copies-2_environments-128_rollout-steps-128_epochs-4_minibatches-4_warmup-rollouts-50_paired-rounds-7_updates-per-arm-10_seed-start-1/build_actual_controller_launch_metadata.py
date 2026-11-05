"""Freeze actual completed metadata for root review; never import/execute a controller or tool."""
import ast
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import stat
import time

P=Path(__file__).resolve(strict=True).parent;R=P.parent.parent;I=R.parents[2];U=R/'data'/P.name
EXPECTED_PLAN='40e85dafdecffaef7c3f9e28b91d8b9f5bea97a09750d822768b67d49a451da3'
APPROVAL='df55b20ea77ab5a4c775c2b32f286f3953655c2f06b3cf7f6489e5a93a3ce141'

def require(value,message):
 if not value:raise ValueError(message)

def raw(path):
 path=Path(path);require(path.resolve(strict=True)==path,'exact ordinary source path required')
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 try:
  before=os.fstat(fd);require(stat.S_ISREG(before.st_mode) and before.st_uid==2618919 and 0<=before.st_size<=4*1024*1024,'bounded owned metadata only')
  with os.fdopen(fd,'rb',closefd=False) as f:value=f.read()
  after=os.fstat(fd)
  require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) and len(value)==before.st_size,'metadata changed during read')
  return value
 finally:os.close(fd)

def sha(path):return hashlib.sha256(raw(path)).hexdigest()
def ref(path):return {'path':str(path),'sha256':sha(path)}
def read(path):return json.loads(raw(path))
def publish(path,value):
 path=Path(path);temporary=path.with_name(path.name+'.prepared')
 require(not os.path.lexists(path) and not os.path.lexists(temporary),'preserve prior metadata publication attempt: '+path.name)
 data=value if type(value) is bytes else (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
 with temporary.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.link(temporary,path);fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:os.fsync(fd);temporary.unlink();os.fsync(fd)
 finally:os.close(fd)
 return ref(path)

def main():
 require(os.getuid()==os.geteuid()==2618919 and os.environ.get('SLURM_JOB_ID')=='6821437'
  and os.environ.get('SLURM_STEP_ID','').isdigit(),'exact owned numeric CPU metadata step required')
 require(not any(os.path.lexists(P/name) for name in ('controller_source_freeze.json','planned_launch_command_preparation.json','actual_launch_command_preparation.json','root_start_controller_once.py')),'root launch metadata must still be fresh')
 start=time.time();plan=read(U/'measurement_plan.json');request=read(P/'preparation_request.json')
 require(ref(U/'measurement_plan.json')['sha256']==EXPECTED_PLAN and plan['mode']=='confirmation' and plan['copy_seeds']==[1,2]
  and plan['configuration']['learner']['n_copies']==2 and plan['rounds']==7 and plan['iterations']==10 and plan['minimum_updates_per_round']==10,'actual copy-two plan/protocol changed')
 require(plan['source_commit']=='d28e6735e79788f618acb0ebfde8609db1f776af' and plan['helpers']['source_commit']=='69f1b5075530418adbe02c1e433c5647848a61ba' and len(plan['helpers']['files'])==29,'chosen numerical/helper source changed')
 binding=read(P/'controller_bound_metadata.json');packet=read(P/'packet_metadata_freeze.json');mapping=read(P/'canonical_controller_source_mapping.json')
 require(binding['plan']==ref(U/'measurement_plan.json') and binding['packet_freeze']==ref(P/'packet_metadata_freeze.json'),'actual bound plan or packet changed')
 for key in ('plan','preparation','software','packet_freeze','request','serial_predecessor_exit'):require(ref(Path(binding[key]['path']))==binding[key],'actual bound input changed: '+key)
 for path,expected in packet['files'].items():require(sha(Path(path))==expected,'actual frozen packet bytes changed')
 require(len(packet['files'])==53,'actual complete packet metadata closure changed')
 for source,entry in mapping['files'].items():require(sha(Path(source))==entry['sha256']==sha(Path(entry['staged'])),'canonical and staged source differ')
 approval=I/'.review_metadata/timing_predecessor_schema/independent_request_controller_observer_review.json'
 require(sha(approval)==APPROVAL and read(approval)['passed'] is True,'genuine independent source approval changed')
 claim_path=P/'root_actual_current_admission_claim_probe_stdout.raw';claim=read(claim_path)
 require(claim['kind']=='cpu_shared_direct_current_admission_probe' and claim['passed'] is True and claim['actual_prepared_unit_checked'] is True
  and claim['canonical_source_checked'] is True and claim['fixture_only'] is False and claim['gpu_target_checked'] is False
  and claim['plan']==ref(U/'measurement_plan.json') and claim['helpers']==plan['helpers']
  and claim['claim']['source_commit']==plan['source_commit'] and claim['claim']['frozen_admission']['sha256']=='4a71ef2b245ae784933c55bb2b5159aa1bb7f8f626eb94b15390e2f9a304ee7c','actual prepared-unit admission probe failed')
 phases={}
 for phase in ('prepare','verify','current_admission_claim_probe','bind'):
  path=P/('root_actual_'+phase+'_return.json');value=read(path)
  require(type(value['returncode']) is int and value['returncode']==0 and type(value['seconds']) in (int,float)
   and math.isfinite(value['seconds']) and 0<=value['seconds']<=300,'actual bounded metadata phase did not pass: '+phase)
  phases[phase]=ref(path)
 creation_path=P/'root_actual_controller_log_directory_created.json';creation=read(creation_path)
 visibility_path=P/'root_actual_log_directory_target_visibility_stdout.raw';visibility=read(visibility_path)
 visibility_return=P/'root_actual_log_directory_target_visibility_return.json'
 logs=R/'slurm/direct_jobs'/P.name
 require(creation['kind']=='actual_root_created_owned_controller_log_directory' and creation['directory']==str(logs)
  and creation['uid']==2618919 and creation['controller_stdout_stderr_absent'] is True,'actual owned log-directory creation missing')
 require(read(visibility_return)['returncode']==0 and visibility['host']=='serval02' and visibility['uid']==2618919
  and visibility['directory']==str(logs) and visibility['directory_is_owned'] is True
  and visibility['directory_writable_and_searchable'] is True and visibility['stdout_stderr_absent'] is True
  and visibility['standing_gpu_lock_operations']==0,'actual target log-directory visibility failed')
 log_standards={}
 for entry in creation['standards']:
  path=Path(entry['path']);require(path==logs/entry['name'] and sha(path)==entry['sha256']==visibility['files'][entry['name']],
   'actual log standard differs from created/target bytes')
  log_standards[entry['name']]=ref(path)
 require(set(log_standards)=={'manifest.yaml','config_resolved.yaml','command.txt','experiment_background.md'},'all four actual log standards required')
 observer=P/'root_exact_serial_predecessor_exit_observation.json'
 require(ref(observer)==read(P/'serial_predecessor_exit_inputs.json')['exit_observation'],'actual fresh serial observer reference changed')
 protected={}
 for section in ('core','paired_source'):
  for relative,entry in plan[section]['files'].items():
   expected=entry['sha256'] if type(entry) is dict else entry;selected=Path(plan[section]['source_root'])/relative;current=I/relative
   require(sha(selected)==expected==sha(current),'actual protected numerical source changed')
   protected[relative]={'selected':ref(selected),'publication_path':ref(current)}
 require(len(protected)==13,'all thirteen original numerical files required')
 for relative,expected in plan['helpers']['files'].items():require(sha(U/'code'/Path(relative).name)==expected==sha(Path(plan['helpers']['source_root'])/relative),'selected helper checkout/staging changed')
 new=ast.parse(raw(P/'direct_qualification_controller.py'));old=ast.parse(raw(I/'.review_metadata/timing_predecessor_schema/controller_published_d225.py'))
 functions=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
 a,b=functions(old),functions(new);changed=[name for name in a if a[name]!=b[name]]
 require(set(a)==set(b) and changed==['check_serial_predecessor_inputs'],'only the approved serial predicate may change')
 publication={'canonical_controller_commit':'cba869200cb31131b354c985f77b43386f90f9b9',
  'source_publication':read(P/'root_actual_prepare_intent.json')['request_publication']}
 require(publication['source_publication']=='444c84e3ac07052569cd6758142c3aa0a54da22e','actual request/source publication changed')
 startup={'kind':'reviewed_bounded_child_startup_observation_protocol','controller_source_commit':publication['canonical_controller_commit'],
  'controller_source_sha256':sha(P/'direct_qualification_controller.py'),'maximum_startup_seconds':30,
  'wait_only_for_truly_empty_raw_command_bytes':True,'every_snapshot_requires_same_actual_owned_PID_birth_parent_affinity_and_raw_hashes':True,
  'exact_complete_expected_raw_command_required_before_registration':True,'wrong_nonempty_or_changed_identity_or_exit_or_deadline_refuses':True,
  'independent_source_review':ref(approval),'changed_existing_controller_functions':changed,'other_controller_functions_unchanged':len(a)-1,
  'numerical_and_helper_source_unchanged':True,'new_scientific_or_hardware_admission':False}
 publish(P/'controller_startup_protocol.json',startup)
 # These are archival regression/source-review copies, never executed by this builder.
 for source,destination in ((approval,P/'serial_timing_predecessor_source_independent_review.json'),
  (I/'10_jax_exploration_platform/scripts/atari_pixel_campaign/tests/test_timing_predecessor_actual_schema.py',P/'test_timing_predecessor_actual_schema.py'),
  (I/'10_jax_exploration_platform/scripts/atari_pixel_campaign/tests/test_empty_command_startup.py',P/'test_empty_command_startup.py'),
  (I/'10_jax_exploration_platform/scripts/atari_pixel_campaign/tests/test_initial_child_identity.py',P/'test_initial_child_identity.py')):publish(destination,raw(source))
 files=set(Path(name) for name in packet['files'])
 files.update(path for path in P.iterdir() if path.is_file() and not path.name.endswith('.prepared'))
 files.update(I/relative for relative in protected)
 files.update(Path(value['path']) for value in log_standards.values())
 require(all(path.is_relative_to(I) for path in files),'publication file escaped integration')
 frozen={'schema_version':1,'kind':'direct_qualification_controller_source_freeze','canonical_source_commit':publication['canonical_controller_commit'],
  'canonical_source_publication':publication['source_publication'],'baseline_source_commit':'d2253fa853290cb8fcd0a9212e332dd33a63f2f9',
  'files':{str(path):sha(path) for path in sorted(files)},'actual_plan':ref(U/'measurement_plan.json'),
  'actual_packet_freeze':ref(P/'packet_metadata_freeze.json'),'controller_binding':ref(P/'controller_bound_metadata.json'),
  'independent_source_approval':ref(P/'serial_timing_predecessor_source_independent_review.json'),
  'actual_prepared_unit_claim_probe':ref(claim_path),'actual_metadata_phase_returns':phases,
  'actual_fresh_serial_predecessor_exit_observation':ref(observer),
  'preserved_old_failed_attempt':ref(P/'preserved_07_20_failure_references.json'),
  'actual_log_directory_creation':ref(creation_path),'actual_target_log_directory_visibility':ref(visibility_path),
  'actual_target_log_directory_visibility_return':ref(visibility_return),'actual_log_standards':log_standards,
  'scope':{'changed_existing_controller_functions':changed,'unchanged_other_controller_function_count':len(a)-1,
   'entire_controller_source_unchanged':False,'original_serial_one_correctness_branch_unchanged':True,
   'serial_two_actual_result_bound_mib':64,'global_metadata_bound_mib_unchanged':16,'maximum_startup_seconds':30,
   'all_NUM_and_helper69f_and_scientific_details_unchanged':True,'no_actual_metadata_phase_replayed':True},
  'protected_numerical_source_roster':protected,'root_launch_review_pending':True,'controller_owns_gpu_standing_descriptor':False,
  'numerical_and_remote_actions_by_freeze':0,'source_files_equal_reviewed_canonical_bytes':True}
 freeze=publish(P/'controller_source_freeze.json',frozen)
 command_prefix=['/usr/bin/env','-i','PATH=/usr/local/bin:/usr/bin:/bin','LANG=C.UTF-8','TZ=UTC',
  'CUDA_VISIBLE_DEVICES=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3','PYTHONDONTWRITEBYTECODE=1','PYTHONNOUSERSITE=1',
  '/usr/bin/python3','-I','-S','-B',str(P/'direct_qualification_controller.py'),'--launch','--publication-commit']
 command=[plan['python'],str(U/'code/measure_selected.py'),'--prepared-unit',str(U)]
 require(shlex.split(raw(U/'command.txt').decode())==command,'actual producer command differs')
 planned={'schema_version':1,'kind':'planned_common_timing_controller_command_awaiting_actual_publication',
  'host':'serval02','publication':None,'command':None,'command_prefix':command_prefix,
  'required_final_argument_source':'root actual inner publication including exact freeze plus new independent root launch review',
  'future_actual_binding_file':str(P/'actual_launch_command_preparation.json'),'future_actual_binding_must_be_created_once_after_real_publication':True,
  'plan':ref(U/'measurement_plan.json'),'packet_freeze':ref(P/'packet_metadata_freeze.json'),'controller_source_freeze':freeze,
  'controller_bound_metadata':ref(P/'controller_bound_metadata.json'),'source_approval':ref(P/'serial_timing_predecessor_source_independent_review.json'),
  'controller_review':None,'producer_command':command,'actual_controller_pid':None,'actual_numerical_child_pid':None,
  'actual_log_directory_creation':ref(creation_path),'actual_target_log_directory_visibility':ref(visibility_path),'actual_log_standards':log_standards,
  'stdout':str(R/'slurm/direct_jobs'/P.name/'controller.stdout'),'stderr':str(R/'slurm/direct_jobs'/P.name/'controller.stderr'),
  'automatic_retry_permitted':False,'controller_owns_gpu_standing_descriptor':False,'producer_owns_gpu_standing_descriptor':True,
  'actual_prepare_verify_claim_bind_actions_by_builder':0,'remote_actions':0,'scientific_credit':0,'case_credit':0,'idea_credit':0}
 planned_ref=publish(P/'planned_launch_command_preparation.json',planned)
 receipt=publish(P/'root_launch_metadata_preparation_receipt.json',{'kind':'actual_two_copy_bound_controller_launch_metadata_preparation',
  'passed':True,'source_freeze':freeze,'planned_launch_command':planned_ref,'actual_phase_invocations':0,'source_file_count':len(files),
  'owned_cpu_job':os.environ['SLURM_JOB_ID'],'owned_cpu_step':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'host':os.uname().nodename,
  'started_at_seconds':start,'finished_at_seconds':time.time(),'no_controller_default_invoked':True,'no_scientific_consumers_replayed':True,
  'actual_launch_command_preparation_absent':not os.path.lexists(P/'actual_launch_command_preparation.json'),'root_starter_absent':True,
  'numerical_or_gpu_or_remote_actions':0})
 print(json.dumps({'source_freeze':freeze,'planned_command':planned_ref,'receipt':receipt,'file_count':len(files),'elapsed_seconds':time.time()-start},sort_keys=True))

if __name__=='__main__':main()
