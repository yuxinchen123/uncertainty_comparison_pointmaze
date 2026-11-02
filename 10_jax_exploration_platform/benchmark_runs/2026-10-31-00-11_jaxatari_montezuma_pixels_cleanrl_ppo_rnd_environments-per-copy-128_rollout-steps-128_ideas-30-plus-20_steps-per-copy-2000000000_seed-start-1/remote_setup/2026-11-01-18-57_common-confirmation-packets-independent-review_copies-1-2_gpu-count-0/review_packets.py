RUN='/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1'
OUTPUT='/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-18-57_common-confirmation-packets-independent-review_copies-1-2_gpu-count-0'
"""Independently validate both frozen intermediate packets without job or GPU actions."""
import ast,builtins,ctypes,datetime,fcntl,hashlib,importlib.util,io,json,os,pathlib,shlex,stat,subprocess,sys,unittest
from unittest import mock
R=pathlib.Path(RUN);O=pathlib.Path(OUTPUT);F=R/'remote_setup/2026-11-01-19-00_common-confirmation_prepared-packets_copies-1-2_rounds-7_updates-per-arm-10.json'
assert os.environ['SLURM_JOB_ID']=='6783003' and os.environ['SLURM_STEP_ID'].isdigit()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p.read_bytes())}
def read(p):return json.loads(p.read_bytes())
def save(name,value):
 p=O/name;raw=value if isinstance(value,bytes) else (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
 with p.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
 return ref(p)
assert ref(F)['sha256']=='44d4c779c7d29a5e708f0c57ae5e4c4a9cf419382a945c6c0121e6c77200ccd0'
frozen=read(F);assert frozen['count']==len(frozen['files'])==128 and frozen['passed'] is True
def images():
 result={}
 for name in frozen['files']:
  p=pathlib.Path(name);s=p.lstat();assert stat.S_ISREG(s.st_mode) and p.resolve(strict=True)==p
  raw=p.read_bytes();result[name]={'bytes':len(raw),'mode':stat.S_IMODE(s.st_mode),'sha256':sha(raw)}
 return result
before=images();assert before==frozen['files']
NUM={'jax','jaxlib','numpy','torch','cupy','flax','optax','scipy','jaxtari','exploration_platform','pynvml'}
original_import=builtins.__import__;blocked=[];allowed=[]
def guarded_import(name,*args,**kwargs):
 if name.split('.')[0] in NUM:blocked.append(name);raise RuntimeError('real numerical import forbidden')
 return original_import(name,*args,**kwargs)
builtins.__import__=guarded_import
real_popen=subprocess.Popen
packet_paths=[pathlib.Path(p['packet']) for p in frozen['packets']]
unit_paths=[pathlib.Path(p['unit']) for p in frozen['packets']]
class MetadataOnly(real_popen):
 def __init__(self,args,*other,**kwargs):
  cmd=list(args) if isinstance(args,(tuple,list)) else [];ok=False
  if cmd[:2]==['git','-C'] and len(cmd)>3:ok=pathlib.Path(cmd[2]).is_absolute() and cmd[3] in {'show','cat-file','rev-parse','status','symbolic-ref','merge-base','ls-tree'}
  elif cmd[:2]==['bash','-n'] and len(cmd)==3:ok=pathlib.Path(cmd[2]) in [p/'job.sbatch' for p in packet_paths]
  elif cmd[:4]==['/usr/bin/python3','-I','-S','-B'] and len(cmd)>=5:
   p=pathlib.Path(cmd[4])
   if p in [u/'code/common_confirmation.py' for u in unit_paths]:ok=cmd[5:]==['--plan-stdin']
   elif p in [p/'wait_for_owned_inputs.py' for p in packet_paths]:
    ok=len(cmd)==5 or '--unit' in cmd and pathlib.Path(cmd[cmd.index('--unit')+1]).is_relative_to('/tmp')
   elif cmd[4]=='-c' and len(cmd)==6:
    ok=cmd[5].startswith('import importlib.util,subprocess; s=') and 'm.install_metadata_guard()' in cmd[5] and 'scheduler blocked before execution' in cmd[5]
  if not ok:blocked.append(cmd);raise RuntimeError('unexpected child command: '+repr(cmd))
  allowed.append(cmd);super().__init__(args,*other,**kwargs)
subprocess.Popen=MetadataOnly
def audit(event,args):
 if event in {'socket.connect','socket.bind','os.system'} or event=='ctypes.dlopen' and args[0] is not None:raise RuntimeError('live operation forbidden: '+event)
 if event=='open' and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith(('/dev/nvidia','/dev/dri')):raise RuntimeError('device operation forbidden')
sys.addaudithook(audit)
packet_results={};extra_test_count=0
for item in frozen['packets']:
 copies=item['copies'];hours=item['hours'];p=pathlib.Path(item['packet']);u=pathlib.Path(item['unit']);plan=read(u/'measurement_plan.json')
 assert copies in (1,2) and hours=={1:2,2:3}[copies] and ref(u/'measurement_plan.json')==item['plan']
 assert plan['kind']=='selected_configuration_measurement' and plan['mode']=='confirmation' and plan['source_commit']=='d28e6735e79788f618acb0ebfde8609db1f776af'
 assert plan['helpers']['source_commit']==frozen['helper_commit']=='9fcc43a478b93d731df780805fd7c3c514cdf67f' and len(plan['helpers']['files'])==21
 assert plan['gpu_uuid'] is None and plan['configured_isolation']['physical_identity']['gpu_uuid'] is None
 assert [plan[k] for k in ['rounds','iterations','minimum_updates_per_round']]==[7,10,10] and all(type(plan[k]) is int for k in ['rounds','iterations','minimum_updates_per_round'])
 assert plan['copy_seeds']==list(range(1,copies+1)) and all(type(x) is int for x in plan['copy_seeds'])
 assert not any(k in plan for k in ['target_qualification','final_confirmation_protocol','completed_study','device_memory_limit_protocol','direct_binding','scaling_protocol'])
 changes=[(scope,k,plan['reference_configuration'][scope][k],v) for scope in ['learner','environment','execution'] for k,v in plan['configuration'][scope].items() if v!=plan['reference_configuration'][scope][k]]
 assert sorted(changes)==[('environment','cache_room_backgrounds',False,True),('environment','render_final_two_only',False,True)]
 expected=['sbatch','--parsable','--hold','--no-requeue','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=64G','--time='+str(hours).zfill(2)+':00:00','--partition=gpu','--account=sl5nw_base','--qos=csresnolim','--nodelist=jaguar03','--reservation=sl5nw_168','--gres=gpu:nvidia_rtx_a4500:1',str(p/'job.sbatch')]
 command=read(p/'held_submission_command.json')
 assert command['command']==expected and command['job_id'] is None and command['gpu_uuid'] is None and command['new_jobs_submitted']==0
 manifest=read(p/'manifest.yaml');assert manifest['job_id'] is None and manifest['gpu_uuid'] is None and manifest['submitted'] is False and manifest['pair_orders']==[['reference','candidate'] if j%2==0 else ['candidate','reference'] for j in range(7)]
 assert manifest['requested_hours']==hours and manifest['requested_cpus']==8 and manifest['requested_host_memory_gib']==64 and manifest['requested_gpu_count']==1
 assert len(item['standard_data_records'])==5 and {pathlib.Path(v['path']).name for v in item['standard_data_records']}=={'manifest.yaml','config_resolved.yaml','command.txt','experiment_background.md','progress.json'}
 for value in item['standard_data_records']:assert ref(pathlib.Path(value['path']))==value and value['path'] in before
 assert read(u/'manifest.yaml')==plan and read(u/'progress.json')['state']=='prepared'
 assert '\n\n' in (p/'experiment_background.md').read_text() and '\\n' not in (p/'experiment_background.md').read_text()
 assert (u/'measurement_prepared.json').read_bytes() and read(u/'measurement_prepared.json')=={'plan_sha256':item['plan']['sha256']}
 for relative,expected_sha in plan['helpers']['files'].items():
  helper=pathlib.Path(plan['helpers']['source_root'])/relative
  assert sha(helper.read_bytes())==expected_sha==sha((u/'code'/helper.name).read_bytes())
  assert subprocess.check_output(['git','-C',plan['helpers']['source_root'],'cat-file','blob',plan['helpers']['source_commit']+':'+relative])==helper.read_bytes()
 for group in ['core','paired_source']:
  for relative,digest in plan[group]['files'].items():
   digest=digest['sha256'] if isinstance(digest,dict) else digest
   assert sha((pathlib.Path(plan[group]['source_root'])/relative).read_bytes())==digest
 for name in ['submission_intent.json','submission_receipt.json','release_receipt.json','held_binding.json']:assert not os.path.lexists(p/name)
 for name in ['owned_allocation_inputs.json','configured_isolation_pre_release_inputs.json','common_correctness_validation_attempt.json','result.json','failure.json']:assert not os.path.lexists(u/name)
 for line in (p/'source_checksums.sha256').read_text().splitlines():
  digest,name=line.split('  ',1);assert sha(pathlib.Path(name).read_bytes())==digest and pathlib.Path(name).name!='progress.json'
 syntax=subprocess.run(['bash','-n',str(p/'job.sbatch')],capture_output=True);assert syntax.returncode==0
 spec=importlib.util.spec_from_file_location('independent_barrier_tests_'+str(copies),p/'test_wait_for_owned_inputs.py');tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
 log=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(tests.BarrierTests)
 result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
 assert result.wasSuccessful() and result.testsRun==22,log.getvalue()
 save('copies_'+str(copies)+'_independent_barrier_tests.json',{'passed':True,'tests':result.testsRun,'failures':[],'errors':[],'output':log.getvalue(),'source':ref(p/'wait_for_owned_inputs.py')})
 # Four independent coherent changes are applied only to fresh private fixture records.
 extra=[]
 for name,change in [
  ('wrong_same_copy_seed',lambda obj:obj.plan.update(copy_seeds=[2]*copies)),
  ('wrong_common_review_reference',lambda obj:obj.plan['common_correctness_inputs']['review'].update(sha256='f'*64)),
  ('reduced_timing_window',lambda obj:obj.plan.update(iterations=1)),
  ('final_evidence_field',lambda obj:obj.plan.update(completed_study={}))]:
  fixture=tests.BarrierTests();fixture.setUp()
  try:
   change(fixture);fixture.rebind_changed_plan()
   try:fixture.run_gate()
   except ValueError:extra.append(name)
   else:raise AssertionError('coherent wrong plan passed: '+name)
  finally:fixture.tearDown()
 assert len(extra)==4;extra_test_count+=4
 # Revalidate actual raw correctness in one fresh system-Python child, not in the parent.
 raw=(u/'measurement_plan.json').read_bytes();child_command=['/usr/bin/python3','-I','-S','-B',str(u/'code/common_confirmation.py'),'--plan-stdin']
 with (R/'queue/cpu_monitor_large_artifact_read.lock').open('r+') as lock:
  assert os.fstat(lock.fileno()).st_uid==os.getuid();fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
  child=subprocess.run(child_command,input=raw,capture_output=True,timeout=180,check=False)
 outref=save('copies_'+str(copies)+'_actual_prerequisite.stdout.raw',child.stdout);errref=save('copies_'+str(copies)+'_actual_prerequisite.stderr.raw',child.stderr)
 assert child.returncode==0 and child.stderr==b'',child.stderr.decode()
 proof=json.loads(child.stdout);assert proof['passed'] is True and proof['plan_sha256']==sha(raw) and proof['common_correctness_inputs']==plan['common_correctness_inputs'] and proof['parent_pid']==os.getpid() and not proof['numerical_imports'] and proof['gpu_operations']==0
 assert not NUM.intersection(sys.modules)
 packet_results[str(copies)]={'plan':item['plan'],'packet':str(p),'unit':str(u),'independent_barrier_tests':22,'extra_independent_refusals':extra,'prerequisite_stdout':outref,'prerequisite_stderr':errref,'prerequisite_child_returncode':0,'actual_prior_correctness_job_id':proof['correctness_job_id'],'actual_prior_correctness_step':proof['correctness_numeric_step'],'hours':hours,'helper_count':21,'standard_data_records':item['standard_data_records']}
after=images();assert before==after==frozen['files'] and not blocked
value={'schema_version':1,'kind':'independent_common_a4500_confirmation_packets_and_startup_review','approved':True,'passed':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cpu_job_id':'6783003','cpu_step_id':os.environ['SLURM_STEP_ID'],'packet_freeze':ref(F),'frozen_paths_checked':len(before),'before':before,'after':after,'packets':packet_results,'independent_barrier_test_count':44,'extra_independent_refusal_count':extra_test_count,'source_commit':'d28e6735e79788f618acb0ebfde8609db1f776af','helper_commit':frozen['helper_commit'],'qualification_reuse_scope':frozen['qualification_scope'],'independent_source_review':frozen['independent_source_review'],'time_budget':frozen['time_budget'],'monitor_scope':'Current79a monitor progress/status only; every scientific confirmation acceptance requires the exact newly staged reviewed consumer on owned compute.','submission_utility_approval':False,'release_action_approval':False,'new_jobs_submitted':0,'actual_job_queries':0,'actual_gpu_operations':0,'numerical_work':False,'scientific_credit':0,'case_credit':0,'idea_credit':0,'admission_granted':False,'worker_admitted':False,'allowed_metadata_children':allowed,'blocked_parent_operations':blocked}
reference=save('review.json',value);print(json.dumps({'passed':True,'approved':True,'review':reference,'frozen_paths':len(before),'barrier_tests':44,'extra_refusals':extra_test_count}))
