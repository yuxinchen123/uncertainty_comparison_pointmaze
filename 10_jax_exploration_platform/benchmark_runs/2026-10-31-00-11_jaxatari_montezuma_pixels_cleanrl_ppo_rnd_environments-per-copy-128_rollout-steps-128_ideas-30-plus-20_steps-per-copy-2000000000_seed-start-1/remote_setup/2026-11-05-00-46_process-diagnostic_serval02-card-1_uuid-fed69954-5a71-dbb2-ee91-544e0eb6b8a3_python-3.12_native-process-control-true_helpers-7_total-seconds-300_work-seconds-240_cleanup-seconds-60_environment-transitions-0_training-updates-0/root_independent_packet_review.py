import pathlib,json,hashlib,subprocess,os,datetime,shlex
root=pathlib.Path(__file__).resolve().parents[5]
packet=pathlib.Path(__file__).resolve().parent
run=packet.parent.parent
unit=run/'data'/packet.name
assert os.environ.get('SLURM_JOB_ID')=='6783003'
assert '/job_6783003/' in pathlib.Path('/proc/self/cgroup').read_text()
def read(p):
 p=pathlib.Path(p);assert p.resolve(strict=True)==p and p.is_file() and p.stat().st_size<3000000
 return json.loads(p.read_bytes())
def ref(p):
 p=pathlib.Path(p);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def pub(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
plan=read(unit/'measurement_plan.json')
assert plan['helpers']['source_commit']=='39c93b44af70f9dfa63f38cde93f78b3604cc365'
assert len(plan['helpers']['files'])==7
source=pathlib.Path(plan['helpers']['source_root'])
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==plan['helpers']['source_commit']
for rel,sha in plan['helpers']['files'].items():
 raw=subprocess.check_output(['git','-C',str(source),'show',plan['helpers']['source_commit']+':'+rel]);assert hashlib.sha256(raw).hexdigest()==sha
 assert (source/rel).read_bytes()==(unit/'code'/pathlib.Path(rel).name).read_bytes()==raw
assert plan['diagnostic_time_budget']=={'schema_version':1,'total_seconds':300,'work_seconds':240,'cleanup_seconds':60}
assert plan['maximum_active_seconds']==240
assert plan['python']=='/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python'
assert plan['gpu_uuid']=='GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3'
assert plan['native_process_control']['mode']=='observe_target_contract_for_zero_credit_diagnostic'
assert plan['native_process_control']['qualification'] is False and plan['native_process_control']['expected_target_contract'] is None
for k,v in plan['diagnostic_boundary'].items():assert v is False or type(v) is int and v==0
assert plan['profile_tool']==read(run/'remote_setup/serval02_nsight_tool_record_verified_bytes_2026-11-05.json')
assert read(unit/'manifest.yaml')==plan
assert read(unit/'measurement_prepared.json')=={'plan_sha256':ref(unit/'measurement_plan.json')['sha256']}
assert read(packet/'actual_preparation_return.json')['returncode']==0
cmd=read(packet/'proposed_target_launch.json')
assert cmd['argv'][:8]==['ssh','-o','BatchMode=yes','-o','ControlMaster=no','-o','ConnectTimeout=15','serval02']
expected=['env','CUDA_VISIBLE_DEVICES='+plan['gpu_uuid'],plan['python'],'-I','-S','-B',str(unit/'code/diagnose_direct_nsight_process.py'),'--controller-unit',str(unit)]
assert shlex.split(cmd['argv'][8])==expected and cmd['no_automatic_retry'] is True
assert not any((unit/n).exists() for n in ['controller_spawn.json','controller_wait.json','failure.json','diagnostic_controller_time_origin.json'])
n=read(run/'next_actions.json');c=n['h100_common_correctness_preparation']['copies_2'];refs={}
for name in ['actual_exit_observation','actual_normal_wait','actual_controller_spawn','actual_child_spawn','independent_actual_review','actual_review_verification','plan','actual_raw_validation']:
 assert ref(c[name]['path'])==c[name];refs[name]=c[name]
exited=read(c['actual_exit_observation']['path']);wait=read(c['actual_normal_wait']['path']);controller=read(c['actual_controller_spawn']['path']);spawn=read(c['actual_child_spawn']['path']);review=read(c['independent_actual_review']['path']);verified=read(c['actual_review_verification']['path']);oldplan=read(c['plan']['path'])
assert exited['all_original_processes_exited'] is True and exited['gpu_or_standing_lock_operations']==0
assert controller['actual_pid']==c['controller_pid']==3889606 and controller['actual_birth_counter']==c['controller_birth_counter']=='79607669'
assert spawn['popen_pid']==wait['pid']==wait['waited_pid']==c['actual_numerical_child_pid']==3892092
assert wait['process_start_time']==c['actual_numerical_child_birth_counter']=='79633156'
assert spawn['controller']['pid']==controller['actual_pid'] and spawn['controller']['process_start_time']==controller['actual_birth_counter']
assert wait['exited_normally'] is True and type(wait['returncode']) is int and wait['returncode']==0 and wait['timeouts']==[]
assert wait['command']==spawn['command'] and wait['waiter_before']['pid']==wait['waiter_after']['pid']==controller['actual_pid']
assert [(o['role'],o['expected_pid'],o['expected_birth'],o['observation'],o['original_process_exited']) for o in exited['observations']]==[('controller',3889606,'79607669','absent',True),('numerical_child',3892092,'79633156','absent',True)]
assert review['passed'] is True and review['approved'] is True and review['independent_of_producer'] is True and review['copies']==2 and review['gpu_uuid']==plan['gpu_uuid']
assert verified['passed'] is True and verified['actual_raw_pass_before_review'] is True and verified['review']==c['independent_actual_review'] and verified['raw']==c['actual_raw_validation']
assert oldplan['source_commit']==plan['source_commit'] and review['source_commit']==plan['source_commit']
exit_review={'kind':'independent_saved_completed_predecessor_exit_review','passed':True,'approved':True,'independent_of_producer':True,'all_exact_processes_exited':True,'gpu_uuid':plan['gpu_uuid'],'copies':2,'source_commit':plan['source_commit'],'unresolved_anomalies':[],'references':refs,'scope':'Exact already accepted historical process waits and saved exit observations only; no new live query or GPU action.','review_job_id':os.environ['SLURM_JOB_ID'],'review_step_id':os.environ['SLURM_STEP_ID'],'review_pid':os.getpid(),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
pub(packet/'independent_exit_review.json',exit_review)
launch={'kind':'reviewed_direct_nsight_descriptor_diagnostic_launch','approved':True,'passed':True,'plan':ref(unit/'measurement_plan.json'),'helper_commit':plan['helpers']['source_commit'],'gpu_uuid':plan['gpu_uuid'],'standing_descriptor_owner':'tool producer only','metadata_controller_owns_standing_descriptor':False,'actual_c2_exit_verified':True,'c2_exit_review':ref(packet/'independent_exit_review.json'),'total_budget_seconds':300,'scientific_credit':0,'worker_admission':False,'review_job_id':os.environ['SLURM_JOB_ID'],'review_step_id':os.environ['SLURM_STEP_ID'],'review_pid':os.getpid(),'actual_target_diagnostic_run':False}
pub(unit/'diagnostic_launch_review.json',launch)
pub(packet/'root_independent_packet_review.json',{'passed':True,'source_files_verified':7,'plan':ref(unit/'measurement_plan.json'),'launch_review':ref(unit/'diagnostic_launch_review.json'),'exit_review':ref(packet/'independent_exit_review.json'),'proposal':ref(packet/'proposed_target_launch.json'),'metadata_only':True,'gpu_or_tool_operations':0,'maximum_seconds':300,'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID']})
print(json.dumps({'passed':True,'step':os.environ['SLURM_STEP_ID'],'source_files':7,'target_launched':False}))
