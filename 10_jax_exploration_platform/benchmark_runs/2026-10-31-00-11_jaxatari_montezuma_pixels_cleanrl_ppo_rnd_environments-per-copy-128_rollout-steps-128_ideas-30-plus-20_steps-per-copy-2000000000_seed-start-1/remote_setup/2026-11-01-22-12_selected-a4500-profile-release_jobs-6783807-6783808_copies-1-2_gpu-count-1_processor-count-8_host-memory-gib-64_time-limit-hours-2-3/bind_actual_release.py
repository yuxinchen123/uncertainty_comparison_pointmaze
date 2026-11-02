"""Bind real held receipts and fresh saved monitoring evidence, without scheduler or GPU work."""
import ast,ctypes,datetime,hashlib,importlib.abc,importlib.util,json,os,stat,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1]
V2=R/'remote_setup/2026-11-01-22-00_selected-a4500-profile-held-submission_readiness-monitor-step-140_copies-1-2_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-2-3'
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
NUMERICAL={'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy','exploration_platform','pynvml'}
assert not NUMERICAL.intersection(sys.modules)
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in NUMERICAL:raise RuntimeError('numerical import forbidden during release binding')
sys.meta_path.insert(0,NoNumerical())
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('shared-library load forbidden during release binding'))
def audit(event,values):
 if event in {'subprocess.Popen','socket.connect','socket.bind','socket.getaddrinfo','os.system','os.kill','os.killpg'}:
  raise RuntimeError('scheduler,child,network or process action forbidden during release binding')
sys.addaudithook(audit)
def raw(path):
 p=Path(path);assert p.is_absolute() and p.resolve(strict=True)==p and p.is_file() and not p.is_symlink()
 b=p.read_bytes();assert len(b)<=64*1024*1024;return b
def read(path):return json.loads(raw(path))
def sha(path):return hashlib.sha256(raw(path)).hexdigest()
def ref(path):return {'path':str(path),'sha256':sha(path)}
def record(path):return {'sha256':sha(path),'bytes':len(raw(path))}
def put(name,value,bytes_only=False):
 p=P/name;b=value if bytes_only else (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 return ref(p)
def utc(value):
 t=datetime.datetime.fromisoformat(value.replace('Z','+00:00'));assert t.tzinfo and t.utcoffset().total_seconds()==0
 assert 0<=(datetime.datetime.now(datetime.timezone.utc)-t).total_seconds()<1200
 return t
spec=importlib.util.spec_from_file_location('metadata_only_fixed_release',P/'release_selected_profiles.py')
release=importlib.util.module_from_spec(spec);spec.loader.exec_module(release)
scope=read(P/'draft_scope.json');submit=read(V2/'submission_inputs.json')
M=Path(read(R/'next_actions.json')['persistent_monitor']['packet']);prefix='scheduled_check_2026-11-02-06-12-00'
V=M/(prefix+'_verification.json');S=M/(prefix+'_status.json');REPORT=M/(prefix+'_report.md')
verification,status=read(V),read(S);monitor=read(R/'next_actions.json')['persistent_monitor']
cron_path=R/'remote_setup/monitoring_task_2026-11-02-06-16.json';cron=read(cron_path)
assert verification['passed'] is True and type(verification['returncode']) is int and verification['returncode']==0
assert verification['monitor_job_id']=='6783003' and verification['step_id']=='145' and verification['source_commit']=='79a01e851ff92ddba7d0e12eaf8164cab2028878'
assert verification['queue_counts']=={'pending':0,'running':0,'completed':120,'failed':0}
assert verification['report_sha256']==sha(REPORT) and verification['status_sha256']==sha(S)
assert all(label in raw(REPORT).decode().splitlines() for label in ('TRAINING','WAITING','HELD','FINISHED','PLANNED'))
assert [row for row in status['jobs'] if row[0]=='6783003']==[['6783003','RUNNING','heartpiece','gres/gpu:0','heartpiece']]
assert cron['id']=='576bcd02-b929-4599-8d78-ddfe41dbe7ea' and cron['cron']=='*/20 * * * *' and cron['cadence_seconds']==1200 and cron['list_verified'] is True and cron['persistent_job_id']=='6783003'
mtime=utc(verification['checked_at_utc'])
validated={}
for copies in (1,2):
 item=scope['jobs'][str(copies)];job=item['job_id'];packet,unit=Path(item['packet']),Path(item['unit'])
 query_path=R/'slurm/jobs'/job/'root_post_full145_held_query.json';query=read(query_path)
 assert query['argv']==['scontrol','show','job','-o',job] and query['actual_job_id']==job and query['no_job_action'] is True
 assert type(query['returncode']) is int and query['returncode']==0 and query['stderr']==''
 # The fresh capture has no ledger-path field; require exact live own-ledger membership below.
 started,finished=utc(query['started_at_utc']),utc(query['finished_at_utc']);assert mtime<=started<=finished
 assert raw(R/'slurm/submitted_jobids_uva.txt').decode().splitlines().count(job)==1
 release.PACKET,release.UNIT,release.JOB,release.REQUESTED_HOURS=packet,unit,job,{1:2,2:3}[copies]
 release.validate_held(query['stdout'])
 row=[job,'PENDING','','gres/gpu:nvidia_rtx_a4500:1','(JobHeldUser)']
 assert [current for current in status['jobs'] if current[0]==job]==[row]
 assert ref(packet/'submission_receipt.json')==item['actual_submission'] and ref(packet/'held_binding.json')==item['actual_held_binding'] and ref(unit/'configured_isolation_pre_release_inputs.json')==item['actual_pre_release_inputs']
 assert not any(os.path.lexists(packet/name) for name in release.ATTEMPT_FILES)
 assert not os.path.lexists(unit/'owned_allocation_inputs.json') and not os.path.lexists(unit/'.owned_allocation_inputs.json.pending')
 validated[str(copies)]={'query':query,'query_path':query_path,'started':started,'finished':finished,'row':row}
proofs={}
for copies in (1,2):
 item=scope['jobs'][str(copies)];data=validated[str(copies)];query=data['query']
 packet,unit,job=Path(item['packet']),Path(item['unit']),item['job_id']
 stdout=put('held_query_copies_'+str(copies)+'_stdout.raw',query['stdout'].encode('utf-8'),True)
 stderr=put('held_query_copies_'+str(copies)+'_stderr.raw',query['stderr'].encode('utf-8'),True)
 qrecord={'command':query['argv'],'at_utc':query['finished_at_utc'],'returncode':query['returncode'],'stdout':stdout,'stderr':stderr,
  'kind':'actual_exact_owned_held_release_query','job_id':job,'started_at_utc':query['started_at_utc'],'job_action_performed':False}
 qref=put('held_query_copies_'+str(copies)+'_return.json',qrecord)
 release.PACKET,release.UNIT,release.JOB,release.REQUESTED_HOURS=packet,unit,job,{1:2,2:3}[copies]
 fields={name:release.field(query['stdout'],name) for name in ('JobId','UserId','Account','Partition','QOS','JobState','Reason','ReqNodeList','Reservation','Requeue','Restarts','NumCPUs','NumTasks','CPUs/Task','MinMemoryNode','TimeLimit','NumNodes','Command','TresPerNode','ReqTRES','AllocTRES','NodeList')}
 resource=put('actual_held_resource_review_copies_'+str(copies)+'.json',{'schema_version':1,'kind':'exact_owned_held_profile_resource_review','passed':True,'job_id':job,'fields':fields,'actual_query_source':ref(data['query_path']),'actual_raw_query_return':qref,'no_job_action':True,'unallocated_physical_identity_unresolved':True,'scientific_credit':0,'admission_granted':False})
 expiry=min(mtime,data['started'],data['finished'])+datetime.timedelta(seconds=1200)
 proof={'schema_version':1,'kind':'owned_uva_configured_isolation_actual_full_held_monitor_proof','passed':True,'job_id':job,'copies':copies,
  'state':'PENDING','reason':'JobHeldUser','publication_commit':'2d356602a400168a275c6dac604d19a08f634d93',
  'helper_source_commit':'fdb35f886ad01c6d4ed387dc55b172d0221a9853','at_utc':query['finished_at_utc'],
  'derived_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'expires_at_utc':expiry.isoformat(),
  'submission':ref(packet/'submission_receipt.json'),'plan':ref(unit/'measurement_plan.json'),'job_script':ref(packet/'job.sbatch'),
  'pre_release_inputs':ref(unit/'configured_isolation_pre_release_inputs.json'),'held_binding':ref(packet/'held_binding.json'),
  'owned_query_return':qref,'owned_query_stdout':stdout,'owned_query_stderr':stderr,'actual_query_source':ref(data['query_path']),
  'root_resource_review':resource,'monitor':{'job_id':'6783003','source_commit':verification['source_commit'],'step_id':verification['step_id'],
   'cadence_seconds':1200,'persistent_running_verified':True,'runtime_and_first_full_report_review':ref(Path(monitor['runtime_and_first_full_report_review'])),
   'full_verification':ref(V),'full_status':ref(S),'full_report':ref(REPORT),'row':data['row']},
  'recurring_session_task':{'id':cron['id'],'cron':cron['cron'],'cadence_seconds':1200,
   'exact_prompt_and_schedule_verified_before_old_deleted':True,'actual_record':ref(cron_path)},
  'frozen_physical_identity_remains_unresolved':True,'released':False,'scientific_credit':0,'case_credit':0,'idea_credit':0,
  'admission_granted':False,'new_scheduler_calls':0,'proof_is_metadata_not_a_release':True,
  'monitor_unit_display_is_not_scientific_acceptance':True}
 proofs[str(copies)]=put('held_monitor_proof_copies_'+str(copies)+'.json',proof)
progress={str(Path(item['unit'])/'progress.json') for item in scope['jobs'].values()}
assert all(ref(value['path'])==value for value in submit['required_files'])
shared={value['path']:record(value['path']) for value in submit['required_files'] if value['path'] not in progress}
assert len(submit['required_files'])-len(shared)==2
jobs={}
for copies in (1,2):
 item=scope['jobs'][str(copies)];packet,unit=Path(item['packet']),Path(item['unit']);proof=read(proofs[str(copies)]['path']);own={}
 def collect(value):
  if isinstance(value,dict):
   if set(value)=={'path','sha256'}:
    assert ref(value['path'])==value
    if value['path'] not in shared:own[value['path']]=record(value['path'])
   else:
    for child in value.values():collect(child)
  elif isinstance(value,list):
   for child in value:collect(child)
 collect(proof);own[proofs[str(copies)]['path']]=record(proofs[str(copies)]['path'])
 for name in ('submission_intent.json','submission_stdout.raw','submission_stderr.raw','submission_return.json','submission_receipt.json','held_binding.json'):
  own[str(packet/name)]=record(packet/name)
 for name in ('progress.json','configured_isolation_owned_job_ids.txt','configured_isolation_owned_submissions.tsv','configured_isolation_pre_release_inputs.json'):
  own[str(unit/name)]=record(unit/name)
 own={path:value for path,value in own.items() if path not in shared}
 jobs[str(copies)]={**item,'files':own,'plan':ref(unit/'measurement_plan.json'),'script':ref(packet/'job.sbatch'),
  'pre_release_inputs':ref(unit/'configured_isolation_pre_release_inputs.json'),'submission':ref(packet/'submission_receipt.json'),
  'held_binding':ref(packet/'held_binding.json'),'held_proof':proofs[str(copies)],'recurring_session_task_id':cron['id']}
frozen=put('release_frozen_inputs.json',{'schema_version':1,'kind':'fixed_selected_profile_actual_release_inputs',
 'source_commit':'fdb35f886ad01c6d4ed387dc55b172d0221a9853','initial_publication':'2d356602a400168a275c6dac604d19a08f634d93',
 'shared_files':shared,'jobs':jobs,'selected_target_only_progress_and_ownership':True,'maximum_startup_wait_seconds':180,
 'actual_releases':0,'admission_granted':False})
path=P/'release_selected_profiles.py';before=raw(path);assert before.count(b'FROZEN_INPUTS_SHA = None')==1
put('source_before_actual_readiness_binding.py',before,True)
after=before.replace(b'FROZEN_INPUTS_SHA = None',('FROZEN_INPUTS_SHA = '+repr(frozen['sha256'])).encode())
with path.open('wb') as f:f.write(after);f.flush();os.fsync(f.fileno())
put('release_author_review.json',{'schema_version':1,'kind':'fixed_selected_profile_actual_release_implementation_review',
 'source_sha256':sha(path),'jobs':{key:item['job_id'] for key,item in jobs.items()},'release_executed':False,
 'maximum_startup_wait_seconds':180,'real_monitor_proofs':proofs,'shared_static_file_count':len(shared),
 'selected_target_only_mutable_state':True,'frozen_inputs':frozen,'NUM_work':False,'admission_granted':False})
names=('require','sha','encoded','contents','document','ref','checked_ref','put','field','validate_held','core','validate_pre_release','utc_observation','validate_proof','validate_frozen_files','require_unattempted_release','capture','release_once','main')
def functions(source):return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(source).body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
original=functions(raw(V2/'release_selected_profiles.py'));bound=functions(raw(path))
assert all(original[name]==bound[name] for name in names)
put('actual_binding_verification.json',{'schema_version':1,'kind':'actual_selected_profile_release_binding','passed':True,
 'source':ref(path),'frozen_inputs':frozen,'actual_jobs':{key:item['job_id'] for key,item in jobs.items()},
 'actual_monitor_step':'145','actual_native_task':cron['id'],'proofs':proofs,
 'action_and_reader_function_ASTs_identical':{name:True for name in names},'actual_scheduler_queries':0,'actual_releases':0,
 'minimum_expiry_utc':min(read(value['path'])['expires_at_utc'] for value in proofs.values())})
print(json.dumps({'passed':True,'source':ref(path),'inputs':frozen,'proofs':proofs,
 'expiry_utc':min(read(value['path'])['expires_at_utc'] for value in proofs.values()),'binding_cpu_step':os.environ['SLURM_STEP_ID']}))
