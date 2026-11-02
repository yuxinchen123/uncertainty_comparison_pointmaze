"""Independently review recovery source and metadata; never run a monitor tick."""
import ast,contextlib,ctypes,hashlib,importlib,importlib.util,io,json,os,socket,stat,subprocess,sys,tarfile
from datetime import datetime,timezone
from pathlib import Path
from zoneinfo import ZoneInfo
W=Path(__file__).resolve().parents[1]
R=Path(json.loads((W/'.review_metadata/monitor_7fd_preparation_inputs.json').read_text())['packet']).parent.parent
P=R/'remote_setup/2026-11-01-16-33_pixel-study-monitor-recovery_source-commit-79a01e851ff92ddba7d0e12eaf8164cab2028878_processor-count-1_host-memory-gib-2_time-limit-days-7_poll-interval-seconds-1200_partition-nolim_account-sl5nw-base_gpu-count-0'
OUT=R/'remote_setup'/(datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d-%H-%M-%S')+'_cpu-monitor-recovery-static-independent-review_gpu-count-0')
OUT.mkdir();assert socket.gethostname().split('.')[0]=='portal05' and not os.environ.get('SLURM_JOB_ID')
BLOCKED=[];FORBIDDEN={'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy','jaxtari','pynvml'}
def guard(event,args):
 denied=event=='import' and args[0].split('.')[0] in FORBIDDEN
 denied=denied or event in {'socket.connect','socket.bind','os.system'} or event=='ctypes.dlopen' and args[0] is not None
 denied=denied or event=='open' and str(args[0]).startswith(('/dev/nvidia','/dev/dri'))
 if event=='subprocess.Popen':
  argv=args[1];denied=not(isinstance(argv,(list,tuple)) and (Path(argv[0]).name=='git' or argv[:2]==['bash','-n']))
 if denied:BLOCKED.append(event);raise RuntimeError('static recovery review forbids '+event)
sys.addaudithook(guard)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def ref(path):return {'path':str(path),'sha256':sha(path)}
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def save(name,value):
 raw=(json.dumps(value,indent=2)+'\n').encode();f=(OUT/name).open('xb');f.write(raw);f.flush();os.fsync(f.fileno());f.close();return ref(OUT/name)
def main():
 assert sha(P/'static_source_freeze.json')=='c7aae037f1b73d424d8fd7f092480edc449ba12a3a581bd50ac2fb3123438ecd'
 assert sha(P/'static_publication_roster.json')=='7d90454edca9f55f178a46e2781cbe17f84e99d0d9544a450eb55a8bba72e15c'
 frozen=read(P/'static_source_freeze.json');roster=read(P/'static_publication_roster.json');assert len(frozen['files'])==len(roster['files'])==297
 before={name:sha(name) for name in frozen['files']};assert before==frozen['files']
 for item in roster['files']:
  path=Path(item['path']);assert path.resolve(strict=True)==path and stat.S_ISREG(path.lstat().st_mode)
  assert sha(path)==item['sha256'] and path.stat().st_size==item['bytes'] and format(stat.S_IMODE(path.stat().st_mode),'04o')==item['mode']
 manifest=read(P/'source_snapshot.json');commit='79a01e851ff92ddba7d0e12eaf8164cab2028878';source=Path(manifest['source_root']);assert manifest['source_commit']==commit and len(manifest['files'])==265
 repo=R.parents[2];tree={}
 raw=subprocess.check_output(['git','--no-optional-locks','-C',str(repo),'ls-tree','-r',commit,'--','10_jax_exploration_platform/src','10_jax_exploration_platform/scripts'],text=True)
 for line in raw.splitlines():
  metadata,name=line.split('\t');mode,kind,blob=metadata.split();assert kind=='blob';tree[name]=(mode,blob)
 assert set(tree)==set(manifest['files'])
 with tarfile.open(P/'source_git_archive.tar') as archive:
  members=[item for item in archive.getmembers() if not item.isdir()];assert len(members)==265
  for item in members:
   assert item.isfile() and item.name.startswith(commit+'/');name=item.name.removeprefix(commit+'/');raw=archive.extractfile(item).read();mode,blob=tree[name];entry=manifest['files'][name]
   assert hashlib.sha256(raw).hexdigest()==entry['sha256']==sha(source/name)
   assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==blob==entry['git_blob'] and mode==entry['git_mode']
   assert item.mode&0o111==(0o111 if mode=='100755' else 0) and (source/name).stat().st_mode&0o111==(0o111 if mode=='100755' else 0)
 output=io.StringIO()
 with contextlib.redirect_stdout(output):load('recovery_static_source_verifier',P/'verify_source.py').main()
 source_visibility=json.loads(output.getvalue());assert source_visibility['files_verified']==265 and source_visibility['full_tick_executed'] is False
 old=Path(json.loads((W/'.review_metadata/monitor_7fd_preparation_inputs.json').read_text())['packet'])
 functions=lambda path:{node.name:ast.dump(node,include_attributes=False) for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef)}
 assert functions(P/'allocation_guard.py')==functions(old/'allocation_guard.py')
 guard=load('recovery_static_allocation_guard',P/'allocation_guard.py')
 fields={'CPUs/Task':'1','NumNodes':'1','MinMemoryNode':'2G','ReqTRES':'cpu=1,mem=2G,node=1','NumCPUs':'1','AllocTRES':'cpu=1,mem=2G,node=1','NodeList':'synthetic-node'};node={'NodeName':'synthetic-node','ThreadsPerCore':'2'}
 guard.validate_resources(fields,node);two={**fields,'NumCPUs':'2','AllocTRES':'cpu=2,mem=2G,node=1'};guard.validate_resources(two,node)
 for bad,topology in (({**fields,'ReqTRES':'cpu=2,mem=2G,node=1'},node),({**fields,'AllocTRES':'cpu=1,mem=2G,node=1,gres/gpu=1'},node),(two,{**node,'ThreadsPerCore':'1'}),(fields,{**node,'NodeName':'foreign-node'})):
  try:guard.validate_resources(bad,topology)
  except ValueError:pass
  else:raise AssertionError('invalid runtime shape accepted')
 plan=read(P/'monitor_replacement_plan.json');assert plan['helper_commit']==commit and plan['job_id'] is None and plan['gpu_uuid'] is None and plan['reservation'] is None
 assert plan['job_shape']=={'account':'sl5nw_base','partition':'nolim','qos':'normal','nodes':1,'ntasks':1,'cpus_per_task':1,'host_memory_gib':2,'gpus_per_node':0,'time':'7-00:00:00','interval_seconds':1200,'requeue':False}
 assert '--hold' in plan['submission_command'] and '--no-requeue' in plan['submission_command'] and '--gpus-per-node=0' in plan['submission_command']
 subprocess.run(['bash','-n',str(P/'monitor.sbatch')],check=True)
 basis=read(P/'recovery_basis.json');assert basis['old_failed_job_id']=='6782816' and basis['prelaunch_full_candidate_available'] is basis['prelaunch_full_candidate_executed'] is basis['prelaunch_actual_full_report_available'] is basis['old_monitor_running_required'] is False
 assert basis['actual_first_full_report_required_after_start'] is basis['actual_source_and_runtime_review_required_after_start'] is basis['future_work_blocked_until_first_full_report_independent_review'] is True
 for record in (*basis['source_validation'].values(),basis['native_cron'],basis['old_terminal_evidence']):assert ref(record['path'])==record
 for record in basis['source_validation'].values():
  result=read(record['path']);assert result['passed'] is True and result['tests']==23 and result['job_id']=='6782816'
 failed=read(basis['old_terminal_evidence']['path']);assert failed['old_allocation_usable'] is False and failed['actual_current_old_allocation_validation'] is False
 for name in ('terminal_query','terminal_stdout','terminal_stderr','historical_runtime'):assert ref(failed[name]['path'])==failed[name]
 assert '6782816|FAILED|' in Path(failed['terminal_stdout']['path']).read_text()
 cron=read(basis['native_cron']['path']);assert cron['actual_task_tool_queried_by_this_preparation'] is False and cron['root_reported_current_task']['id']=='ca8af776-26dd-4ee8-8c00-45ddc8bbc0cd' and cron['root_reported_current_task']['cron']=='*/20 * * * *'
 assert read(P/'owned_compute_source_visibility.json')['compute_visibility_verified'] is False
 scripts=source/'10_jax_exploration_platform/scripts/atari_pixel_campaign';sys.path.insert(0,str(scripts));imports={}
 for name in ('monitor','controller','configured_measurement','uva_slurm_configured_isolation','measure_selected','profile_selected','post_comparison_validation','portable_evidence','device_memory_limit','h100_nvl_slurm_site_validation','uva_active_configuration_observation','passive_installed_slurm_observation','direct_measurement','serval02_card1_binding','validate_training_deployment','measure_gpu_scaling','short_learning'):
  module=importlib.import_module(name);assert Path(module.__file__).resolve()==scripts/(name+'.py');imports[name]=str(module.__file__)
 rows=read(R/'processes.json');assert len(sys.modules['monitor'].process_hosts(rows))==len(rows)
 for name in ('measure_selected','profile_selected'):
  module=sys.modules[name];assert module.required_helpers()==module.HELPERS
  for a,b in ((True,False),(False,True)):
   helpers=module.required_helpers(a,b)
   for filename in helpers:assert sha(scripts/filename)==manifest['files'][(scripts/filename).relative_to(source).as_posix()]['sha256']
 assert before=={name:sha(name) for name in frozen['files']} and not BLOCKED
 assert not any(name.split('.')[0] in FORBIDDEN for name in sys.modules)
 review={'schema_version':1,'kind':'independent_static_cpu_monitor_recovery_review','approved':True,'metadata_host':'portal05','metadata_allocation':None,'source_commit':commit,'source_files':265,'frozen297_paths_before':before,'frozen297_paths_after':before,'all_git_archive_blob_hash_mode_checks_passed':True,'unchanged_guard_function_asts':True,'resource_tests':{'valid':2,'refused':4},'actual_process_ledger_hosts_parsed_without_query':len(rows),'cold_metadata_imports':imports,'native_cron_scope':'root-verified frozen basis; no task tool query by reviewer','recovery_basis':ref(P/'recovery_basis.json'),'static_freeze':ref(P/'static_source_freeze.json'),'prelaunch_full_candidate_available':False,'full_monitor_tick_executed':False,'actual_first_runtime_source_and_full_report_review_required_after_start':True,'future_work_blocked_until_full_report_review':True,'actual_submissions_or_releases':0,'numerical_imports':False,'device_library_network_actions':0,'scientific_credit':0,'worker_admitted':False}
 saved=save('review.json',review);print(json.dumps({'approved':True,'receipt':saved,'source_files':265,'frozen_files':297,'full_tick_executed':False}),flush=True)
try:main()
except BaseException as error:
 import traceback;save('failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'blocked':BLOCKED,'actual_job':None,'full_tick_executed':False});raise
