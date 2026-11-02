"""Read one already discovered Nsight executable's version/hash under the shared artifact lock."""
import ctypes,fcntl,hashlib,importlib.abc,importlib.util,json,os,socket,stat,subprocess,sys,datetime
from pathlib import Path
P=Path(__file__).resolve().parent
c=json.loads((P/'config_resolved.yaml').read_bytes());R=Path(c['campaign_run_dir']);target=Path(c['discovered_executable'])
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
assert os.getuid()==os.geteuid()==2618919 and (R/'slurm/submitted_jobids_uva.txt').read_text().splitlines().count('6783003')==1
blocked=[]
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform','pynvml'}:
   blocked.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,NoNumerical())
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Python shared-library call forbidden'))
sys.addaudithook(lambda event,args:(_ for _ in ()).throw(RuntimeError('network/device action forbidden')) if event in {'socket.connect','socket.bind','socket.getaddrinfo','os.system'} else None)
def put(name,value,raw=False):
 p=P/name;b=value if raw else (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest()}
original_discovery=Path(c['discovery']['path']);assert hashlib.sha256(original_discovery.read_bytes()).hexdigest()==c['discovery']['sha256']
assert json.loads(original_discovery.read_bytes())['actual_candidate_paths']==[str(target)]
assert target.resolve(strict=True)==target and target.is_file() and os.access(target,os.X_OK) and target.stat().st_size==58059912
allocation={'job_id':'6783003','step_id':os.environ['SLURM_STEP_ID'],'hostname':socket.gethostname(),'pid':os.getpid(),'parent_pid':os.getppid(),
 'process_start_time':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'cgroup':Path('/proc/self/cgroup').read_text(),
 'cpu_affinity':sorted(os.sched_getaffinity(0)),'uids':next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('Uid:')),
 'requested_cpus':1,'requested_host_memory_mib':256,'requested_gpus':0,'requested_step_time_limit_minutes':c['maximum_minutes'],'slurm_environment':{k:os.environ.get(k) for k in ('SLURM_JOB_ID','SLURM_STEP_ID','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE','SLURM_JOB_GPUS','SLURM_STEP_GPUS')}}
put('actual_allocation.json',allocation)
scripts=Path(c['published_helper_root'])/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
source=scripts/'profile_selected.py';assert hashlib.sha256(source.read_bytes()).hexdigest()==c['published_profile_source_sha256']
sys.path.insert(0,str(scripts));spec=importlib.util.spec_from_file_location('published_nsight_metadata_reader',source);profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)
real_popen=subprocess.Popen
def forbidden(*a,**k):raise RuntimeError('unrelated subprocess forbidden')
subprocess.Popen=forbidden
def captured(command,*args,**kwargs):
 assert list(command)==[str(target),'--version']
 started=datetime.datetime.now(datetime.timezone.utc).isoformat();child=real_popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 process={'pid':child.pid,'parent_pid':os.getpid(),'command':list(command),'started_at_utc':started}
 try:process['process_start_time']=Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:process['process_start_time']=None;process['limit']='child exited before a birth counter was observed'
 put('version_process.json',process)
 try:out,err=child.communicate(timeout=15);returned={'returncode':child.returncode,'completed':True}
 except subprocess.TimeoutExpired as error:
  child.kill();out,err=child.communicate();returned={'returncode':child.returncode,'completed':False,'error_type':type(error).__name__}
 stdout=put('version.stdout.raw',out,True);stderr=put('version.stderr.raw',err,True)
 put('version_return.json',{'command':list(command),'at_utc':started,**returned,'stdout':stdout,'stderr':stderr})
 if not returned['completed'] or returned['returncode']!=0:raise RuntimeError('the actual version command failed; retain its raw return')
 return out.decode('utf-8',errors='strict') if kwargs.get('text') else out
profile.subprocess.check_output=captured
with (R/'queue/cpu_monitor_large_artifact_read.lock').open('r+') as lock:
 info=os.fstat(lock.fileno());assert stat.S_ISREG(info.st_mode) and info.st_uid==os.getuid()
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
 put('shared_read_lock.json',{'path':str(R/'queue/cpu_monitor_large_artifact_read.lock'),'inode':info.st_ino,'device':info.st_dev,'owner_uid':info.st_uid,'exclusive':True,'purpose':'serialize the58059912-byte tool hash with other artifact readers'})
 record=profile.tool_record(str(target))
tool=put('tool_record.json',record)
assert set(record)=={'path','version','sha256'} and record['path']==str(target) and not blocked
put('result.json',{'schema_version':1,'kind':'actual_nsight_version_and_executable_hash_metadata','passed':True,'tool_record':record,'tool_record_receipt':tool,
 'actual_allocation':allocation,'published_helper_commit':c['published_helper_commit'],'only_external_command':[str(target),'--version'],
 'capture_or_profile_requested':False,'Python_numerical_import_attempts':blocked,'Python_shared_library_calls_blocked':True,
 'external_tool_library_mapping_audit':'not performed; only the authorized --version argv was executed',
 'case_credit':0,'scientific_credit':0,'admission_granted':False})
print(json.dumps({'passed':True,'tool_record':record,'record':tool,'actual_job':'6783003','actual_step':os.environ['SLURM_STEP_ID'],'packet':str(P)}))

