"""Discover Nsight metadata on the existing zero-GPU CPU allocation without a capture."""
import ctypes,datetime,hashlib,importlib.abc,importlib.util,json,os,re,shutil,socket,stat,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent
context=json.loads((P/'config_resolved.yaml').read_bytes());R=Path(context['campaign_run_dir'])
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
assert os.getuid()==os.geteuid()==2618919
assert (R/'slurm/submitted_jobids_uva.txt').read_text().splitlines().count('6783003')==1
blocked=[]
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform','pynvml'}:
   blocked.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,NoNumerical())
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('Python shared-library load forbidden'))
sys.addaudithook(lambda event,args:(_ for _ in ()).throw(RuntimeError('network/device operation forbidden')) if event in {'socket.connect','socket.bind','socket.getaddrinfo','os.system'} else None)
def put(name,value,raw=False):
 p=P/name;b=value if raw else (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 return {'path':str(p),'sha256':hashlib.sha256(b).hexdigest()}
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
proc=Path('/proc/self/stat').read_text();status=Path('/proc/self/status').read_text()
allocation={'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'hostname':socket.gethostname(),
 'pid':os.getpid(),'parent_pid':os.getppid(),'process_start_time':proc.rsplit(')',1)[1].split()[19],
 'uids':next(line for line in status.splitlines() if line.startswith('Uid:')),
 'cpu_affinity':sorted(os.sched_getaffinity(0)),'cgroup':Path('/proc/self/cgroup').read_text(),
 'requested_step':{'cpus':1,'host_memory_mib':256,'gpu_count':0,'maximum_minutes':5},
 'slurm_environment':{key:os.environ.get(key) for key in ('SLURM_JOB_ID','SLURM_STEP_ID','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE','SLURM_JOB_GPUS','SLURM_STEP_GPUS')}}
put('actual_allocation.json',allocation)
located=shutil.which('nsys');candidates=[];module_records=[]
if located:candidates=[str(Path(located).resolve(strict=True))]
else:
 fixed=['/usr/bin/nsys','/usr/local/bin/nsys','/usr/local/cuda/bin/nsys','/opt/nvidia/nsight-systems/bin/nsys']
 patterns=['/opt/nvidia/nsight-systems/*/bin/nsys','/opt/nvidia/nsight-systems/*/host-linux-x64/nsys',
  '/sw/ubuntu/custom/nsight-systems/*/bin/nsys','/sw/ubuntu2204/custom/nsight-systems/*/bin/nsys']
 import glob
 for pattern in patterns:
  fixed.extend(glob.glob(pattern)[:16])
 for filename in fixed:
  p=Path(filename)
  if p.is_file() and os.access(p,os.X_OK):candidates.append(str(p.resolve(strict=True)))
 if not candidates:
  roots=[Path(value) for value in os.environ.get('MODULEPATH','').split(':') if value.startswith(('/sw/','/etc/','/usr/share/','/opt/'))]
  for root in roots[:8]:
   if not root.is_dir():continue
   for entry in sorted(root.iterdir())[:256]:
    if not any(name in entry.name.lower() for name in ('nsight','nsys','cuda','nvidia')):continue
    files=[entry] if entry.is_file() else sorted(entry.iterdir())[:32] if entry.is_dir() else []
    for file in files:
     if not file.is_file() or file.stat().st_size>65536:continue
     raw=file.read_bytes();text=raw.decode('utf-8',errors='strict')
     paths=re.findall(r'(/[A-Za-z0-9_.+/-]*(?:nsight|nsys)[A-Za-z0-9_.+/-]*)',text,re.I)
     module_records.append({'path':str(file),'sha256':hashlib.sha256(raw).hexdigest(),'literal_candidate_roots':paths})
     for value in paths[:16]:
      for tail in ('','bin/nsys','host-linux-x64/nsys'):
       path=Path(value)/tail
       if path.is_file() and os.access(path,os.X_OK):candidates.append(str(path.resolve(strict=True)))
candidates=sorted(set(candidates))
put('discovery.json',{'which_nsys':located,'actual_candidate_paths':candidates,'module_metadata':module_records,
 'selection_policy':'Use the actual PATH result, otherwise only a unique bounded discovered executable; no version preference is guessed.'})
record=None;failure=None
if len(candidates)==1 and Path(candidates[0]).stat().st_size<=32*1024*1024:
 candidate=Path(candidates[0]);published=Path(context['published_helper_root'])/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
 source=published/'profile_selected.py';assert digest(source)==context['published_profile_source_sha256']
 sys.path.insert(0,str(published));spec=importlib.util.spec_from_file_location('published_profile_tool_reader',source)
 profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)
 real_popen=subprocess.Popen
 def captured(command,*a,**kwargs):
  assert list(command)==[str(candidate),'--version']
  started=datetime.datetime.now(datetime.timezone.utc).isoformat()
  child=real_popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  own={'pid':child.pid,'parent_pid':os.getpid(),'command':list(command),'started_at_utc':started}
  try:
   child_stat=Path('/proc/'+str(child.pid)+'/stat').read_text();own['process_start_time']=child_stat.rsplit(')',1)[1].split()[19]
  except FileNotFoundError:own['process_start_time']=None;own['process_observation_limit']='child exited before its birth counter was read'
  put('version_process.json',own)
  try:
   stdout,stderr=child.communicate(timeout=15);returned={'returncode':child.returncode,'completed':True}
  except subprocess.TimeoutExpired as error:
   child.kill();stdout,stderr=child.communicate();returned={'returncode':child.returncode,'completed':False,'error_type':type(error).__name__}
  out=put('version.stdout.raw',stdout,True);err=put('version.stderr.raw',stderr,True)
  put('version_return.json',{'command':list(command),'at_utc':started,**returned,'stdout':out,'stderr':err})
  if not returned['completed'] or returned['returncode']!=0:raise RuntimeError('the one version command failed; preserve its actual return')
  return stdout.decode('utf-8',errors='strict') if kwargs.get('text') else stdout
 profile.subprocess.check_output=captured
 try:record=profile.tool_record(str(candidate))
 except BaseException as error:failure={'error_type':type(error).__name__,'error':str(error)}
elif not candidates:failure={'reason':'nsys was not found in PATH or the bounded known/module paths'}
elif len(candidates)>1:failure={'reason':'multiple actual executable paths remain; no selection was guessed'}
else:failure={'reason':'tool image exceeds the light-phase 32MiB hash bound; no heavy read or version call was made'}
assert not blocked
put('result.json',{'schema_version':1,'kind':'actual_nsight_tool_metadata_discovery','metadata_completed':True,'tool_record':record,
 'discovery_failure_or_limit':failure,'allocation':allocation,'version_command_only':True,'capture_or_profile_requested':False,
 'Python_numerical_import_attempts':blocked,'Python_shared_library_calls_blocked':True,'external_tool_library_mapping_audit':'not performed; the only external argv, when found, was --version',
 'scientific_credit':0,'case_credit':0,'idea_credit':0,'admission_granted':False})
print(json.dumps({'metadata_completed':True,'found':record is not None,'tool_record':record,'failure_or_limit':failure,'actual_step':os.environ['SLURM_STEP_ID'],'packet':str(P)}))

