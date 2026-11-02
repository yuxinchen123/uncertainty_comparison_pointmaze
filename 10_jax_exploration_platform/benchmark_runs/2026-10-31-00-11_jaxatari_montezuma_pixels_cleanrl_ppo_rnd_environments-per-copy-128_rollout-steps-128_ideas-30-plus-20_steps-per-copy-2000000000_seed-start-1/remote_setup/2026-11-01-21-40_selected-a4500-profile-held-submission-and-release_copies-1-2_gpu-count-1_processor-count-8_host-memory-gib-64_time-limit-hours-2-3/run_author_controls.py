"""Run metadata-only utility controls on the exact owned CPU; forbid live scheduler work."""
import ctypes,datetime,fcntl,hashlib,importlib.abc,io,json,os,stat,subprocess,sys,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parent;R=P.parents[1]
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
names=('submit_selected_profiles_held.py','release_selected_profiles.py','test_submit_selected_profiles_held.py','test_release_selected_profiles.py','test_profile_submission_scope.py','submission_inputs.json')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
images=lambda:{n:sha(P/n) for n in names}
before=images();inputs=json.loads((P/'submission_inputs.json').read_bytes())
protected={row['path']:row['sha256'] for row in inputs['required_files']}
blocked=[]
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy','exploration_platform'}:
   blocked.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,NoNumerical())
real_popen=subprocess.Popen
class GitOnly(real_popen):
 def __init__(self,args,*a,**k):
  cmd=list(args) if isinstance(args,(list,tuple)) else []
  if not (len(cmd)>3 and cmd[:3]==['git','-C',str(R.parents[2])] and cmd[3] in {'show','cat-file','rev-parse','status','diff','ls-tree','merge-base','symbolic-ref'}):
   blocked.append(cmd);raise RuntimeError('real scheduler,network or child process forbidden')
  super().__init__(args,*a,**k)
subprocess.Popen=GitOnly
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('shared-library load forbidden'))
def audit(event,args):
 if event in {'socket.connect','socket.bind','socket.getaddrinfo','os.system','os.kill','os.killpg'}:
  blocked.append(event);raise RuntimeError('network or process action forbidden')
sys.addaudithook(audit)
sys.path.insert(0,str(P))
import test_submit_selected_profiles_held as held
import test_release_selected_profiles as released
import test_profile_submission_scope as scope
import test_profile_publication as publication
with (R/'queue/cpu_monitor_large_artifact_read.lock').open('r+') as lock:
 st=os.fstat(lock.fileno());assert stat.S_ISREG(st.st_mode) and st.st_uid==os.getuid();fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
 assert all(sha(Path(p))==h for p,h in protected.items())
 suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module) for module in (held,released,scope,publication))
 output=io.StringIO();started=datetime.datetime.now(datetime.timezone.utc).isoformat()
 result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
 defaults=[]
 for copies in (1,2):
  raw,item,plan=held.REAL_VALIDATE_INPUTS(copies)
  held.submit.validate_readiness(raw)
  defaults.append({'copies':copies,'kind':plan['kind'],'plan':item['plan'],'scheduler_queries':0,'submissions':0})
 # The real release default cannot reach any query before actual held proofs are bound.
 with patch.object(released.release.socket,'gethostname',return_value='portal05'),patch.object(released.release.os,'getuid',return_value=2618919):
  try:released.release.checked(1)
  except ValueError as error:release_unbound=str(error);assert 'not frozen' in release_unbound
  else:raise AssertionError('unbound release was accepted')
 assert images()==before and not blocked and all(sha(Path(p))==h for p,h in protected.items())
prefix='author_controls_6783003_'+os.environ['SLURM_STEP_ID']
def put(name,raw):
 p=P/name
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return {'path':str(p),'sha256':sha(p)}
raw_ref=put(prefix+'.raw',output.getvalue().encode())
value={'schema_version':1,'kind':'selected_profile_held_and_unbound_release_metadata_author_controls','passed':result.wasSuccessful(),
 'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'started_at_utc':started,'job_id':'6783003','step_id':os.environ['SLURM_STEP_ID'],
 'pid':os.getpid(),'birth':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'cgroup':Path('/proc/self/cgroup').read_text(),
 'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'error_details':[{'test':test.id(),'traceback':message} for test,message in result.errors],
 'failure_details':[{'test':test.id(),'traceback':message} for test,message in result.failures],'raw':raw_ref,'actual_local_defaults':defaults,
 'actual_unbound_release_refusal':release_unbound,'source_before':before,'source_after':images(),'protected_files':len(protected),'protected_bytes_unchanged':True,
 'actual_scheduler_queries':0,'actual_submissions':0,'actual_releases':0,'numerical_import_attempts':blocked,'GPU_operations':0,'scientific_credit':0,'admission_granted':False}
reference=put(prefix+'.json',(json.dumps(value,indent=2,sort_keys=True)+'\n').encode())
print(json.dumps({'passed':value['passed'],'tests':result.testsRun,'receipt':reference,'actual_step':os.environ['SLURM_STEP_ID']}))
raise SystemExit(0 if result.wasSuccessful() else 1)
