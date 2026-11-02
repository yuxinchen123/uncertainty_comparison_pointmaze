"""Run bounded metadata release tests on the existing owned CPU allocation."""
import ctypes,datetime,hashlib,importlib.abc,json,os,pathlib,socket,subprocess,sys,traceback,unittest
from unittest.mock import patch
P=pathlib.Path(__file__).resolve().parent
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdecimal()
step=os.environ['SLURM_STEP_ID'];print('ACTUAL_OWNED_STEP=6783003.'+step,flush=True)
names=['release_common_confirmation.py','test_release_common_confirmation.py','release_frozen_inputs.json','author_review.json']
images=lambda:{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in names}
before=images();blocked=[]
frozen=json.loads((P/'release_frozen_inputs.json').read_text());protected={path:value['sha256'] for path,value in {**frozen['shared_files'],**frozen['jobs']['1']['files'],**frozen['jobs']['2']['files']}.items()}
protected_before={path:hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() for path in protected};assert protected_before==protected
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy'}:blocked.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,Block())
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('shared-library loads forbidden'))
socket.socket=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('network action forbidden'))
real_popen=subprocess.Popen
allowed={str(P.parents[4]),'/p/rlprojects/RND_worktrees/atari-cleanrl-source-d28e673','/p/rlprojects/RND_worktrees/2026-11-01-18-47_common-confirmation-source_commit-9fcc43a478b93d731df780805fd7c3c514cdf67f'}
class GitOnly(real_popen):
 def __init__(self,args,*a,**k):
  cmd=list(args) if isinstance(args,(list,tuple)) else []
  assert len(cmd)>3 and cmd[:2]==['git','-C'] and cmd[2] in allowed and cmd[3] in {'show','cat-file','rev-parse','status','symbolic-ref','ls-tree','merge-base'},'live command forbidden: '+repr(cmd)
  super().__init__(args,*a,**k)
subprocess.Popen=GitOnly
sys.path.insert(0,str(P));import test_release_common_confirmation as tests
suite=unittest.defaultTestLoader.loadTestsFromModule(tests);result=unittest.TextTestRunner(verbosity=2).run(suite)
extra={}
try:
 with patch.object(tests.release.socket,'gethostname',return_value='portal05'):
  for copies in (1,2):
   actual=tests.release.checked(copies)
   assert actual['job_id']=={'1':'6783295','2':'6783296'}[str(copies)] and actual['publication_commit'] is None
   extra[str(copies)]={'actual_local_default_passed':True,'job_id':actual['job_id'],'no_intent_written':not (tests.release.PACKET/'release_intent.json').exists()}
except BaseException:
 extra['traceback']=traceback.format_exc()
protected_after={path:hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() for path in protected};assert images()==before and protected_after==protected_before and not blocked
raw=pathlib.Path('/proc/self/stat').read_text()
v={'schema_version':1,'kind':'fixed_common_confirmation_release_author_metadata_tests','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'6783003','step_id':step,'pid':os.getpid(),'birth':raw[raw.rfind(')')+2:].split()[19],'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'error_details':[{'test':t.id(),'traceback':s} for t,s in result.errors],'failure_details':[{'test':t.id(),'traceback':s} for t,s in result.failures],'passed':result.wasSuccessful() and set(extra)=={'1','2'},'actual_defaults':extra,'protected_file_count':len(protected),'protected_before':protected_before,'protected_after':protected_after,'source_before':before,'source_after':images(),'numerical_import_attempts':blocked,'shared_library_loads_blocked':True,'only_real_subprocesses_are_read_only_git':True,'actual_scheduler_queries':0,'actual_release_calls':0,'final_owned_inputs_written':False}
f=P/('author_controls_6783003_'+step+'.json')
with f.open('x') as stream:json.dump(v,stream,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps({'receipt':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'passed':v['passed'],'tests':result.testsRun,'actual_defaults':extra}));sys.exit(0 if v['passed'] else 1)
