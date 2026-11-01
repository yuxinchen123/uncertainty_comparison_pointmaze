"""Run the direct-core metadata suite only on the already owned CPU allocation."""
import ctypes,datetime,hashlib,importlib.abc,json,os,pathlib,subprocess,sys,unittest
W=pathlib.Path(__file__).resolve().parents[1];S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
def require(value,message):
 if not value:raise RuntimeError(message)
require(os.environ.get('SLURM_JOB_ID')=='6781231','only ownedCPU6781231 may run this suite')
step=os.environ.get('SLURM_STEP_ID','');require(step.isdecimal(),'an actual numeric step is required')
frozen=W/'.review_metadata/direct_serval02_core_fourth_source_freeze.json';expected=json.loads(frozen.read_text())['files']
images=lambda:{name:hashlib.sha256((W/name).read_bytes()).hexdigest() for name in expected}
before=images();require(before==expected,'source changed before tests')
blocked=[]
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy'}:
   blocked.append(fullname);raise RuntimeError('numerical import prohibited: '+fullname)
sys.meta_path.insert(0,Block())
real_popen=subprocess.Popen
class LocalGitOnly(real_popen):
 def __init__(self,args,*a,**kw):
  cmd=list(args) if isinstance(args,(tuple,list)) else []
  allowed=False
  if cmd[:2]==['git','clone']:
   allowed=pathlib.Path(cmd[-2])==W and pathlib.Path(cmd[-1]).is_absolute() and str(pathlib.Path(cmd[-1])).startswith('/tmp/')
  elif len(cmd)>3 and cmd[:2]==['git','-C']:
   root=pathlib.Path(cmd[2]);allowed=root.is_absolute() and (str(root).startswith('/tmp/') or root==W and cmd[3] in {'show','status','rev-parse','ls-tree','symbolic-ref','cat-file'})
  require(allowed and not any(x in {'fetch','push','pull','remote'} for x in cmd),'only local temporary Git metadata operations are permitted')
  super().__init__(args,*a,**kw)
subprocess.Popen=LocalGitOnly
ctypes.CDLL=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('shared-library loads prohibited'))
sys.path[:0]=[str(S),str(S/'tests')]
suite=unittest.defaultTestLoader.loadTestsFromName('test_serval02_card1_binding');result=unittest.TextTestRunner(verbosity=2).run(suite)
after=images();require(before==after,'source changed during tests');require(not blocked,'numerical imports were attempted')
raw=pathlib.Path('/proc/self/stat').read_text()
value={'schema_version':1,'kind':'direct_serval02_core_author_metadata_tests','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'job_id':'6781231','step_id':step,'pid':os.getpid(),'birth':raw[raw.rfind(')')+2:].split()[19],
 'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
 'source_freeze_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),'source_before':before,'source_after':after,
 'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'passed':result.wasSuccessful(), 'error_details':[{'test':test.id(),'traceback':trace} for test,trace in result.errors], 'failure_details':[{'test':test.id(),'traceback':trace} for test,trace in result.failures],
 'blocked_numerical_imports':blocked,'process_calls_limited_to_local_git':True,'temporary_fork_test_uses_only_own_lock_metadata':True,
 'shared_library_loads_blocked':True,'actual_serval02_contact':False,'actual_gpu_operations':False,'scientific_credit':0}
p=W/'.review_metadata'/('direct_serval02_core_fourth_author_6781231_'+step+'.json')
with p.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'receipt':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'passed':result.wasSuccessful(),'tests':result.testsRun}))
sys.exit(0 if result.wasSuccessful() else 1)
