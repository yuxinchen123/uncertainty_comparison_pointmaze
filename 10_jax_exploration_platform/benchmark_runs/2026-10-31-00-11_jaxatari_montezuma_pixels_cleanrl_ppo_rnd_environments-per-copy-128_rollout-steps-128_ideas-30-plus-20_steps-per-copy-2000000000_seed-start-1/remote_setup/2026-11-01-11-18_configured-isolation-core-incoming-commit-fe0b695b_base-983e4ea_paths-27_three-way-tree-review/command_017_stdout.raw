"""Run only guarded metadata tests inside the already owned CPU monitor allocation."""
import ctypes,datetime,hashlib,importlib.abc,json,os,pathlib,subprocess,sys,unittest
W=pathlib.Path(__file__).resolve().parents[1]
S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
require=lambda condition,message: condition or (_ for _ in ()).throw(RuntimeError(message))
require(os.environ.get('SLURM_JOB_ID')=='6781231','only owned CPU 6781231 may run these tests')
step=os.environ.get('SLURM_STEP_ID','');require(step.isdecimal(),'an actual numeric step is required')
freeze=W/'.review_metadata/uva_configured_isolation_initial_source_freeze.json'
expected=json.loads(freeze.read_text())['files']
check=lambda:{path:hashlib.sha256((W/path).read_bytes()).hexdigest() for path in expected}
before=check();require(before==expected,'source differs before metadata tests')
blocked=[]
class BlockNumericalImports(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy'}:
   blocked.append(fullname);raise RuntimeError('numerical import prohibited: '+fullname)
sys.meta_path.insert(0,BlockNumericalImports())
real_popen=subprocess.Popen
class LocalGitOnly(real_popen):
 def __init__(self,args,*a,**kw):
  cmd=list(args) if isinstance(args,(tuple,list)) else []
  require(len(cmd)>3 and cmd[0]=='git' and cmd[1]=='-C' and pathlib.Path(cmd[2]).is_absolute()
    and str(pathlib.Path(cmd[2])).startswith('/tmp/'),'only local temporary Git metadata commands are permitted')
  require(not any(x in {'clone','fetch','push','pull','remote'} for x in cmd),'network Git is prohibited')
  super().__init__(args,*a,**kw)
subprocess.Popen=LocalGitOnly
ctypes.CDLL=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('shared-library loads are prohibited'))
sys.path[:0]=[str(S),str(S/'tests')]
suite=unittest.defaultTestLoader.loadTestsFromName('test_uva_slurm_configured_isolation')
result=unittest.TextTestRunner(verbosity=2).run(suite)
after=check();require(after==before,'source changed during tests')
require(not blocked,'numerical imports were attempted')
raw=pathlib.Path('/proc/self/stat').read_text()
record={'schema_version':1,'kind':'uva_configured_isolation_author_metadata_tests','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'job_id':'6781231','step_id':step,'pid':os.getpid(),'birth':raw[raw.rfind(')')+2:].split()[19],
 'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
 'source_freeze_sha256':hashlib.sha256(freeze.read_bytes()).hexdigest(),'source_before':before,'source_after':after,
 'tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'passed':result.wasSuccessful(),
 'numerical_imports_attempted':blocked,'all_process_commands_restricted_to_local_temporary_git':True,'shared_library_loading_prohibited':True,
 'scientific_credit':0,'gpu_qualification':False}
p=W/'.review_metadata'/('uva_configured_isolation_author_6781231_'+step+'.json')
with p.open('x') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'receipt':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'passed':result.wasSuccessful(),'tests':result.testsRun}))
sys.exit(0 if result.wasSuccessful() else 1)
