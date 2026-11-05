"""Run only synthetic group-package fixtures on the owned CPU with actual actions and backend reads blocked."""
import ctypes,datetime,hashlib,importlib.abc,io,json,os,pathlib,subprocess,sys,tempfile,unittest
W=pathlib.Path(__file__).resolve().parents[2];S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
paths=[S/'group_backend_package.py',S/'tests/test_group_backend_package.py',S/'direct_backend_image.py']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p):sha(p) for p in paths}
assert before[str(paths[0])]=='3a312803d173d767317e99efb53e56a6ba011374d8bde250db814ae1d9ddd76a'
assert before[str(paths[1])]=='972dbbf5a9bd054c470443c49e6f257fb97877e3f261d51f847eb0e8bc9470d1'
blocked=[];private_roots=[]
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','pynvml','numba','jaxtari','exploration_platform'}:
   blocked.append(name);raise RuntimeError('real numerical or GPU package import forbidden')
sys.meta_path.insert(0,NoNumerical())
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('actual shared-library load forbidden'))
subprocess.Popen=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('actual process or scheduler command forbidden'))
def audit(event,args):
 if event=='tempfile.mkdtemp':private_roots.append(str(args[0]))
 if event in {'socket.connect','socket.bind','socket.getaddrinfo','os.system','os.kill','os.killpg'}:
  blocked.append(event);raise RuntimeError('live operation forbidden')
 if event=='open' and isinstance(args[0],str):
  p=args[0]
  if p.startswith('/dev/nvidia') or p=='/proc/self/maps' or (p.endswith('.so') and not any(p.startswith(root+'/') for root in private_roots)) or p.startswith('/p/rlprojects/RND/.venvs/'):
   blocked.append(p);raise RuntimeError('actual backend/library/device read forbidden')
sys.addaudithook(audit)
sys.path.insert(0,str(S/'tests'))
import test_group_backend_package as tests
suite=unittest.defaultTestLoader.loadTestsFromModule(tests)
out=io.StringIO();result=unittest.TextTestRunner(stream=out,verbosity=2).run(suite)
after={str(p):sha(p) for p in paths};assert before==after and not blocked
prefix='corrected_author_fixtures_6783003_'+os.environ['SLURM_STEP_ID'];directory=pathlib.Path(__file__).resolve().parent
with (directory/(prefix+'.raw')).open('x') as stream:stream.write(out.getvalue());stream.flush();os.fsync(stream.fileno())
value={'kind':'group_backend_package_corrected_author_fixture_run','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':result.wasSuccessful(),'job_id':'6783003','step_id':os.environ['SLURM_STEP_ID'],'tests':result.testsRun,'errors':[{'test':t.id(),'traceback':m} for t,m in result.errors],'failures':[{'test':t.id(),'traceback':m} for t,m in result.failures],'source_before':before,'source_after':after,'blocked_actions':blocked,'only_small_synthetic_package_bytes_read':True,'actual_backend_binary_reads':0,'actual_library_or_device_or_GPU_or_network_operations':0,'GPU_qualification':False,'group_completeness':False,'scientific_credit':0,'admission_granted':False,'raw':{'path':str(directory/(prefix+'.raw')),'sha256':sha(directory/(prefix+'.raw'))}}
p=directory/(prefix+'.json')
with p.open('x') as stream:stream.write(json.dumps(value,indent=2,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps({'passed':value['passed'],'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'receipt':str(p),'sha256':sha(p)}))
raise SystemExit(0 if result.wasSuccessful() else 1)
