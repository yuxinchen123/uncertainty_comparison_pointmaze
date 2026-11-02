"""Run only synthetic release controls while real held proofs are being prepared."""
import ctypes,hashlib,importlib.abc,json,os,pathlib,subprocess,sys,unittest
P=pathlib.Path(__file__).resolve().parent
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdecimal()
names=['release_common_confirmation.py','test_release_common_confirmation.py']
images=lambda:{name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in names}
before=images();blocked=[]
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy'}:blocked.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,Block())
def denied(*args,**kwargs):raise RuntimeError('real subprocess forbidden in synthetic controls')
subprocess.Popen=denied
sys.addaudithook(lambda event,args:(_ for _ in ()).throw(RuntimeError('network/library action forbidden')) if event in {'socket.connect','socket.bind','socket.getaddrinfo','ctypes.dlopen'} else None)
sys.path.insert(0,str(P));import test_release_common_confirmation as tests
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
assert before==images() and not blocked
value={'kind':'unbound_fixed_confirmation_release_synthetic_author_controls','passed':result.wasSuccessful(),'job_id':'6783003','step_id':os.environ['SLURM_STEP_ID'],'tests':result.testsRun,'errors':[{'test':t.id(),'traceback':e} for t,e in result.errors],'failures':[{'test':t.id(),'traceback':e} for t,e in result.failures],'source_before':before,'source_after':images(),'numerical_import_attempts':blocked,'real_subprocesses':0,'actual_release_calls':0,'full_monitor_proofs_not_yet_bound':True}
out=P/('draft_author_controls_6783003_'+os.environ['SLURM_STEP_ID']+'.json')
with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'passed':value['passed'],'receipt':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'tests':result.testsRun}));raise SystemExit(0 if value['passed'] else 1)

