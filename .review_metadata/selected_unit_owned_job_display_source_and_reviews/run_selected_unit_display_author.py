"""Run only monitor display metadata tests on the verified existing CPU allocation."""
import ctypes,datetime,hashlib,importlib.abc,io,json,os
from pathlib import Path
import subprocess,sys,unittest
W=Path(__file__).resolve().parent.parent
S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
paths=[S/'monitor.py',S/'tests/test_monitor_selected_unit_jobs.py']
images=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
before=images();blocked=[]
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform','pynvml','matplotlib'}:
   blocked.append(name);raise RuntimeError('numerical import forbidden: '+name)
sys.meta_path.insert(0,NoNumerical())
def no_child(*args,**kwargs):blocked.append(repr(args));raise RuntimeError('real child process forbidden')
subprocess.Popen=no_child
sys.addaudithook(lambda event,args:(_ for _ in ()).throw(RuntimeError('network/device/library action forbidden: '+event)) if event in {'socket.connect','socket.bind','socket.getaddrinfo','ctypes.dlopen','os.system'} else None)
sys.path[:0]=[str(S),str(S/'tests')]
modules=['test_monitor_selected_unit_jobs','test_monitor','test_monitor_process_hostname']
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in modules)
log=io.StringIO();started=datetime.datetime.now(datetime.timezone.utc)
result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
assert before==images()
value={'schema_version':1,'kind':'selected_unit_owned_job_display_author_metadata_tests','passed':result.wasSuccessful() and not blocked,'job_id':'6783003','step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'cgroup':Path('/proc/self/cgroup').read_text(),'at_utc':started.isoformat(),'modules':modules,'tests':result.testsRun,'errors':[{'test':t.id(),'traceback':e} for t,e in result.errors],'failures':[{'test':t.id(),'traceback':e} for t,e in result.failures],'source_before':before,'source_after':images(),'output':log.getvalue(),'blocked_actions':blocked,'numerical_imports_or_GPU_work':False,'actual_scheduler_queries':0,'actual_job_actions':0}
p=W/'.review_metadata'/('selected_unit_display_author_6783003_'+os.environ['SLURM_STEP_ID']+'.json')
with p.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'passed':value['passed'],'tests':result.testsRun,'receipt':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'errors':value['errors'],'failures':value['failures']}));raise SystemExit(0 if value['passed'] else 1)

