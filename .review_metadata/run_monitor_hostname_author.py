"""Test only monitor metadata on the current owned CPU allocation."""
import ctypes,datetime,hashlib,importlib.abc,json,os,pathlib,socket,subprocess,sys,unittest
from unittest.mock import patch
W=pathlib.Path(__file__).resolve().parents[1];S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
assert os.environ.get('SLURM_JOB_ID')=='6782816' and os.environ.get('SLURM_STEP_ID','').isdecimal()
step=os.environ['SLURM_STEP_ID'];print('ACTUAL_OWNED_STEP=6782816.'+step,flush=True)
files=[S/'monitor.py',S/'tests/test_monitor_process_hostname.py',S/'tests/fixtures/actual_processes_host_and_hostname_2026_11_02.json']
hashes=lambda:{str(p.relative_to(W)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
before=hashes();blocked=[]
root=pathlib.Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl');campaign=root/'10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1'
protected_paths=[root/'10_jax_exploration_platform/scripts/atari_pixel_campaign/monitor.py',campaign/'processes.json']
protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected_paths}
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'numpy','jax','jaxlib','torch','flax','optax','cupy','scipy'}:blocked.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,Block())
subprocess.Popen=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('live commands forbidden'))
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('shared-library loads forbidden'))
socket.socket=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('network actions forbidden'))
sys.path[:0]=[str(S),str(S/'tests')]
modules=['test_monitor_process_hostname','test_monitor']
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in modules);r=unittest.TextTestRunner(verbosity=2).run(suite)
after=hashes();protected_after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected_paths}
assert before==after and protected_after==protected and not blocked
v={'schema_version':1,'kind':'monitor_process_host_hostname_author_metadata_validation','job_id':'6782816','step_id':step,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'modules':modules,'tests':r.testsRun,'errors':len(r.errors),'failures':len(r.failures),'error_details':[{'test':t.id(),'traceback':s} for t,s in r.errors],'failure_details':[{'test':t.id(),'traceback':s} for t,s in r.failures],'passed':r.wasSuccessful(),'source_before':before,'source_after':after,'protected_before':protected,'protected_after':protected_after,'live_scheduler_commands':0,'numerical_import_attempts':blocked,'gpu_library_calls':0,'network_actions':0,'original_checkout_writes':0,'source_admission_changed':False}
p=W/'.review_metadata'/('monitor_host_hostname_author_6782816_'+step+'.json')
with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'receipt':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'passed':v['passed'],'tests':r.testsRun}),flush=True)
sys.exit(0 if v['passed'] else 1)
