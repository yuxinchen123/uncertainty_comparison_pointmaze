"""Run the bounded import-provenance controls only on the newly verified owned CPU monitor."""
import argparse,ctypes,datetime,hashlib,importlib.abc,json,os,pathlib,socket,subprocess,sys,tempfile,unittest
from unittest.mock import patch
W=pathlib.Path(__file__).resolve().parents[1];S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
parser=argparse.ArgumentParser();parser.add_argument('--audit-job-id',required=True);args=parser.parse_args()
assert args.audit_job_id.isdecimal() and os.environ.get('SLURM_JOB_ID')==args.audit_job_id and os.environ.get('SLURM_STEP_ID','').isdecimal()
assert args.audit_job_id not in {'6782816','6781231','6781087','6780635'}
step=os.environ['SLURM_STEP_ID'];print('ACTUAL_OWNED_STEP='+args.audit_job_id+'.'+step,flush=True)
r=pathlib.Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1');unit=next((r/'data').glob('2026-11-01-14-53_full-original-direct-qualification*'))
files=[S/'serval02_card1_binding.py',S/'tests/test_direct_metadata_import_provenance.py',S/'tests/direct_metadata_import_order_probe.py'];hashes=lambda:{str(p.relative_to(W)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};before=hashes();blocked=[]
protected_paths=[unit/'failure.json',unit/'measurement_plan.json',unit/'direct_controller_outcome.json',unit/'direct_process_wait_return.json',unit/'code/serval02_card1_binding.py',unit/'code/measure_selected.py']
protected=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected_paths};saved=protected()
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy','exploration_platform'}:blocked.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,NoNumerical());ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('shared-library load forbidden'))
sys.addaudithook(lambda event,a:(_ for _ in ()).throw(RuntimeError('network forbidden')) if event in {'socket.connect','socket.bind','socket.getaddrinfo'} else None)
real_popen=subprocess.Popen;workspace=tempfile.TemporaryDirectory(prefix='direct_import_provenance_metadata_');old_temp=tempfile.tempdir;tempfile.tempdir=workspace.name
allowed_probe=[]
class LocalOnly(real_popen):
 def __init__(self,command,*a,**k):
  assert isinstance(command,(list,tuple)) and command,'structured command required'
  if list(command[:5])==['git','clone','--shared','--no-checkout','--quiet']:
   assert len(command)==7 and command[5]==str(W) and pathlib.Path(command[6]).is_relative_to(pathlib.Path(workspace.name)),'only the exact local immutable-source fixture clone is allowed'
  elif command[0]=='git':
   assert len(command)>3 and command[1]=='-C' and ((pathlib.Path(command[2]).is_relative_to(pathlib.Path(workspace.name)) and command[3] in {'init','add','-c','commit','checkout','rev-parse','symbolic-ref','status','show','ls-tree','cat-file','update-index','sparse-checkout'}) or (command[2]==str(W) and command[3] in {'rev-parse','status','show','ls-tree','cat-file','merge-base'})),'only isolated metadata Git fixtures may change; the exact source worktree is read-only'
  else:assert list(command) in allowed_probe,'only the exact cold metadata probe is allowed'
  super().__init__(command,*a,**k)
subprocess.Popen=LocalOnly;sys.path[:0]=[str(S),str(S/'tests')]
modules=['test_direct_metadata_import_provenance','test_serval02_card1_binding'];suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in modules);result=unittest.TextTestRunner(verbosity=2).run(suite)
probes=[]
for core,expected in [(unit/'code/serval02_card1_binding.py','legacy-failure'),(S/'serval02_card1_binding.py','corrected-success')]:
 command=[sys.executable,'-I','-S','-B',str(S/'tests/direct_metadata_import_order_probe.py'),'--core',str(core),'--unit',str(unit),'--expected',expected];allowed_probe.append(command)
 actual=subprocess.run(command,capture_output=True,text=True,timeout=180)
 record={'command':command,'returncode':actual.returncode,'stdout':actual.stdout,'stderr':actual.stderr}
 if actual.returncode==0:record['parsed']=json.loads(actual.stdout)
 probes.append(record)
tempfile.tempdir=old_temp;workspace.cleanup()
assert hashes()==before and protected()==saved and not blocked
value={'schema_version':1,'kind':'direct_metadata_import_provenance_author_controls','job_id':args.audit_job_id,'step_id':step,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'modules':modules,'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'error_details':[{'test':t.id(),'traceback':s} for t,s in result.errors],'failure_details':[{'test':t.id(),'traceback':s} for t,s in result.failures],'cold_actual_order_probes':probes,'passed':result.wasSuccessful() and all(p['returncode']==0 for p in probes),'source_before':before,'source_after':hashes(),'protected_before':saved,'protected_after':protected(),'real_numerical_import_attempts':blocked,'actual_gpu_actions':0,'live_scheduler_queries':0,'network_actions':0,'source_admission_changed':False,'scientific_credit':0}
out=W/'.review_metadata'/('import_provenance_author_'+args.audit_job_id+'_'+step+'.json')
with out.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'receipt':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'passed':value['passed'],'tests':result.testsRun,'cold_probe_returncodes':[p['returncode'] for p in probes]}),flush=True);sys.exit(0 if value['passed'] else 1)
