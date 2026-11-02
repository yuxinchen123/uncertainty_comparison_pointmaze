"""Run the bounded import-provenance controls only on the newly verified owned CPU monitor."""
import ast,argparse,ctypes,datetime,hashlib,importlib.abc,json,os,pathlib,socket,subprocess,sys,tempfile,unittest
from unittest.mock import patch
W=pathlib.Path('/p/rlprojects/RND_worktrees/2026-11-01-16-36_direct-qualification-metadata-import-provenance_preserve-frozen-numerical-helper-paths');S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
parser=argparse.ArgumentParser();parser.add_argument('--audit-job-id',required=True);args=parser.parse_args()
assert args.audit_job_id.isdecimal() and os.environ.get('SLURM_JOB_ID')==args.audit_job_id and os.environ.get('SLURM_STEP_ID','').isdecimal()
assert args.audit_job_id not in {'6782816','6781231','6781087','6780635'}
step=os.environ['SLURM_STEP_ID'];print('ACTUAL_OWNED_STEP='+args.audit_job_id+'.'+step,flush=True)
r=pathlib.Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1');unit=next((r/'data').glob('2026-11-01-14-53_full-original-direct-qualification*'))
files=[S/'serval02_card1_binding.py',S/'tests/test_direct_metadata_import_provenance.py',S/'tests/direct_metadata_import_order_probe.py'];hashes=lambda:{str(p.relative_to(W)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};before=hashes();blocked=[]
protected_paths=[unit/'failure.json',unit/'measurement_plan.json',unit/'direct_controller_outcome.json',unit/'direct_process_wait_return.json',unit/'code/serval02_card1_binding.py',unit/'code/measure_selected.py']
plan=json.loads((unit/'measurement_plan.json').read_text())
protected_paths.extend(pathlib.Path(plan['source_root'])/name for name in plan['core']['files'])
protected_paths.extend([r/'SOURCE_AND_GATES_READY.json',r.parents[2]/'10_jax_exploration_platform/scripts/atari_pixel_campaign/serval02_card1_binding.py',r.parents[2]/'10_jax_exploration_platform/scripts/atari_pixel_campaign/measure_selected.py'])
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
modules=['test_direct_metadata_import_provenance','test_serval02_card1_binding']
class IndependentAdditionalChecks(unittest.TestCase):
 def test_added_and_nested_assignment_operations_refuse(self):
  "Reject repeated declarations using annotations or augmented assignment."
  from test_direct_metadata_import_provenance import MetadataImportProvenanceTests,EXPECTED
  fixture=MetadataImportProvenanceTests()
  for suffix in ["SOURCE_SCOPES += ('other/source',)","if False:\n    SOURCE_SCOPES: tuple = ()"]:
   with self.subTest(suffix=suffix),self.assertRaises(ValueError):
    fixture.read_fixture('SOURCE_SCOPES = '+repr(EXPECTED)+'\n'+suffix+'\n')
 def test_every_existing_function_and_remaining_admission_statement_is_unchanged(self):
  "Compare every old function while allowing only the intended metadata lookup."
  old=ast.parse(subprocess.check_output(['git','-C',str(W),'show','79a01e851ff92ddba7d0e12eaf8164cab2028878:10_jax_exploration_platform/scripts/atari_pixel_campaign/serval02_card1_binding.py']).decode())
  new=ast.parse((S/'serval02_card1_binding.py').read_text())
  old_functions={node.name:node for node in old.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
  new_functions={node.name:node for node in new.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
  self.assertEqual(set(new_functions),set(old_functions)|{'original_source_scopes'})
  replacement=ast.parse("scopes = adjacent('controller').SOURCE_SCOPES").body[0]
  function=new_functions['check_live_admission']
  matches=[index for index,node in enumerate(function.body) if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='scopes']
  self.assertEqual(len(matches),1);function.body[matches[0]]=replacement
  for name in old_functions:
   self.assertEqual(ast.dump(old_functions[name],include_attributes=False),ast.dump(new_functions[name],include_attributes=False),name)
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in modules)
suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentAdditionalChecks))
result=unittest.TextTestRunner(verbosity=2).run(suite)
probes=[]
for core,expected in [(unit/'code/serval02_card1_binding.py','legacy-failure'),(S/'serval02_card1_binding.py','corrected-success')]:
 command=[sys.executable,'-I','-S','-B',str(S/'tests/direct_metadata_import_order_probe.py'),'--core',str(core),'--unit',str(unit),'--expected',expected];allowed_probe.append(command)
 actual=subprocess.run(command,capture_output=True,text=True,timeout=180)
 record={'command':command,'returncode':actual.returncode,'stdout':actual.stdout,'stderr':actual.stderr}
 if actual.returncode==0:record['parsed']=json.loads(actual.stdout)
 probes.append(record)
tempfile.tempdir=old_temp;workspace.cleanup()
assert hashes()==before and protected()==saved and not blocked
value={'schema_version':1,'kind':'direct_metadata_import_provenance_independent_controls','job_id':args.audit_job_id,'step_id':step,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'modules':modules,'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'error_details':[{'test':t.id(),'traceback':s} for t,s in result.errors],'failure_details':[{'test':t.id(),'traceback':s} for t,s in result.failures],'cold_actual_order_probes':probes,'passed':result.wasSuccessful() and all(p['returncode']==0 for p in probes),'source_before':before,'source_after':hashes(),'protected_before':saved,'protected_after':protected(),'real_numerical_import_attempts':blocked,'actual_gpu_actions':0,'live_scheduler_queries':0,'network_actions':0,'source_admission_changed':False,'scientific_credit':0}
out=pathlib.Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-17-24-25_direct-metadata-import-provenance-independent-review_gpu-count-0')/('independent_controls_'+args.audit_job_id+'_'+step+'.json')
with out.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'receipt':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'passed':value['passed'],'tests':result.testsRun,'cold_probe_returncodes':[p['returncode'] for p in probes]}),flush=True);sys.exit(0 if value['passed'] else 1)
