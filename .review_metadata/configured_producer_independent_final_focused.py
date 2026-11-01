import ast,builtins,ctypes,hashlib,json,os,pathlib,socket,subprocess,sys,time,unittest
w=pathlib.Path('/p/rlprojects/RND_worktrees/2026-11-01-10-44_selected-and-profile-configured-isolation_single-a4500_gpu-count-1_processor-count-8_host-memory-gib-64_unresolved-device-identity')
s=w/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
files=['configured_measurement.py','measure_selected.py','profile_selected.py','post_comparison_validation.py','device_memory_limit.py','measure_gpu_scaling.py','validate_training_deployment.py','tests/test_configured_measurement.py','uva_slurm_configured_isolation.py']
before={p:hashlib.sha256((s/p).read_bytes()).hexdigest() for p in files}
assert os.environ.get('SLURM_JOB_ID')=='6781231'
freeze=pathlib.Path(sys.argv[1]);assert freeze.parent==w/'.review_metadata' and freeze.name.endswith('_source_freeze.json')
assert hashlib.sha256(freeze.read_bytes()).hexdigest()=='8398f24bfe62254523d84d5187fc3c5b1c52d7aa73b8b9b5600778ac349e4523'
expected=json.loads(freeze.read_text());assert before=={**expected['own_files'],'uva_slurm_configured_isolation.py':expected['borrowed_dependency']['sha256']}
previous=json.loads((w/'.review_metadata/configured_selected_profile_producer_renderer_complete_source_freeze.json').read_text())
changed={name for name in expected['own_files'] if expected['own_files'][name]!=previous['own_files'][name]}
assert changed=={'configured_measurement.py','tests/test_configured_measurement.py'}
assert expected['borrowed_dependency']['sha256']=='d659dd1f21ddd2d7bb0474b4ded0ae4ea129c00365984c1fe7f84439b8eba694'
limits={'new_jobs':0,'GPU_experiments':0,'source_admission_changed':False}
assert all(expected.get(k)==v for k,v in limits.items())
tree=ast.parse((s/'tests/test_configured_measurement.py').read_text())
probe=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ISOLATED_IMPORT_PROBE' for t in n.targets))
protected_root=pathlib.Path('/p/rlprojects/RND_worktrees/atari-cleanrl-source-d28e673')
protected_paths=sorted((protected_root/'10_jax_exploration_platform/src/exploration_platform/atari_pixels').rglob('*.py'))
protected_paths.extend(protected_root/'10_jax_exploration_platform/scripts/atari_pixel_campaign'/name for name in ('paired_benchmark.py','ideas.py','check_rendering.py'))
protected_before={str(p.relative_to(protected_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected_paths}
history_paths=sorted((w/'.review_metadata').glob('*.json'))
history_before={str(p.relative_to(w)):hashlib.sha256(p.read_bytes()).hexdigest() for p in history_paths}
blocked=[];allowed_probes=[]
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in {'jax','jaxlib','numpy','torch','jaxtari','exploration_platform','pynvml','cupy','numba','scipy','matplotlib'}:
  blocked.append(name);raise AssertionError('numerical import forbidden:'+name)
 return original(name,*args,**kwargs)
builtins.__import__=guarded
real_popen=subprocess.Popen
def command_guard(command,*args,**kwargs):
 if isinstance(command,list) and len(command)==6 and command[:4]==['git','clone','-q','--no-hardlinks']:
  origin,target=(pathlib.Path(x) for x in command[4:])
  assert origin.is_absolute() and target.is_absolute() and origin.resolve(strict=True)==origin and origin.is_relative_to('/tmp') and target.is_relative_to('/tmp')
  assert origin.stat().st_uid==os.getuid() and (origin/'.git').is_dir() and not target.exists()
  return real_popen(command,*args,**kwargs)
 if isinstance(command,list) and command and command[0]=='git' and not any(x in command for x in ('fetch','push','pull','clone','ls-remote')):
  return real_popen(command,*args,**kwargs)
 if isinstance(command,list) and len(command)==7 and command[0]==sys.executable and command[1:4]==['-I','-B','-c'] and command[4]==probe:
  directory=pathlib.Path(command[5]);assert directory.is_absolute() and directory.resolve(strict=True).is_relative_to(pathlib.Path('/tmp'))
  names=json.loads(command[6]);assert isinstance(names,list) and all(isinstance(x,str) and pathlib.Path(x).name==x and x.endswith('.py') for x in names)
  allowed_probes.append({'argv':command,'probe_sha256':hashlib.sha256(probe.encode()).hexdigest()})
  return real_popen(command,*args,**kwargs)
 blocked.append(str(command));raise AssertionError('live command forbidden')
subprocess.Popen=command_guard
def audit(event,args):
 if event in {'socket.connect','ctypes.dlopen'}:
  blocked.append(event);raise AssertionError('live operation forbidden:'+event)
 if event=='open' and isinstance(args[0],str) and args[0].startswith('/dev/'):
  blocked.append(event);raise AssertionError('device open forbidden')
sys.addaudithook(audit)
sys.path.insert(0,str(s));sys.path.insert(0,str(s/'tests'))
modules=['test_configured_measurement']
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in modules)
from unittest.mock import patch
import types,threading,tempfile
import configured_measurement as support
import test_configured_measurement as fixture
extra={}
class DirectClosures(unittest.TestCase):
 def test_allocator_failure_retains_exact_raw_observation(self):
  with tempfile.TemporaryDirectory() as temporary:
   d=pathlib.Path(temporary)/'paired_memory';v=support.MemoryObservation.__new__(support.MemoryObservation)
   v.uuid,v.pid,v.directory=fixture.GPU,fixture.PID,d;v.window_started=1.5;v.samples=[fixture.memory_window()['samples'][0]];v.errors=[];v.normal_stop=True;v.counter=0
   v.stop=threading.Event();v.thread=types.SimpleNamespace(is_alive=lambda:False,join=lambda timeout:None)
   v.sample=lambda:fixture.memory_window()['samples'][-1]
   def fail():raise RuntimeError('synthetic allocator failure')
   v.device=types.SimpleNamespace(memory_stats=fail)
   with self.assertRaises(RuntimeError):v.finish()
   proof=json.loads((d/'memory_window_0.json').read_text());self.assertIsNone(proof['allocator']);self.assertEqual(proof['allocator_error']['observation'],'unknown');self.assertEqual(len(proof['samples']),2);self.assertTrue(proof['errors'])
   extra['allocator_failure_retains_proof']=True
 def test_invalid_successful_device_row_retains_only_anonymous_facts(self):
  with tempfile.TemporaryDirectory() as temporary:
   d=pathlib.Path(temporary)/'paired_memory';v=support.MemoryObservation.__new__(support.MemoryObservation);v.uuid,v.pid,v.directory=fixture.GPU,os.getpid(),d
   birth=pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
   v.binding={'process':{'process_start_time':birth},'original_environment':{}}
   raw=b'GPU-ffffffff-ffff-ffff-ffff-ffffffffffff, 20470, 1, 20469\n'
   with patch.object(support,'bounded_query',return_value=fixture.raw(raw)):
    with self.assertRaises(ValueError):v.sample()
   paths=list(d.iterdir());self.assertEqual(len(paths),1);proof=json.loads(paths[0].read_text())
   self.assertEqual(proof['stdout_sha256'],hashlib.sha256(raw).hexdigest());self.assertEqual(proof['stdout_bytes'],len(raw));self.assertNotIn('GPU-ffffffff',paths[0].read_text());self.assertNotIn('stdout',proof);self.assertFalse(proof['measurement_valid'])
   extra['invalid_device_capture_retains_anonymous_proof']=True
 def test_nonboolean_capture_completeness_refuses(self):
  sample=support.process_sample(fixture.GPU,fixture.PID,lambda command:fixture.raw(f'{fixture.GPU}, {fixture.PID}, 512\n'.encode(),complete=1))
  self.assertFalse(sample['valid']);extra['nonboolean_completeness_refused']=True
 def test_actual_backend_renderer_matrix_is_fully_derived(self):
  run=pathlib.Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
  path=run/'data/idea-position-16__fuse_network_bias_and_activation__hardware-jaguar03_rtx_a4500__copies-2__seed-start-1__updates-per-round-1__source-commit-d28e6735e79788f618acb0ebfde8609db1f776af/attempts/attempt-0004/installed_backend.json'
  backend=json.loads(path.read_text());root=pathlib.Path('/p/rlprojects/RND_worktrees/atari-cleanrl-source-d28e673');relative='10_jax_exploration_platform/scripts/atari_pixel_campaign/check_rendering.py'
  digest=hashlib.sha256((root/relative).read_bytes()).hexdigest();self.assertEqual(digest,'79e21acb6d1166b8cad73a9e8eab6c360a9655fbaba628dbedd8307c511da81e')
  plan={'source_root':str(root),'source_commit':'d28e6735e79788f618acb0ebfde8609db1f776af','qualification_sources':{'files':{relative:digest}}}
  matrix=support.renderer_contract(plan,backend)
  self.assertEqual({k:matrix[k] for k in ('rooms','cases','synthetic_pixel_cases')},{'rooms':33,'cases':132,'synthetic_pixel_cases':11})
  extra['actual_backend_renderer_contract']={'installed_backend_path':str(path),'installed_backend_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'matrix':matrix,'scope':'ordinary source/JSON AST only; no numerical import or new qualification'}
 def test_actual_pinned_original_test_contract_is_fully_derived(self):
  root=pathlib.Path('/p/rlprojects/RND_worktrees/atari-cleanrl-source-d28e673');commit='d28e6735e79788f618acb0ebfde8609db1f776af'
  paths=support.qualification_source_paths(root,commit)
  plan={'source_root':str(root),'source_commit':commit,'qualification_sources':{'files':{p:'source listed' for p in paths}}}
  contract=support.test_collection_contract(plan)
  self.assertTrue(contract);self.assertTrue(all(type(v) is int and v>0 for v in contract.values()))
  extra['actual_d28_test_contract']={'files':len(paths),'test_functions':len(contract),'cases':sum(contract.values()),'per_function_cases':contract}
suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(DirectClosures))
started=time.time();result=unittest.TextTestRunner(verbosity=1).run(suite)
after={p:hashlib.sha256((s/p).read_bytes()).hexdigest() for p in files}
protected_after={str(p.relative_to(protected_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected_paths}
history_after={str(p.relative_to(w)):hashlib.sha256(p.read_bytes()).hexdigest() for p in history_paths}
record={'schema_version':1,'kind':'configured_selected_profile_independent_final_metadata','extra':extra,'linked_full_run':{'path':str(w/'.review_metadata/configured_selected_profile_independent_final_6781231_54.json'),'sha256':'efc08f682f14d8286fd0920233f9cd2b83132b1821d9bf8530ade7105020c8bc','tests_run':173,'passes':172,'actual_native_scope_failure_retained':True},'changes_since_full_run':sorted(changed),'freeze_sha256':hashlib.sha256(freeze.read_bytes()).hexdigest(),'process_start_time':pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'hostname':socket.gethostname(),'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'modules':modules,'tests_run':result.testsRun,'failures':[(str(t),v) for t,v in result.failures],'errors':[(str(t),v) for t,v in result.errors],'before':before,'after':after,'unchanged':before==after,'blocked_actions':blocked,'duration_seconds':time.time()-started,'source_base':'983e4ea2b1b484f48a97757bad6a047df09ae795','protected_numerical_before':protected_before,'protected_numerical_after':protected_after,'protected_numerical_unchanged':protected_before==protected_after,'history_before':history_before,'history_after':history_after,'history_unchanged':history_before==history_after,'allowed_isolated_import_probes':allowed_probes,'wrapper_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'numerical_work':False,'job_actions':0}
out=w/'.review_metadata';out.mkdir(exist_ok=True)
p=out/('configured_selected_profile_independent_final_focus_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']+'.json')
with p.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print('RECEIPT',p,'SHA',hashlib.sha256(p.read_bytes()).hexdigest(),'STEP',os.environ['SLURM_STEP_ID'],flush=True)
raise SystemExit(0 if result.wasSuccessful() and before==after and protected_before==protected_after and history_before==history_after and not blocked else 1)
