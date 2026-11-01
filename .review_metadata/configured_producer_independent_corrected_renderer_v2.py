import ast,builtins,ctypes,hashlib,json,os,pathlib,socket,subprocess,sys,time,unittest
w=pathlib.Path('/p/rlprojects/RND_worktrees/2026-11-01-10-44_selected-and-profile-configured-isolation_single-a4500_gpu-count-1_processor-count-8_host-memory-gib-64_unresolved-device-identity')
s=w/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
files=['configured_measurement.py','measure_selected.py','profile_selected.py','post_comparison_validation.py','device_memory_limit.py','measure_gpu_scaling.py','validate_training_deployment.py','tests/test_configured_measurement.py','uva_slurm_configured_isolation.py']
before={p:hashlib.sha256((s/p).read_bytes()).hexdigest() for p in files}
freeze=w/'.review_metadata/configured_selected_profile_producer_corrected_source_freeze.json'
assert hashlib.sha256(freeze.read_bytes()).hexdigest()=='8968590f633c189a8d02c0e58fc4fe4c45be97a5bb35830d50238a503d291bb3'
expected=json.loads(freeze.read_text()); assert before=={**expected['own_files'],'uva_slurm_configured_isolation.py':expected['borrowed_dependency']['sha256']}
assert os.environ.get('SLURM_JOB_ID')=='6781231'
tree=ast.parse((s/'tests/test_configured_measurement.py').read_text())
probe=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ISOLATED_IMPORT_PROBE' for t in n.targets))
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
modules=['test_configured_measurement.QualificationReaderTests']
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in modules)
import configured_measurement as support
import test_configured_measurement as tc
extra={}
class DirectChecks(unittest.TestCase):
 def test_reduced_renderer_is_currently_accepted(self):
  f=tc.QualificationReaderTests();f.setUp()
  try:
   f.plan['source_commit']='d28e6735e79788f618acb0ebfde8609db1f776af';f.result['source_commit']=f.plan['source_commit']
   value=f.validate();self.assertEqual(value['renderer']['rooms'],2);self.assertEqual(value['renderer']['synthetic_pixel_cases'],1)
   core=pathlib.Path('/p/rlprojects/RND/reference_repos/jaxtari_pixel_study/src/jaxtari/games/montezuma_revenge/core.py')
   tree=ast.parse(core.read_text()); default=None
   for n in ast.walk(tree):
    if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='MAX_ROOMS':
     default=next(ast.literal_eval(k.value) for k in n.value.keywords if k.arg=='default')
   self.assertEqual(default,33)
   source=pathlib.Path('/p/rlprojects/RND_worktrees/atari-cleanrl-source-d28e673/10_jax_exploration_platform/scripts/atari_pixel_campaign/check_rendering.py')
   renderer=ast.parse(source.read_text());fun=next(n for n in renderer.body if isinstance(n,ast.FunctionDef) and n.name=='check_synthetic_pixels')
   counts={}
   for n in fun.body:
    if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'colors','patterns'}:
     counts[n.targets[0].id]=len(n.value.args[0].elts)
   self.assertEqual(counts,{'colors':9,'patterns':2})
   extra['reduced_renderer_accepted']={'rooms':2,'cases':8,'synthetic_pixel_cases':1,'actual_source_room_count':default,'actual_source_cases':4*default,'actual_source_synthetic_cases':sum(counts.values()),'unrelated_mocked_checks':['source','physical binding','original configuration'],'constants_source':str(core),'constants_sha256':hashlib.sha256(core.read_bytes()).hexdigest(),'renderer_source':str(source),'renderer_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
  finally:f.doCleanups()
suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(DirectChecks))
started=time.time();result=unittest.TextTestRunner(verbosity=1).run(suite)
after={p:hashlib.sha256((s/p).read_bytes()).hexdigest() for p in files}
record={'schema_version':1,'kind':'configured_selected_profile_independent_corrected_reader_findings','extra':extra,'freeze_sha256':hashlib.sha256(freeze.read_bytes()).hexdigest(),'process_start_time':pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'modules':modules,'tests_run':result.testsRun,'failures':[(str(t),v) for t,v in result.failures],'errors':[(str(t),v) for t,v in result.errors],'before':before,'after':after,'unchanged':before==after,'blocked_actions':blocked,'duration_seconds':time.time()-started,'source_base':'983e4ea2b1b484f48a97757bad6a047df09ae795','allowed_isolated_import_probes':allowed_probes,'wrapper_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'numerical_work':False,'job_actions':0}
out=w/'.review_metadata';out.mkdir(exist_ok=True)
p=out/('configured_selected_profile_independent_corrected_reader_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']+'.json')
with p.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print('RECEIPT',p,'SHA',hashlib.sha256(p.read_bytes()).hexdigest(),'STEP',os.environ['SLURM_STEP_ID'],flush=True)
raise SystemExit(0 if result.wasSuccessful() and before==after and not blocked else 1)
