import ast,builtins,ctypes,hashlib,json,os,pathlib,socket,subprocess,sys,time,unittest
w=pathlib.Path('/p/rlprojects/RND_worktrees/2026-11-01-10-44_selected-and-profile-configured-isolation_single-a4500_gpu-count-1_processor-count-8_host-memory-gib-64_unresolved-device-identity')
s=w/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
files=['configured_measurement.py','measure_selected.py','profile_selected.py','post_comparison_validation.py','device_memory_limit.py','measure_gpu_scaling.py','validate_training_deployment.py','tests/test_configured_measurement.py','uva_slurm_configured_isolation.py']
before={p:hashlib.sha256((s/p).read_bytes()).hexdigest() for p in files}
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
modules=['test_configured_measurement','test_measure_selected','test_profile_selected','test_post_comparison_validation','test_device_memory_limit','test_measure_gpu_scaling','test_validate_training_deployment']
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in modules)
started=time.time();result=unittest.TextTestRunner(verbosity=1).run(suite)
after={p:hashlib.sha256((s/p).read_bytes()).hexdigest() for p in files}
record={'schema_version':1,'kind':'configured_selected_profile_author_metadata','job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'modules':modules,'tests_run':result.testsRun,'failures':[(str(t),v) for t,v in result.failures],'errors':[(str(t),v) for t,v in result.errors],'before':before,'after':after,'unchanged':before==after,'blocked_actions':blocked,'duration_seconds':time.time()-started,'source_base':'983e4ea2b1b484f48a97757bad6a047df09ae795','allowed_isolated_import_probes':allowed_probes,'wrapper_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'numerical_work':False,'job_actions':0}
out=w/'.review_metadata';out.mkdir(exist_ok=True)
p=out/('configured_selected_profile_author_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']+'.json')
with p.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print('RECEIPT',p,'SHA',hashlib.sha256(p.read_bytes()).hexdigest(),'STEP',os.environ['SLURM_STEP_ID'],flush=True)
raise SystemExit(0 if result.wasSuccessful() and before==after and not blocked else 1)
