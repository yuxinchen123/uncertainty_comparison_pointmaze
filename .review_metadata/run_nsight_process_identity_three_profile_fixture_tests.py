"""Run synthetic process identity tests only on the owned CPU, with all real actions blocked."""
import ast,ctypes,datetime,hashlib,importlib.abc,io,json,os,pathlib,subprocess,sys,unittest
W=pathlib.Path(__file__).resolve().parent.parent;S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
paths=[S/name for name in ('profile_selected.py','configured_measurement.py','nsight_process_identity.py','diagnose_nsight_process_tree.py','tests/test_nsight_process_identity.py','tests/test_profile_owned_processes.py','tests/test_nsight_process_diagnostic.py','tests/test_nsight_diagnostic_preparation.py','tests/test_profile_common_confirmation_helpers.py','tests/test_profile_selected.py','tests/test_configured_measurement.py')]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths.append(W/'.review_metadata/prepare_nsight_process_diagnostic.py')
before={str(p):sha(p) for p in paths};blocked=[]
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy','jaxtari','exploration_platform','pynvml'}:
   blocked.append(name);raise RuntimeError('real numerical import forbidden')
sys.meta_path.insert(0,NoNumerical())
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('shared-library loading forbidden'))
actual_popen=subprocess.Popen
cold_function=next(n for n in ast.walk(ast.parse((S/'tests/test_profile_owned_processes.py').read_text())) if isinstance(n,ast.FunctionDef) and n.name=='test_cold_new_helper_union_imports_without_site_or_numerical_modules')
cold_code=ast.literal_eval(next(n.value for n in ast.walk(cold_function) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='code' for t in n.targets)))
probe_tree=ast.parse((S/'tests/test_configured_measurement.py').read_text())
configured_probe=ast.literal_eval(next(n.value for n in probe_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ISOLATED_IMPORT_PROBE' for t in n.targets)))
cold_children=[];local_git_commands=[]
def permitted_popen(argv,*args,**kwargs):
 if argv[0]=='git':
  assert argv[1]=='-C' and pathlib.Path(argv[2]).is_absolute() and str(argv[2]).startswith('/tmp/') and pathlib.Path(argv[2]).name=='source'
  assert not any(value in argv for value in ('push','fetch','pull','clone','submodule','remote','ls-remote'))
  local_git_commands.append(argv)
 elif argv[:5]==[sys.executable,'-I','-S','-B','-c']:
  assert argv[5]==cold_code and len(argv)==7
  stage=pathlib.Path(argv[6]);assert stage.is_absolute() and str(stage).startswith('/tmp/') and stage.is_dir()
  cold_children.append(argv)
 else:
  assert argv[:4]==[sys.executable,'-I','-B','-c'] and argv[4]==configured_probe and len(argv)==7
  stage=pathlib.Path(argv[5]);assert stage.is_absolute() and str(stage).startswith('/tmp/') and stage.is_dir()
  cold_children.append(argv)
 return actual_popen(argv,*args,**kwargs)
subprocess.Popen=permitted_popen
def audit(event,args):
 if event in {'socket.connect','socket.bind','socket.getaddrinfo','os.system','os.kill','os.killpg'}:
  blocked.append(event);raise RuntimeError('network or live signal action forbidden')
sys.addaudithook(audit)
sys.path.insert(0,str(S/'tests'))
import test_nsight_process_identity as tests
import test_profile_owned_processes as profiles
import test_nsight_process_diagnostic as diagnostic
import test_nsight_diagnostic_preparation as preparation
import test_profile_selected as historical
import test_profile_common_confirmation_helpers as common
import test_configured_measurement as configured
suite=unittest.defaultTestLoader.loadTestsFromTestCase(configured.ConfiguredProfileTests)
out=io.StringIO();result=unittest.TextTestRunner(stream=out,verbosity=2).run(suite)
after={str(p):sha(p) for p in paths};assert before==after and not blocked
prefix='nsight_process_identity_three_profile_fixture_author_6783003_'+os.environ['SLURM_STEP_ID']
def put(name,raw):
 p=W/'.review_metadata'/name
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return {'path':str(p),'sha256':sha(p)}
raw=put(prefix+'.raw',out.getvalue().encode())
value={'passed':result.wasSuccessful(),'job_id':'6783003','step_id':os.environ['SLURM_STEP_ID'],'tests':result.testsRun,
 'failures':len(result.failures),'errors':len(result.errors),'raw':raw,'source_before':before,'source_after':after,
 'error_details':[{'test':t.id(),'traceback':m} for t,m in result.errors],'failure_details':[{'test':t.id(),'traceback':m} for t,m in result.failures],
 'synthetic_proc_files_and_mocked_pidfds_only':True,'actual_Nsight_or_profile_experiment':False,'real_processes_launched':len(cold_children),'allowed_cold_import_children':cold_children,'local_private_fixture_git_commands':local_git_commands,
 'actual_GPU_or_job_or_network_actions':0,'scientific_credit':0,'admission_granted':False}
receipt=put(prefix+'.json',(json.dumps(value,indent=2,sort_keys=True)+'\n').encode())
print(json.dumps({'passed':value['passed'],'tests':result.testsRun,'receipt':receipt}))
raise SystemExit(0 if result.wasSuccessful() else 1)
