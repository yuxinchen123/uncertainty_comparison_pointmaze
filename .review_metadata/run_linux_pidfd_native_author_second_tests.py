"""Run bounded selected-Python native pidfd tests only within the exact owned CPU allocation."""
import ast,hashlib,json,os,socket,sys,time,unittest
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ['SLURM_STEP_ID'].isdecimal()
w=Path(__file__).resolve().parents[1];r=w/'.review_metadata';s=w/'10_jax_exploration_platform/scripts/atari_pixel_campaign';frozen=r/'linux_pidfd_native_source_freeze_author_2.json';f=json.loads(frozen.read_text())
assert all(hashlib.sha256((w/p).read_bytes()).hexdigest()==sha for p,sha in f['files'].items())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in f['protected_files'].items())
os.environ['NATIVE_PIDFD_REVIEW_OUTPUT']=str(r/('linux_pidfd_actual_cpu_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']))
sys.path.insert(0,str(s/'tests'));sys.path.insert(0,str(s));start=time.time()
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName('test_linux_pidfd_process_control'),unittest.defaultTestLoader.loadTestsFromName('test_nsight_process_identity.ProcessObservationTests')])
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert all(hashlib.sha256((w/p).read_bytes()).hexdigest()==sha for p,sha in f['files'].items())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in f['protected_files'].items())
core_source=(s/'profile_selected.py').read_text();core=next(n for n in ast.parse(core_source).body if isinstance(n,ast.FunctionDef) and n.name=='_capture_core');assert hashlib.sha256(ast.get_source_segment(core_source,core).encode()).hexdigest()==f['capture_core_source_sha256']
identity_source=(s/'nsight_process_identity.py').read_text()
for name,expected in f['default_process_control_source_hashes'].items():
 node=next(n for n in ast.parse(identity_source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name);assert hashlib.sha256(ast.get_source_segment(identity_source,node).encode()).hexdigest()==expected
v={'kind':'actual_owned_cpu_native_pidfd_source_test_execution','job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'hostname':socket.gethostname(),'pid':os.getpid(),'sys_executable':sys.executable,'sys_version':sys.version,'source_freeze':{'path':str(frozen),'sha256':hashlib.sha256(frozen.read_bytes()).hexdigest()},'started_at_seconds':start,'finished_at_seconds':time.time(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'source_and_protected_unchanged':True,'actual_tool_or_gpu_execution':False,'standing_gpu_lock_opened':False,'native_process_control_only':True,'scientific_credit':0,'admission_granted':False}
p=r/('linux_pidfd_author_'+v['job_id']+'_'+v['step_id']+'.json');p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'record':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'passed':result.wasSuccessful(),'tests':result.testsRun}));sys.exit(0 if result.wasSuccessful() else 1)
