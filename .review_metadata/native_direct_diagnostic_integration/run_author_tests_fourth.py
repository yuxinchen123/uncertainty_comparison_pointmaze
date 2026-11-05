"""Run frozen diagnostic metadata and declared own-process CPU fixtures on the exact owned allocation."""
import hashlib,json,os,socket,subprocess,sys,time,unittest
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ['SLURM_STEP_ID'].isdecimal()
assert not os.environ.get('SLURM_STEP_GPUS') and not os.environ.get('CUDA_VISIBLE_DEVICES')
w=Path(__file__).resolve().parents[2];m=Path(__file__).resolve().parent;s=w/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
frozen=m/'author_source_freeze_4.json';f=json.loads(frozen.read_text())
for path,sha in f['files'].items():assert hashlib.sha256((w/path).read_bytes()).hexdigest()==sha
for path,sha in f['protected_files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
started=time.time();step=os.environ['SLURM_STEP_ID'];artifacts=m/('actual_cpu_'+step)
os.environ['NATIVE_PIDFD_REVIEW_OUTPUT']=str(artifacts/'native_provider')
os.environ['NATIVE_DIRECT_CPU_REVIEW_OUTPUT']=str(artifacts/'native_scope')
os.environ['DIRECT_CPU_SCOPE_REVIEW_OUTPUT']=str(artifacts/'default_scope')
sys.path.insert(0,str(s));sys.path.insert(0,str(s/'tests'))
modules=['test_linux_pidfd_process_control','test_native_process_evidence','test_native_diagnostic_order','test_native_process_role_evidence','test_native_direct_process_cpu','test_direct_profile_process','test_direct_tool_descendants','test_nsight_process_identity.ProcessObservationTests']
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in modules)
def flattened(suite):
 for item in suite:
  if isinstance(item,unittest.TestSuite):yield from flattened(item)
  else:yield item
old_case='test_direct_profile_process.DirectDiagnosticSourceTests.test_numerical_files_and_native_cuda_functions_remain_unchanged'
cases=list(flattened(suite));assert sum(case.id()==old_case for case in cases)==1
suite=unittest.TestSuite(case for case in cases if case.id()!=old_case)
result=unittest.TextTestRunner(verbosity=2).run(suite)
script="import sys,unittest;sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2]);r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName(sys.argv[3]));raise SystemExit(0 if r.wasSuccessful() else 1)"
legacy=subprocess.run(['/usr/bin/python3','-I','-S','-B','-c',script,str(s),str(s/'tests'),old_case],capture_output=True,check=False)
(m/('historical_AST_'+step+'_stdout.raw')).write_bytes(legacy.stdout);(m/('historical_AST_'+step+'_stderr.raw')).write_bytes(legacy.stderr)
print(legacy.stderr.decode(),file=sys.stderr)
passed=result.wasSuccessful() and legacy.returncode==0
for path,sha in f['files'].items():assert hashlib.sha256((w/path).read_bytes()).hexdigest()==sha
for path,sha in f['protected_files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
assert not any(name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform'} for name in sys.modules)
value={'kind':'author_native_direct_diagnostic_integration_owned_CPU_tests','job_id':os.environ['SLURM_JOB_ID'],'step_id':step,'hostname':socket.gethostname(),'pid':os.getpid(),'python':sys.executable,'sys_version':sys.version,'source_freeze':{'path':str(frozen),'sha256':hashlib.sha256(frozen.read_bytes()).hexdigest()},'started_at_seconds':started,'finished_at_seconds':time.time(),'tests_run':result.testsRun+1,'selected_python_tests':result.testsRun,'historical_python_source_tests':1,'historical_source_returncode':legacy.returncode,'passed':passed,'failures':[{'test':str(test),'traceback':text} for test,text in result.failures],'errors':[{'test':str(test),'traceback':text} for test,text in result.errors],'source_and_protected_unchanged':True,'actual_own_self_and_fresh_own_CPU_children_only':True,'tool_or_GPU_or_standing_GPU_lock_action':False,'scientific_credit':0,'admission_granted':False}
p=m/('author_'+os.environ['SLURM_JOB_ID']+'_'+step+'.json')
with p.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps({'receipt':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'tests':result.testsRun+1,'passed':passed,'failures':len(result.failures),'errors':len(result.errors)}))
raise SystemExit(0 if passed else 1)
