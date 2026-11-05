"""Run only corrected descriptor parsers/readers and bounded own-process CPU fixtures."""
import hashlib,json,os,socket,sys,time,unittest
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ['SLURM_STEP_ID'].isdecimal()
assert not os.environ.get('SLURM_STEP_GPUS') and not os.environ.get('CUDA_VISIBLE_DEVICES')
w=Path(__file__).resolve().parents[2];m=Path(__file__).resolve().parent;s=w/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
frozen=m/'fdinfo_correction_source_freeze.json';f=json.loads(frozen.read_text())
for path,sha in f['files'].items():assert hashlib.sha256((w/path).read_bytes()).hexdigest()==sha
for path,sha in f['protected_files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
started=time.time();step=os.environ['SLURM_STEP_ID'];artifacts=m/('fdinfo_correction_actual_cpu_'+step)
os.environ['NATIVE_PIDFD_REVIEW_OUTPUT']=str(artifacts/'native_provider')
os.environ['NATIVE_DIRECT_CPU_REVIEW_OUTPUT']=str(artifacts/'native_scope')
sys.path.insert(0,str(s));sys.path.insert(0,str(s/'tests'))
names=['test_native_pidfd_fdinfo','test_native_process_evidence.NativeSavedCallTests','test_native_process_role_evidence','test_native_direct_process_cpu',
 'test_linux_pidfd_process_control.NativePidfdContractTests.test_last_ordinary_call_allows_only_reserved_close',
 'test_linux_pidfd_process_control.NativePidfdContractTests.test_reserved_term_kill_close_remain_available_after_ordinary_limit']
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in names)
result=unittest.TextTestRunner(verbosity=2).run(suite)
for path,sha in f['files'].items():assert hashlib.sha256((w/path).read_bytes()).hexdigest()==sha
for path,sha in f['protected_files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
assert time.time()-started < 240
assert not any(name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform'} for name in sys.modules)
value={'kind':'author_corrected_raw_fdinfo_join_focused_owned_CPU_checks','job_id':os.environ['SLURM_JOB_ID'],'step_id':step,'hostname':socket.gethostname(),'pid':os.getpid(),'python':sys.executable,'source_freeze':{'path':str(frozen),'sha256':hashlib.sha256(frozen.read_bytes()).hexdigest()},'started_at_seconds':started,'finished_at_seconds':time.time(),'wall_time_limit_seconds':240,'native_worker_individual_time_limit_seconds':30,'tests_run':result.testsRun,'passed':result.wasSuccessful(),'failures':[{'test':str(test),'traceback':text} for test,text in result.failures],'errors':[{'test':str(test),'traceback':text} for test,text in result.errors],'source_and_protected_unchanged':True,'actual_own_self_and_fresh_own_CPU_children_only':True,'Nsight_GPU_or_standing_GPU_lock_actions':False,'scientific_credit':0,'admission_granted':False,'broad_122_case_batch_repeated':False}
p=m/('fdinfo_correction_author_'+os.environ['SLURM_JOB_ID']+'_'+step+'.json')
with p.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps({'receipt':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'tests':result.testsRun,'passed':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors)}))
raise SystemExit(0 if result.wasSuccessful() else 1)
