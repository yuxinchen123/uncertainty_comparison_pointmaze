"""Run only bounded deadline, custody and role-reader checks on exact owned CPU."""
import hashlib,json,os,sys,time,unittest
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ['SLURM_STEP_ID'].isdecimal()
assert not os.environ.get('SLURM_STEP_GPUS') and not os.environ.get('CUDA_VISIBLE_DEVICES')
w=Path(__file__).resolve().parents[2];m=Path(__file__).resolve().parent;s=w/'10_jax_exploration_platform/scripts/atari_pixel_campaign';frozen=m/'author_source_freeze_6.json';f=json.loads(frozen.read_text())
for name,sha in f['files'].items():assert hashlib.sha256((w/name).read_bytes()).hexdigest()==sha
for name,sha in f['protected_files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
step=os.environ['SLURM_STEP_ID'];os.environ['DIAGNOSTIC_BUDGET_CPU_REVIEW_OUTPUT']=str(m/('actual_owned_CPU_'+step))
sys.path.insert(0,str(s));sys.path.insert(0,str(s/'tests'));started=time.time()
names=['test_direct_diagnostic_budget_boundaries','test_direct_diagnostic_time_budget.DeadlineArithmeticTests.test_preparation_records_one_total_budget_in_both_default_and_native_metadata','test_direct_diagnostic_time_budget.OriginAndControllerTests.test_controller_timeout_has_no_extra_wait_or_holder_signal','test_direct_diagnostic_time_budget.ActualControllerTimerTests.test_actual_own_controller_alarm_returns_promptly_and_restores_handler','test_direct_diagnostic_time_budget.DeadlineArithmeticTests.test_a_slow_owned_snapshot_cannot_send_a_signal_after_cleanup_expiry','test_direct_diagnostic_time_budget.ActualControllerTimerTests.test_actual_owned_cpu_cleanup_uses_one_synthetic_early_work_deadline']
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in names);result=unittest.TextTestRunner(verbosity=2).run(suite)
for name,sha in f['files'].items():assert hashlib.sha256((w/name).read_bytes()).hexdigest()==sha
for name,sha in f['protected_files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
assert time.time()-started<240
assert not any(name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform'} for name in sys.modules)
v={'kind':'author_shared_total_diagnostic_budget_focused_owned_CPU_checks','job_id':os.environ['SLURM_JOB_ID'],'step_id':step,'pid':os.getpid(),'python':sys.executable,'requested_memory_MiB':512,'requested_Slurm_time':'00:05:00','validation_timeout_seconds':240,'source_freeze':{'path':str(frozen),'sha256':hashlib.sha256(frozen.read_bytes()).hexdigest()},'started_at_seconds':started,'finished_at_seconds':time.time(),'tests_run':result.testsRun,'passed':result.wasSuccessful(),'errors':[{'test':str(test),'traceback':text} for test,text in result.errors],'failures':[{'test':str(test),'traceback':text} for test,text in result.failures],'source_and_protected_unchanged':True,'actual_own_CPU_timer_and_fresh_own_child_only':True,'Nsight_GPU_or_standing_GPU_lock_operations':False,'broad_122_batch_repeated':False,'scientific_credit':0,'admission_granted':False}
p=m/('author_'+os.environ['SLURM_JOB_ID']+'_'+step+'.json');p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'receipt':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'tests':result.testsRun,'passed':result.wasSuccessful(),'errors':len(result.errors),'failures':len(result.failures)}));raise SystemExit(0 if result.wasSuccessful() else 1)
