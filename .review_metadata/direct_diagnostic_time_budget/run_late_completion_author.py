"""Check only final deadline classification using synthetic clocks and mock process-control operations."""
import hashlib,json,os,sys,time,unittest
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ['SLURM_STEP_ID'].isdecimal()
assert not os.environ.get('SLURM_STEP_GPUS') and not os.environ.get('CUDA_VISIBLE_DEVICES')
w=Path(__file__).resolve().parents[2];m=Path(__file__).resolve().parent;s=w/'10_jax_exploration_platform/scripts/atari_pixel_campaign';frozen=m/'late_completion_source_freeze.json';f=json.loads(frozen.read_text())
for name,sha in f['files'].items():assert hashlib.sha256((w/name).read_bytes()).hexdigest()==sha
for name,sha in f['protected_files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
step=os.environ['SLURM_STEP_ID'];os.environ['LATE_DIAGNOSTIC_COMPLETION_REVIEW_OUTPUT']=str(m/('late_completion_synthetic_'+step));sys.path.insert(0,str(s));sys.path.insert(0,str(s/'tests'));started=time.time()
names=['test_direct_diagnostic_late_completion',
 'test_direct_diagnostic_time_budget.DeadlineArithmeticTests.test_expired_cleanup_keeps_descriptor_and_does_not_signal_or_report_success',
 'test_direct_diagnostic_time_budget.DeadlineArithmeticTests.test_a_slow_owned_snapshot_cannot_send_a_signal_after_cleanup_expiry',
 'test_direct_diagnostic_time_budget.DeadlineArithmeticTests.test_cleanup_end_is_retained_once_and_capped_by_original_total',
 'test_direct_diagnostic_time_budget.DeadlineArithmeticTests.test_saved_snapshot_refuses_extended_domain_or_cleanup_window',
 'test_direct_diagnostic_time_budget.DeadlineArithmeticTests.test_common_work_limit_is_not_restarted_by_each_tool']
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in names);result=unittest.TextTestRunner(verbosity=2).run(suite)
for name,sha in f['files'].items():assert hashlib.sha256((w/name).read_bytes()).hexdigest()==sha
for name,sha in f['protected_files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
assert time.time()-started<60
assert not any(name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform'} for name in sys.modules)
v={'kind':'author_late_kernel_closure_and_final_custody_deadline_classification','job_id':os.environ['SLURM_JOB_ID'],'step_id':step,'pid':os.getpid(),'source_freeze':{'path':str(frozen),'sha256':hashlib.sha256(frozen.read_bytes()).hexdigest()},'started_at_seconds':started,'finished_at_seconds':time.time(),'tests_run':result.testsRun,'passed':result.wasSuccessful(),'errors':[{'test':str(test),'traceback':text} for test,text in result.errors],'failures':[{'test':str(test),'traceback':text} for test,text in result.failures],'source_and_protected_unchanged':True,'synthetic_clocks_and_process_control_only':True,'actual_PIDFD_prctl_GPU_or_standing_lock_operations':False,'requested_memory_MiB':512,'Slurm_time':'00:05:00','outer_validation_seconds':60,'broad53_or122_batches_repeated':False,'scientific_credit':0,'admission_granted':False}
p=m/('late_completion_author_'+step+'.json');p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps({'receipt':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'tests':result.testsRun,'passed':result.wasSuccessful(),'errors':len(result.errors),'failures':len(result.failures)}));raise SystemExit(0 if result.wasSuccessful() else 1)
