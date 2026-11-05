import hashlib,importlib.util,json,os,stat,sys,time,unittest
from pathlib import Path
W=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
if os.environ.get('SLURM_JOB_ID')!='6821437' or not os.environ.get('SLURM_STEP_ID','').isdigit():raise ValueError('owned numeric CPU step required')
def ref(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
paths=[S/'monitor.py',S/'tests/test_monitor_native_component.py',D/'monitor_before_native_display.py']
records=lambda:{str(p.relative_to(W)):{'sha256':ref(p)['sha256'],'bytes':p.stat().st_size,'mode':oct(stat.S_IMODE(p.stat().st_mode))} for p in paths}
before=records();freeze=D/'author_source_freeze_2.json';write(freeze,{'kind':'configured_native_component_monitor_source_freeze','files':before,'metadata_display_only':True,'live_monitor_edited':False,'actual_monitor_deployment':False,'scientific_credit':0})
sys.path.insert(0,str(S/'tests'));started=time.time();spec=importlib.util.spec_from_file_location('native_monitor_tests',S/'tests/test_monitor_native_component.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
methods=('test_component_metadata_reads_are_bounded_and_require_exact_snapshot_size','test_any_present_failure_marker_prevents_completed_display','test_actual_small_metadata_component_is_visible_with_zero_authority','test_failure_priority_and_mismatched_component_results_are_honest')
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName('NativeComponentMonitorTests.'+name,module) for name in methods);result=unittest.TextTestRunner(verbosity=2).run(suite)
after=records();step=os.environ['SLURM_STEP_ID'];p=D/('author_checks_6821437_'+step+'.json')
record={'kind':'configured_native_component_monitor_author_checks','passed':result.wasSuccessful() and before==after,'tests_run':result.testsRun,'errors':[{'test':str(t),'traceback':trace} for t,trace in result.errors],'failures':[{'test':str(t),'traceback':trace} for t,trace in result.failures],'source_freeze':ref(freeze),'source_before':before,'source_after':after,'source_unchanged':before==after,'job_id':os.environ['SLURM_JOB_ID'],'step_id':step,'started_at_seconds':started,'finished_at_seconds':time.time(),'actual_small_6825842_metadata_checked':True,'synthetic_private_negative_fixtures':True,'actual_scheduler_network_gpu_or_native_process_operations':0,'scientific_credit':0,'persistent_monitor_coverage_not_yet_changed':True}
write(p,record);print(json.dumps({'receipt':str(p),'sha256':ref(p)['sha256'],'source_freeze_sha256':ref(freeze)['sha256'],'passed':record['passed'],'tests_run':result.testsRun,'step':step,'elapsed_seconds':record['finished_at_seconds']-started},sort_keys=True))
if not record['passed']:raise SystemExit(1)
