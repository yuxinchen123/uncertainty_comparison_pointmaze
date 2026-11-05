import datetime,hashlib,importlib.util,json,os,stat,sys,time,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
PATHS=('launch_diagnostic_once.py','test_launch_diagnostic_once.py','schema/diagnose_configured_native_process_approved.py','accepted_20_09_launcher.py','accepted_20_09_tests.py')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def records():return {name:{'sha256':digest(D/name),'bytes':(D/name).stat().st_size,'mode':oct(stat.S_IMODE((D/name).stat().st_mode))} for name in PATHS}
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
if os.environ.get('SLURM_JOB_ID')!='6821437' or not os.environ.get('SLURM_STEP_ID','').isdigit():raise ValueError('exact owned CPU step required')
started=time.time();before=records();freeze=D/'author_source_freeze_3.json'
write(freeze,{'kind':'configured_native_diagnostic_launcher_source_freeze','files':before,'actual_prepared_plan_available':False,'actual_packet_binding_created':False,'actual_scheduler_actions':0,'scientific_credit':0})
spec=importlib.util.spec_from_file_location('configured_launcher_tests',D/'test_launch_diagnostic_once.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
methods=("test_origin_tracking_included_commit_allows_metadata_check_without_network","test_absent_origin_refuses_publication","test_absent_origin_tracking_refuses_publication","test_local_only_new_commit_refuses_until_included_in_exact_tracking_history","test_stale_origin_tracking_history_refuses_new_publication","test_different_origin_or_branch_mapping_refuses_publication")
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName("LauncherTests."+name,module) for name in methods);result=unittest.TextTestRunner(verbosity=2).run(suite)
after=records();step=os.environ['SLURM_STEP_ID'];receipt=D/('author_checks_6821437_'+step+'.json')
record={'kind':'configured_native_diagnostic_launcher_author_checks','passed':result.wasSuccessful() and before==after,'tests_run':result.testsRun,'failures':[{'test':str(t),'traceback':v} for t,v in result.failures],'errors':[{'test':str(t),'traceback':v} for t,v in result.errors],'source_freeze':{'path':str(freeze),'sha256':digest(freeze)},'source_before':before,'source_after':after,'source_unchanged':before==after,'job_id':os.environ['SLURM_JOB_ID'],'step_id':step,'hostname':os.uname().nodename,'pid':os.getpid(),'started_at_seconds':started,'finished_at_seconds':time.time(),'actual_scheduler_actions':0,'synthetic_scheduler_only':True,'private_local_git_only':True,'actual_prepared_binding_or_job_created':False,'scientific_credit':0}
write(receipt,record)
print(json.dumps({'passed':record['passed'],'tests_run':result.testsRun,'receipt':str(receipt),'sha256':digest(receipt),'freeze_sha256':digest(freeze),'step':step,'elapsed_seconds':record['finished_at_seconds']-started},sort_keys=True))
if not record['passed']:raise SystemExit(1)
