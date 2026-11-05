import ast,hashlib,importlib.util,io,json,os,pathlib,time,unittest
W=pathlib.Path(__file__).resolve().parent
assert os.environ['SLURM_JOB_ID']=='6825983'
F=W/'source_and_test_freeze.json';freeze=json.loads(F.read_bytes())
def verify():
 for n,r in freeze['files'].items():
  assert hashlib.sha256((W/n).read_bytes()).hexdigest()==r['sha256']
  assert (W/n).stat().st_size==r['size_bytes']
verify()
spec=importlib.util.spec_from_file_location('held_failed_H_observer_tests',W/'test_observe_exact_prior_h100_exits.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
names=['test_absent_roles_keep_real_current_observer_identity_and_utc','test_present_same_birth_is_not_an_exit','test_reused_own_birth_is_reported_without_claiming_absence','test_source_contains_no_process_actions_or_global_enumeration']
suite=unittest.TestSuite(module.PriorExitObserverTests(n)for n in names)
started=time.monotonic();stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);verify()
log=stream.getvalue().encode()
with (W/'author_focused_checks.raw').open('xb')as f:f.write(log);f.flush();os.fsync(f.fileno())
record=dict(kind='failed_h100_exit_observer_author_focused_fixture_check',passed=result.wasSuccessful(),actual_job_id=os.environ['SLURM_JOB_ID'],actual_step_id=os.environ['SLURM_STEP_ID'],tests_run=result.testsRun,seconds=time.monotonic()-started,source_freeze={'path':str(F),'sha256':hashlib.sha256(F.read_bytes()).hexdigest()},methods=names,actual_target_proc_or_remote_access=False,actual_own_proc_read=False,actual_process_actions=False,scope='Four private synthetic proc filesystem fixtures only. No actual target or local live PID is read.',signals_GPU_standing_lock_operations=0,log_sha256=hashlib.sha256(log).hexdigest())
out=W/'author_focused_checks.json';raw=(json.dumps(record,indent=2,sort_keys=True)+'\n').encode()
with out.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
fd=os.open(W,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps({'path':str(out),'sha256':hashlib.sha256(raw).hexdigest(),'passed':result.wasSuccessful(),'step':record['actual_step_id'],'seconds':record['seconds']}))
raise SystemExit(not result.wasSuccessful())

