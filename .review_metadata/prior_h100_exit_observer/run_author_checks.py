"""Check a held read-only exit observer using private proc fixtures and own CPU proc only."""
import hashlib,importlib.util,json,os,pathlib,time,unittest
W=pathlib.Path(__file__).resolve().parent
F=W/'source_and_test_freeze.json';f=json.loads(F.read_text())
def verify():
    """Require held source and test bytes before and after the bounded metadata fixtures."""
    for n,r in f['files'].items():
        assert hashlib.sha256((W/n).read_bytes()).hexdigest()==r['sha256'],n
        assert oct((W/n).stat().st_mode & 0o777)==r['mode'],n
verify();print('ACTUAL_CPU_STEP='+os.environ['SLURM_STEP_ID'],flush=True)
s=importlib.util.spec_from_file_location('held_prior_exit_observer_tests',W/'test_observe_exact_prior_h100_exits.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
started=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m));verify()
record={'kind':'prior_exit_observer_author_cpu_fixtures','slurm_job_id':os.environ['SLURM_JOB_ID'],'slurm_step_id':os.environ['SLURM_STEP_ID'],'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'passed':result.wasSuccessful(),'seconds':time.monotonic()-started,'freeze':{'path':str(F),'sha256':hashlib.sha256(F.read_bytes()).hexdigest()},'target_proc_namespace_accessed':False,'actual_remote_or_target_invocation':False,'signals_or_gpu_operations':0}
p=W/('author_cpu_'+os.environ['SLURM_STEP_ID']+'.json');p.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));raise SystemExit(not result.wasSuccessful())
