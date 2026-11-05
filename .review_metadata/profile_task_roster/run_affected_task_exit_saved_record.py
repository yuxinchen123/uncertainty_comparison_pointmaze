"""Run only the controlled own-CPU task-directory exit fixture."""
from pathlib import Path
import hashlib,importlib.util,json,os,socket,sys,time,unittest
w=Path(__file__).resolve().parents[2]
d=Path(__file__).resolve().parent
freeze=json.loads((d/'fixture_source_freeze_2_parser_argument.json').read_text())
def check():
    for name,row in freeze['files'].items():
        path=w/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],name
        assert path.stat().st_size==row['bytes'] and oct(path.stat().st_mode&0o777)==row['mode'],name
check()
assert os.environ.get('SLURM_JOB_ID')=='6825983'
job=os.environ['SLURM_JOB_ID'];step=os.environ.get('SLURM_STEP_ID')
cgroup=Path('/proc/self/cgroup').read_text()
assert '/job_'+job+'/' in cgroup and step is not None and '/step_'+step+'/' in cgroup
path=w/'10_jax_exploration_platform/scripts/atari_pixel_campaign/tests/test_profile_target_tool_scope_task_exit.py'
spec=importlib.util.spec_from_file_location('actual_task_exit_fixture',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.LiveParentTaskExitTests.evidence_directory=d/'actual_cpu_task_exit_evidence_2'
started=time.time()
result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([unittest.FunctionTestCase(lambda: module.LiveParentTaskExitTests().run_case(True), description='affected new fresh-read and actual saved-record path only; old refusal reused from step176')]))
check()
record={'kind':'actual_own_cpu_task_exit_fixture_checks','passed':result.wasSuccessful(),'methods':result.testsRun,
    'seconds':time.time()-started,'actual_job_id':job,'actual_step_id':step,'actual_host':socket.gethostname(),
    'actual_pid':os.getpid(),'actual_uid':os.getuid(),'actual_cgroup':cgroup,
    'source_freeze':{'path':str(d/'fixture_source_freeze_2_parser_argument.json'),'sha256':hashlib.sha256((d/'fixture_source_freeze_2_parser_argument.json').read_bytes()).hexdigest()},
    'native_provider_role_claimed':False,'scientific_credit':0,'admission_granted':False,
    'scope':'Real own CPU processes, stdlib pidfds and own-child subreaper only; configured-allocation custody premise is explicitly synthetic. No Nsight, GPU, numerical or target qualification.',
    'failures':[str(case)+'\n'+trace for case,trace in result.failures],
    'errors':[str(case)+'\n'+trace for case,trace in result.errors]}
output=d/'actual_task_exit_fixture_checks_2.json'
with output.open('x') as stream:json.dump(record,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps(record,sort_keys=True))
sys.exit(0 if result.wasSuccessful() else 1)
