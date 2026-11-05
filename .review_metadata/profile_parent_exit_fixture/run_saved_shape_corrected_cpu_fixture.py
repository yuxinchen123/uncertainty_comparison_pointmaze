"""Run only synchronized positively owned CPU-process fixtures in the exact owned allocation."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
import time
import unittest

W=Path(__file__).resolve().parents[2]
D=Path(__file__).resolve().parent
S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
job=os.environ.get('SLURM_JOB_ID');step=os.environ.get('SLURM_STEP_ID')
assert job=='6825983' and step and step.isdecimal()
cgroup=Path('/proc/self/cgroup').read_text()
assert '/job_'+job+'/' in cgroup and '/step_'+step+'/' in cgroup
runtime=D.parent/'profile_parent_exit_scope/runtime_source_freeze_initial.json'
assert digest(runtime)=='e2a4b5e77e5229ef2c8107afa0e6d989c3927dcde63e2102bbd1469fbc131b35'
paths=[S/name for name in ('profile_target_tool_scope.py','profile_target_process.py','profile_target_direct_process.py',
    'direct_tool_descendants.py','nsight_process_identity.py','linux_pidfd_process_control.py',
    'tests/test_profile_target_tool_scope_parent_exit.py')]
before={str(path):{'sha256':digest(path),'bytes':path.stat().st_size,'mode':path.stat().st_mode&0o777} for path in paths}
write(D/'author_source_freeze_3.json',{'kind':'controlled_own_cpu_parent_exit_fixture_source_freeze',
    'runtime_source_freeze':{'path':str(runtime),'sha256':digest(runtime)},'files':before,
    'scientific_credit':0,'gpu_qualification':False})
spec=importlib.util.spec_from_file_location('own_cpu_parent_exit_fixture',paths[-1]);module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
output=io.StringIO();started=time.monotonic()
result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.TestSuite([module.ParentExitTests("test_new_scope_reconciles_actual_adoption_and_normal_waits")]))
elapsed=time.monotonic()-started
raw=output.getvalue()
with (D/'author_focused_checks_3.stdout.raw').open('x') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
after={str(path):{'sha256':digest(path),'bytes':path.stat().st_size,'mode':path.stat().st_mode&0o777} for path in paths}
assert before==after,'held fixture or runtime bytes changed during checks'
record={'kind':'controlled_own_cpu_parent_exit_fixture_checks','passed':result.wasSuccessful(),
    'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'elapsed_seconds':elapsed,
    'job_id':job,'step_id':step,'pid':os.getpid(),'uid':os.getuid(),'hostname':socket.gethostname(),'cgroup':cgroup,
    'source_freeze':{'path':str(D/'author_source_freeze_3.json'),'sha256':digest(D/'author_source_freeze_3.json')},
    'scope':'real own CPU children and own-process child-subreaper control; no GPU, Nsight or numerical code',
    'upstream_custody_premise':'explicitly synthetic; no target qualification or admission',
    'native_provider_bootstrap':False,'own_process_libc_prctl_bootstrap':True,
    'scientific_credit':0,'admission_granted':False}
write(D/'author_focused_checks_3.json',record)
print(raw,flush=True);print(json.dumps(record,sort_keys=True),flush=True)
sys.exit(0 if result.wasSuccessful() else 1)
