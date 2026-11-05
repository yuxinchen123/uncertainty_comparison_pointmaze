"""Run only the new metadata guards in the exact owned CPU allocation, without preparing a unit."""
import ctypes  # Standard-library own-process bootstrap before strict import/process checks.
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

D=Path(__file__).resolve().parent;W=D.parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def publish(path,value):
    """Save only this fixture's small records exactly once with real returns."""
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
def audit(event,args):
    """Refuse numerical, GPU, network, native-library and non-Git process work."""
    if event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy'}:raise ValueError('no numerical imports')
    if event in ('ctypes.dlopen','ctypes.dlsym','os.system','os.exec','os.fork') or event.startswith('socket.') and event!='socket.gethostname':raise ValueError('no native, network or nonmetadata process')
    if event=='subprocess.Popen' and (not isinstance(args[1],(list,tuple)) or args[1][0]!='git'):raise ValueError('only read-only source Git is allowed')
    if event=='open' and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith('/dev/') and os.fsdecode(args[0])!='/dev/null':raise ValueError('no device access')

job=os.environ.get('SLURM_JOB_ID');step=os.environ.get('SLURM_STEP_ID');os.environ['GIT_OPTIONAL_LOCKS']='0'
assert job=='6825983' and step and step.isdecimal()
cgroup=Path('/proc/self/cgroup').read_text();assert '/job_'+job+'/step_'+step+'/' in cgroup
paths=[W/'.review_metadata/profile_target_operational_preparation'/name for name in
    ('derive_one_copy_qualification_request.py','derive_two_copy_qualification_request.py')]+[D/'test_runtime_commit_and_closure.py']
before={str(path):{'sha256':sha(path),'bytes':path.stat().st_size,'mode':path.stat().st_mode&0o777} for path in paths}
publish(D/'author_source_freeze_before_actual_runtime_pin.json',{'kind':'profile_qualification_deriver_scope_source_freeze',
    'files':before,'actual_new_runtime_commit_supplied':False,'source_held':True,'scientific_credit':0,'admission_granted':False})
spec=importlib.util.spec_from_file_location('exact_deriver_metadata_checks',paths[-1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
sys.addaudithook(audit);output=io.StringIO();started=time.monotonic()
result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
elapsed=time.monotonic()-started
assert before=={str(path):{'sha256':sha(path),'bytes':path.stat().st_size,'mode':path.stat().st_mode&0o777} for path in paths}
raw=D/'author_checks_before_actual_runtime_pin.stdout.raw'
with raw.open('x') as stream:stream.write(output.getvalue());stream.flush();os.fsync(stream.fileno())
record={'kind':'profile_qualification_deriver_scope_metadata_checks','passed':result.wasSuccessful(),'tests_run':result.testsRun,
    'errors':len(result.errors),'failures':len(result.failures),'seconds':elapsed,'job_id':job,'step_id':step,
    'pid':os.getpid(),'uid':os.getuid(),'hostname':socket.gethostname(),'cgroup':cgroup,
    'source_freeze':{'path':str(D/'author_source_freeze_before_actual_runtime_pin.json'),'sha256':sha(D/'author_source_freeze_before_actual_runtime_pin.json')},
    'actual_new_runtime_commit_supplied':False,'actual_packet_or_unit_created':False,'upstream_premises':'synthetic metadata fixtures only',
    'bootstrap':'own-process standard-library ctypes initialization before guard; no later native load or numerical/device operations',
    'scientific_credit':0,'target_qualification':False,'admission_granted':False}
publish(D/'author_checks_before_actual_runtime_pin.json',record)
print(output.getvalue(),flush=True);print(json.dumps(record,sort_keys=True),flush=True)
sys.exit(0 if result.wasSuccessful() else 1)
