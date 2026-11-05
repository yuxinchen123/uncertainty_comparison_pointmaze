"""Run the three affected source/metadata checks on the exact owned CPU allocation."""
import argparse
import ctypes  # Standard-library own-process bootstrap before strict metadata guard.
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest

D=Path(__file__).resolve().parent;W=D.parents[1]
RUNTIME='4bb3a61cb92fe48ab5ba68fb52f70b47736a1174'
PUBLICATION='ecced8f39c09bddff1e8b275a3a5640df4c41351'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def publish(path,value):
    """Fsync only this check's bounded text records without replacing an earlier run."""
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
def audit(event,args):
    """Refuse numerical, GPU, network, native-library and non-Git process operations."""
    if event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy'}:raise ValueError('no numerical imports')
    if event in ('ctypes.dlopen','ctypes.dlsym','os.system','os.exec','os.fork') or event.startswith('socket.') and event!='socket.gethostname':raise ValueError('no native, network or nonmetadata process')
    if event=='subprocess.Popen' and (not isinstance(args[1],(list,tuple)) or args[1][0]!='git'):raise ValueError('only read-only source Git is allowed')
    if event=='open' and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith('/dev/') and os.fsdecode(args[0])!='/dev/null':raise ValueError('no device access')

parser=argparse.ArgumentParser();parser.add_argument('--helper-root',type=Path,required=True);arguments=parser.parse_args()
job=os.environ.get('SLURM_JOB_ID');step=os.environ.get('SLURM_STEP_ID');os.environ['GIT_OPTIONAL_LOCKS']='0'
assert job=='6825983' and step and step.isdecimal()
cgroup=Path('/proc/self/cgroup').read_text();assert '/job_'+job+'/step_'+step+'/' in cgroup
paths=[W/'.review_metadata/profile_target_operational_preparation'/name for name in
    ('derive_one_copy_qualification_request.py','derive_two_copy_qualification_request.py','start_direct_profile_qualification_once.py')]+[D/'test_final_runtime_pin.py']
before={str(path):{'sha256':sha(path),'bytes':path.stat().st_size,'mode':path.stat().st_mode&0o777} for path in paths}
publish(D/'source_freeze.json',{'kind':'profile_qualification_final_runtime_pin_source_freeze','files':before,
    'actual_runtime_commit':RUNTIME,'actual_runtime_publication':PUBLICATION,'scientific_credit':0,'admission_granted':False})
spec=importlib.util.spec_from_file_location('actual_final_runtime_pin_checks',paths[-1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.FinalRuntimePinTests.helper_root=arguments.helper_root
sys.addaudithook(audit);started=time.monotonic()
subprocess.check_call(['git','-C',str(W),'merge-base','--is-ancestor',RUNTIME,'refs/remotes/origin/merge-atari-cleanrl'])
assert subprocess.check_output(['git','-C',str(arguments.helper_root),'rev-parse','HEAD'],text=True).strip()==RUNTIME
assert subprocess.check_output(['git','-C',str(arguments.helper_root),'status','--porcelain','--untracked-files=all'],text=True)==''
output=io.StringIO();result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
assert before=={str(path):{'sha256':sha(path),'bytes':path.stat().st_size,'mode':path.stat().st_mode&0o777} for path in paths}
with (D/'author_checks.stdout.raw').open('x') as stream:stream.write(output.getvalue());stream.flush();os.fsync(stream.fileno())
record={'kind':'profile_qualification_final_runtime_pin_metadata_checks','passed':result.wasSuccessful(),'tests_run':result.testsRun,
    'failures':len(result.failures),'errors':len(result.errors),'seconds':time.monotonic()-started,
    'job_id':job,'step_id':step,'pid':os.getpid(),'uid':os.getuid(),'hostname':socket.gethostname(),'cgroup':cgroup,
    'source_freeze':{'path':str(D/'source_freeze.json'),'sha256':sha(D/'source_freeze.json')},
    'actual_runtime_commit':RUNTIME,'actual_runtime_publication':PUBLICATION,'actual_helper_receipts':module.FinalRuntimePinTests.actual_receipts,
    'canonical_preparation_or_actual_packet_created':False,'AA_launcher_changed':False,'H100_one_copy_transport_guard_unchanged':True,
    'bootstrap':'own-process standard-library ctypes before strict metadata guard; no subsequent native-library, numerical or GPU work',
    'scientific_credit':0,'target_qualification':False,'admission_granted':False}
publish(D/'author_checks.json',record)
print(output.getvalue(),flush=True)
print(json.dumps({key:value for key,value in record.items() if key!='actual_helper_receipts'},sort_keys=True),flush=True)
sys.exit(0 if result.wasSuccessful() else 1)
