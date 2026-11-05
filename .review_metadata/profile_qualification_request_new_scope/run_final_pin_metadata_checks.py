"""Check the real published runtime pin and only the added H100 transport literal boundary."""
import argparse
import ctypes  # Own-process standard-library bootstrap before strict metadata guard.
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
RUNTIME='2aa6df92f7d9da20e2fbc1d15a20df16ba10ba1e'
PUBLICATION='cfd5fb82e06cc57c08c56b05d4105390570066f7'
PREFIX='10_jax_exploration_platform/scripts/atari_pixel_campaign/'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def publish(path,value):
    """Write only small actual CPU metadata, without replacing an earlier check."""
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
def audit(event,args):
    """Refuse numerical, GPU, network, native-library and non-Git process work."""
    if event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy'}:raise ValueError('no numerical imports')
    if event in ('ctypes.dlopen','ctypes.dlsym','os.system','os.exec','os.fork') or event.startswith('socket.') and event!='socket.gethostname':raise ValueError('no native, network or nonmetadata process')
    if event=='subprocess.Popen' and (not isinstance(args[1],(list,tuple)) or args[1][0]!='git'):raise ValueError('only read-only source Git is allowed')
    if event=='open' and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith('/dev/') and os.fsdecode(args[0])!='/dev/null':raise ValueError('no device access')
def load(path,name):
    """Load only explicit local metadata source bytes."""
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

parser=argparse.ArgumentParser();parser.add_argument('--helper-root',type=Path,required=True);args=parser.parse_args()
job=os.environ.get('SLURM_JOB_ID');step=os.environ.get('SLURM_STEP_ID');os.environ['GIT_OPTIONAL_LOCKS']='0'
assert job=='6825983' and step and step.isdecimal()
cgroup=Path('/proc/self/cgroup').read_text();assert '/job_'+job+'/step_'+step+'/' in cgroup
paths=[W/'.review_metadata/profile_target_operational_preparation'/name for name in
    ('derive_one_copy_qualification_request.py','derive_two_copy_qualification_request.py','start_direct_profile_qualification_once.py')]+[D/'test_runtime_commit_and_closure.py']
before={str(path):{'sha256':sha(path),'bytes':path.stat().st_size,'mode':path.stat().st_mode&0o777} for path in paths}
publish(D/'final_source_freeze.json',{'kind':'profile_qualification_deriver_scope_source_freeze','files':before,
    'actual_runtime_commit':RUNTIME,'actual_runtime_publication':PUBLICATION,'scientific_credit':0,'admission_granted':False})
sys.addaudithook(audit);started=time.monotonic()
subprocess.check_call(['git','-C',str(W),'merge-base','--is-ancestor',RUNTIME,'refs/remotes/origin/merge-atari-cleanrl'])
assert subprocess.check_output(['git','-C',str(args.helper_root),'rev-parse','HEAD'],text=True).strip()==RUNTIME
assert subprocess.check_output(['git','-C',str(args.helper_root),'status','--porcelain','--untracked-files=all'],text=True)==''
modules=[load(path,'actual_operational_metadata_'+str(index)) for index,path in enumerate(paths[:3])]
assert all(module.RUNTIME==RUNTIME for module in modules)
q=modules[0].runtime(args.helper_root)
profile=q.adjacent({'helpers':{'files':{PREFIX+'profile_selected.py':sha(args.helper_root/PREFIX/'profile_selected.py')}}},'profile_selected')
receipts=[]
for configured in (False,True):
    receipt=profile.helper_receipt(configured=configured,direct=not configured,owned_processes=configured,
        direct_profile=not configured,configured_native=configured,configured_prerequisites=configured,profile_target=True)
    assert receipt['source_commit']==RUNTIME and receipt['source_root']==str(args.helper_root.resolve())
    assert len(receipt['files'])==(36 if configured else 45) and PREFIX+'profile_target_tool_scope.py' in receipt['files']
    receipts.append({'hardware_class':'jaguar03_rtx_a4500' if configured else 'h100','helpers':receipt})
tests=load(paths[-1],'actual_affected_transport_metadata_test');output=io.StringIO()
result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.TestSuite([
    tests.RuntimeSourceAndClosureTests('test_transport_changes_only_runtime_pin_and_h100_closure_count')]))
assert before=={str(path):{'sha256':sha(path),'bytes':path.stat().st_size,'mode':path.stat().st_mode&0o777} for path in paths}
with (D/'final_pin_checks.stdout.raw').open('x') as stream:stream.write(output.getvalue());stream.flush();os.fsync(stream.fileno())
record={'kind':'profile_qualification_actual_runtime_pin_metadata_checks','passed':result.wasSuccessful(),'tests_run':result.testsRun,
    'failures':len(result.failures),'errors':len(result.errors),'seconds':time.monotonic()-started,
    'job_id':job,'step_id':step,'pid':os.getpid(),'uid':os.getuid(),'hostname':socket.gethostname(),'cgroup':cgroup,
    'source_freeze':{'path':str(D/'final_source_freeze.json'),'sha256':sha(D/'final_source_freeze.json')},
    'actual_runtime_commit':RUNTIME,'actual_runtime_publication':PUBLICATION,'actual_helper_receipts':receipts,
    'previous_four_methods_reused':{'path':str(D/'author_checks_before_actual_runtime_pin.json'),'sha256':sha(D/'author_checks_before_actual_runtime_pin.json')},
    'canonical_preparation_or_actual_packet_created':False,'AA_launcher_changed':False,'H100_one_copy_transport_guard_unchanged':True,
    'bootstrap':'own-process standard-library ctypes before strict metadata guard; no subsequent native-library, numerical or GPU work',
    'scientific_credit':0,'target_qualification':False,'admission_granted':False}
publish(D/'final_pin_checks.json',record)
print(output.getvalue(),flush=True);print(json.dumps(record,sort_keys=True),flush=True)
sys.exit(0 if result.wasSuccessful() else 1)
