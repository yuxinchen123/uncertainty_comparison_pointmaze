"""Focused independent controller-size boundary; no actual payload is opened."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
ROOT=Path(__file__).resolve().parents[2]
M=Path(__file__).resolve().parent
STEP=os.environ['SLURM_STEP_ID']
assert os.environ['SLURM_JOB_ID']=='6783003' and STEP.isdecimal() and os.getresuid()==(2618919,)*3
assert Path('/proc/self/cgroup').read_text().splitlines()==['0::/system.slice/slurmstepd.scope/job_6783003/step_'+STEP+'/user/task_0']
F=M/'narrow_frozen_source_for_independent_review.json'
assert hashlib.sha256(F.read_bytes()).hexdigest()=='34edb1122b5f25dbed1740eda69691b56bfe6de991fdfdf1aa651b5b336cc53b'
frozen=json.loads(F.read_bytes())
def source_matches():
    return all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==v['sha256'] and oct((ROOT/n).stat().st_mode&0o777)==v['mode'] for n,v in frozen['files'].items())
assert source_matches()
START=time.monotonic()
def audit(event,args):
    if event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform'}:raise RuntimeError('numerical import refused')
    if event in {'subprocess.Popen','ctypes.dlopen','ctypes.dlsym','socket.connect','socket.bind','socket.getaddrinfo','os.system'}:raise RuntimeError('native/process/network work refused')
    if event=='open' and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith(('/dev/','/tmp/sl5nw_serval02_card1.lock')):raise RuntimeError('device/standing lock refused')
sys.addaudithook(audit)
SCRIPTS=ROOT/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
spec=importlib.util.spec_from_file_location('independent_controller_scope_author',SCRIPTS/'tests/test_accepted_evidence_snapshot_reference_scope.py')
author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
S=author.snapshot
OBS=[]
class ActualSizeBoundary(unittest.TestCase):
    def test_typed_wrong_recorded_controller_size_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary,S.ValidationFiles(30) as files:
            unit=Path(temporary)
            outputs={'schema_version':1,'kind':'actual_direct_child_output_hashes_after_wait'}
            for stream in ('stdout','stderr'):
                path=unit/('direct_controller_'+stream+'.raw');raw=stream.encode();path.write_bytes(raw)
                outputs[stream]={'path':str(path),'sha256':S.digest(raw),'size_bytes':len(raw)+(1 if stream=='stdout' else 0)}
            authority=unit/'direct_controller_output_hashes.json';authority.write_bytes(S.encoded(outputs))
            ref={'path':str(authority),'sha256':S.digest(authority.read_bytes())}
            inventory=S.AcceptedInventory(files);inventory.accepted_units[unit]={'plan':'explicit synthetic fixture'}
            inventory.add(ref)
            accepted=False;err=None;proof=None
            try:
                proof=inventory.hash_complete();accepted=True
            except ValueError as error:err=str(error)
            OBS.append({'case':'controller_wrong_typed_size','declared_stdout_size':outputs['stdout']['size_bytes'],
                'actual_stdout_size':(unit/'direct_controller_stdout.raw').stat().st_size,
                'complete_snapshot_accepted':accepted,'error':err,'proof':proof})
            self.assertFalse(accepted,'incorrect controller size_bytes was silently discarded before a complete hash proof')
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(author.ExactReferenceScopeTests),unittest.defaultTestLoader.loadTestsFromTestCase(ActualSizeBoundary)])
out=io.StringIO();result=unittest.TextTestRunner(stream=out,verbosity=2).run(suite)
assert source_matches()
receipt={'schema_version':1,'kind':'independent_narrow_reference_scope_size_review','approved':result.wasSuccessful(),'passed':result.wasSuccessful(),
    'job_id':'6783003','step_id':STEP,'checks':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'elapsed_seconds':time.monotonic()-START,
    'freeze':{'path':str(F),'sha256':hashlib.sha256(F.read_bytes()).hexdigest()},'all18frozenbytes_modes_unchanged':True,'observations':OBS,
    'payload_reads':0,'actual_full_snapshot':False,'native_GPU_tool_operations':0,'scientific_credit':0,'worker_admission':False}
for suffix,raw in [('.json',(json.dumps(receipt,indent=2,sort_keys=True)+'\n').encode()),('.stdout.raw',out.getvalue().encode())]:
    with (M/('independent_size_'+STEP+suffix)).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
print(json.dumps(receipt,sort_keys=True));print(out.getvalue(),file=sys.stderr)
raise SystemExit(0 if result.wasSuccessful() else 1)

