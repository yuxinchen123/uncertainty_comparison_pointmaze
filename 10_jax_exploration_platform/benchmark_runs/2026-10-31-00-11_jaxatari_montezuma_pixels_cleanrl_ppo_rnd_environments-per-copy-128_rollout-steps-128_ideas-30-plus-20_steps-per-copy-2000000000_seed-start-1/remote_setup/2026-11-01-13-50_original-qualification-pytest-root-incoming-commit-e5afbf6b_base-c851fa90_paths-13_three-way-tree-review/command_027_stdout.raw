"""Run only metadata root/reader regressions on the owned CPU, preserving the actual failed GPU unit."""
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
import unittest

W=Path(__file__).resolve().parents[1]
M=W/'.review_metadata'
S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
P=Path('/tmp/not-used-on-compute')
UNIT=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/data/2026-11-01-12-36_full-original-qualification_jaguar03_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-3_policies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-1_driver-580.159.04')
BLOCKED=[]

def audit(event,args):
    """Refuse real numerical/device/library/network/cluster operations and writes to the failed unit."""
    denied=False
    if event=='import':denied=args[0].split('.')[0] in {'jax','jaxlib','numpy','scipy','torch','flax','optax','jaxtari','pynvml','cupy'}
    elif event=='ctypes.dlopen':denied=args[0] is not None
    elif event.startswith('socket.'):denied=event not in {'socket.__new__','socket.gethostname'}
    elif event=='subprocess.Popen':denied=True
    elif event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).absolute();mode,flags=args[1],args[2]
        denied=str(path).startswith('/dev/nvidia') or (path.is_relative_to(UNIT) and ((isinstance(mode,str) and any(char in mode for char in 'wax+')) or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
    elif event in {'os.rename','os.remove','os.rmdir'}:
        denied=any(isinstance(arg,(str,bytes,os.PathLike)) and Path(os.fsdecode(arg)).absolute().is_relative_to(UNIT) for arg in args[:2])
    if denied:BLOCKED.append({'event':event,'args':repr(args)[:300]});raise RuntimeError('metadata-only regression refused live action: '+event)

def hashes(paths):
    """Record every exact frozen or failed-unit byte without modifying it."""
    return {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

def main():
    """Verify the frozen source and actual failed bytes around the focused tests only."""
    assert os.environ['SLURM_JOB_ID']=='6781231' and os.environ['SLURM_STEP_ID'].isdigit()
    print('owned metadata step '+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],flush=True)
    freeze=M/'qualification_pytest_root_source_freeze.json';spec=json.loads(freeze.read_text())
    before=hashes([W/name for name in spec['files']]);assert before=={str(W/name):digest for name,digest in spec['files'].items()}
    actual_paths=sorted(path for path in UNIT.rglob('*') if path.is_file());actual_before=hashes(actual_paths)
    sys.addaudithook(audit)
    sys.path[:0]=[str(S/'tests'),str(S)]
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName('test_qualification_pytest_root'),
        unittest.defaultTestLoader.loadTestsFromName('test_configured_measurement.QualificationReaderTests')])
    log=io.StringIO();started=time.time();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    after=hashes([W/name for name in spec['files']]);actual_after=hashes(actual_paths)
    assert before==after and actual_before==actual_after and set(actual_paths)==set(path for path in UNIT.rglob('*') if path.is_file())
    assert not BLOCKED and not any(name.split('.')[0] in {'jax','jaxlib','numpy','torch','jaxtari','flax','optax'} for name in sys.modules)
    step=os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID'];(M/(step+'_qualification_pytest_root_author.log')).write_text(log.getvalue())
    value={'schema_version':1,'kind':'qualification_pytest_explicit_root_author_metadata_regressions','job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'at_utc':datetime.now(timezone.utc).isoformat(),'executed_source':{'path':str(Path(__file__)),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'freeze':{'path':str(freeze),'sha256':hashlib.sha256(freeze.read_bytes()).hexdigest()},'frozen_before':before,'frozen_after':after,'actual_failed_unit':str(UNIT),'actual_failed_files_before':actual_before,'actual_failed_files_after':actual_after,'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'duration_seconds':time.time()-started,'blocked_actions':BLOCKED,'numerical_imports':False,'new_gpu_work':False,'job_submissions':0,'actual_partial_qualification_credit':0,'actual_failed_test_fixture_work':'48 actual passed GPU test cases retain unquantified test-fixture work; no whole-job transition total is inferred. Full renderer and controlled reference warmup/update were not reached.'}
    out=M/(step+'_qualification_pytest_root_author.json')
    with out.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(log.getvalue(),flush=True);print(json.dumps({'receipt':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors)}),flush=True)
    assert result.wasSuccessful()

if __name__=='__main__':main()
