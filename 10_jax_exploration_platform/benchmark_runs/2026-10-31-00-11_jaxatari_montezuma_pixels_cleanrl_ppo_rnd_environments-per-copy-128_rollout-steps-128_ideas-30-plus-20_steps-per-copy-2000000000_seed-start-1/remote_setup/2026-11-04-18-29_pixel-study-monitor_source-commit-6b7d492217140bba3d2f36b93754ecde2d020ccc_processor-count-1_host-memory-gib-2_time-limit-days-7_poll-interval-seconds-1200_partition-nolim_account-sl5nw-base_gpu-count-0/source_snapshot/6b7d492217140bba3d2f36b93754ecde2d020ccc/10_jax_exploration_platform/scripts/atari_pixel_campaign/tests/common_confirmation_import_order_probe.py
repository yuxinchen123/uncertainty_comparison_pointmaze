"""Check real saved correctness in a child before the unchanged import-order gate."""
import argparse
import ast
import builtins
import ctypes
import fcntl
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import types
from unittest.mock import patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scripts', type=Path, required=True)
parser.add_argument('--run-dir', type=Path, required=True)
parser.add_argument('--copies', type=int, choices=(1, 2), required=True)
arguments = parser.parse_args()
assert os.environ.get('SLURM_JOB_ID') == '6783003' and os.environ.get('SLURM_STEP_ID', '').isdigit()
SCRIPTS = arguments.scripts.absolute()
RUN = arguments.run_dir.absolute()
name = '2026-11-01-15-09_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_jaguar03_copies-'+str(arguments.copies)+'_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1'
actual = RUN/'data'/name
plan = json.loads((actual/'measurement_plan.json').read_bytes())
review = RUN/'remote_setup/2026-11-01-17-44-40_common-a4500-actual-correctness_independent-audit_jobs-6782946-6782947_copies-1-2_gpu-count-0'/('copies_'+str(arguments.copies)+'_independent_review.json')
plan['mode'] = 'confirmation'
plan['unit_id'] = 'synthetic_common_confirmation_import_order_copies_'+str(arguments.copies)
plan['configured_isolation']['owned_inputs_path'] = str(RUN/'data'/plan['unit_id']/'owned_allocation_inputs.json')
plan['helpers'] = {'source_root':str(SCRIPTS.parents[2]), 'source_commit':'6066d9ea35fbc9c045855144550971c2ac5895f9',
    'files':{relative:hashlib.sha256((SCRIPTS/Path(relative).name).read_bytes()).hexdigest() for relative in plan['helpers']['files']}}
relative = '10_jax_exploration_platform/scripts/atari_pixel_campaign/common_confirmation.py'
plan['helpers']['files'][relative] = hashlib.sha256((SCRIPTS/'common_confirmation.py').read_bytes()).hexdigest()
plan['common_correctness_inputs'] = {'result':{'path':str(actual/'result.json'),'sha256':hashlib.sha256((actual/'result.json').read_bytes()).hexdigest()},
    'review':{'path':str(review),'sha256':hashlib.sha256(review.read_bytes()).hexdigest()}}
blocked = []
NUMERICAL = {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy','exploration_platform'}

class NoNumerical(importlib.abc.MetaPathFinder):
    """Forbid every real numerical import while allowing declared synthetic objects later."""
    def find_spec(self, name, path=None, target=None):
        """Stop a real package loader before it reads any numerical implementation."""
        if name.split('.')[0] in NUMERICAL:
            blocked.append(name)
            raise RuntimeError('real numerical import forbidden')

sys.meta_path.insert(0, NoNumerical())
ctypes.CDLL = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('real library load forbidden'))
real_popen = subprocess.Popen
with tempfile.TemporaryDirectory() as temporary:
    stage = Path(temporary)/'code'; stage.mkdir()
    for relative in plan['helpers']['files']:
        shutil.copyfile(SCRIPTS/Path(relative).name, stage/Path(relative).name)
    class LocalMetadataOnly(real_popen):
        """Allow the exact isolated metadata child and read-only local Git operations only."""
        def __init__(self, command, *args, **kwargs):
            """Reject scheduler, network, device and unrelated subprocess commands."""
            allowed = command == ['/usr/bin/python3','-I','-S','-B',str(stage/'common_confirmation.py'),'--plan-stdin']
            allowed |= isinstance(command, list) and len(command)>3 and command[:2]==['git','-C'] and Path(command[2]).is_absolute() and command[3] in {'show','cat-file','rev-parse','status','symbolic-ref','ls-tree','merge-base'}
            assert allowed, command
            super().__init__(command, *args, **kwargs)
    subprocess.Popen = LocalMetadataOnly
    sys.addaudithook(lambda event, values: (_ for _ in ()).throw(RuntimeError('network forbidden')) if event in {'socket.connect','socket.bind','socket.getaddrinfo'} else None)
    sys.path.insert(0, str(stage))
    specification=importlib.util.spec_from_file_location('cold_common_producer',stage/'measure_selected.py')
    producer=importlib.util.module_from_spec(specification);specification.loader.exec_module(producer)
    prerequisite=producer.common_confirmation_support(plan)
    before_path=list(sys.path)
    assert 'ideas' not in sys.modules and 'controller' not in sys.modules
    # The only large reads run while the campaign's existing own metadata lock is held.
    lock = RUN/'queue/cpu_monitor_large_artifact_read.lock'
    with lock.open('r+') as stream:
        observed=os.fstat(stream.fileno());assert stat.S_ISREG(observed.st_mode) and observed.st_uid==os.getuid()
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        proof=prerequisite.request(plan)
    assert before_path==sys.path and 'ideas' not in sys.modules and 'controller' not in sys.modules and not NUMERICAL.intersection(sys.modules)
    # Use exactly the production numerical import/check function and stop at its first JAX setting.
    class AfterSourceGate(RuntimeError):
        """Mark the point after real source-path checks and before any numerical work."""
    fake_jax=types.SimpleNamespace(config=types.SimpleNamespace(update=lambda *a,**k:(_ for _ in ()).throw(AfterSourceGate())))
    package=Path(temporary)/'jaxtari';package.mkdir();(package/'__init__.py').write_text('# explicit synthetic package path fixture\n')
    fake_jaxtari=types.ModuleType('jaxtari');fake_jaxtari.__file__=str(package/'__init__.py')
    synthetic_imports=[];real_import=builtins.__import__
    def synthetic(name, globals=None, locals=None, fromlist=(), level=0):
        """Replace only numerical imports; keep actual paired/ideas imports and source checks."""
        if name=='jax':synthetic_imports.append(name);return fake_jax
        if name=='jaxtari':synthetic_imports.append(name);sys.modules['jaxtari']=fake_jaxtari;return fake_jaxtari
        if name in {'exploration_platform.atari_pixels.config','exploration_platform.atari_pixels.environment','exploration_platform.atari_pixels.training'}:
            synthetic_imports.append(name);return types.SimpleNamespace(PixelPPOConfig=object,AtariPixelEnvironment=object,PixelPPORND=object)
        return real_import(name,globals,locals,fromlist,level)
    reached=False
    with patch.object(builtins,'__import__',side_effect=synthetic),patch.object(producer.metadata,'distribution',return_value=types.SimpleNamespace(locate_file=lambda path:Path(temporary)/path)):
        try:producer._compute(actual,plan,{}, {})
        except AfterSourceGate:reached=True
    assert reached and not blocked
    root=Path(plan['source_root'])/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
    assert Path(sys.modules['paired_benchmark'].__file__).resolve()==root/'paired_benchmark.py'
    assert Path(sys.modules['ideas'].__file__).resolve()==root/'ideas.py' and 'controller' not in sys.modules
    print(json.dumps({'kind':'cold_actual_common_correctness_child_then_unchanged_numerical_source_gate','passed':True,
        'copies':arguments.copies,'review_job_id':'6783003','review_step':os.environ['SLURM_STEP_ID'],
        'synthetic_outer_plan_not_a_prepared_or_admitted_unit':True,'actual_prior_correctness_consumer_passed':True,
        'actual_prior_inputs':plan['common_correctness_inputs'],'actual_prerequisite_child_proof':proof,
        'parent_import_paths_unchanged_by_metadata_child':True,'parent_ideas_and_controller_absent_after_metadata_child':True,
        'staged_helper_count':len(plan['helpers']['files']),'actual_paired_path':sys.modules['paired_benchmark'].__file__,
        'actual_ideas_path':sys.modules['ideas'].__file__,'synthetic_numerical_imports':synthetic_imports,
        'real_numerical_import_attempts':blocked,'actual_gpu_operations':0,'actual_job_actions':0,'scientific_credit':0,'admission_granted':False}),flush=True)
