"""Review the held A4500 launcher with private metadata and a synthetic scheduler."""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import time
import unittest
from unittest.mock import patch

D = Path(__file__).resolve().parent
FREEZE = D/'aa_held_launcher_source_freeze_1.json'
EXPECTED = '1cd1546ce77fef24313fbeb8c2c62e321957b09675a0e3e4669159d7552fca2b'


def sha(path):
    """Hash only the small frozen source and actual review records."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    """Write one durable independent record without replacing prior evidence."""
    raw=value if isinstance(value,bytes) else (json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())


def main():
    """Exercise every new launcher boundary, then preserve the per-job command mismatch."""
    started=time.monotonic();assert sha(FREEZE)==EXPECTED
    freeze=json.loads(FREEZE.read_bytes())
    for path,expected in freeze['files'].items():assert sha(path)==expected
    archive=D/'preserved_aa_launcher_per_job_command_rejection';archive.mkdir()
    manifest=[]
    for path in [Path(path) for path in freeze['files']]+[FREEZE,
            D/'aa_held_launcher_author_checks_1.json',D/'aa_held_launcher_author_handoff_1.json']:
        destination=archive/path.name;shutil.copyfile(path,destination)
        manifest.append({'original':str(path),'preserved':str(destination),'sha256':sha(destination)})
    save(archive/'manifest.json',{'files':manifest,'source_freeze_sha256':EXPECTED})
    spec=importlib.util.spec_from_file_location('independent_aa_launcher_tests',D/'test_submit_profile_qualification_held_once.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    output=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(tests.LaunchTests)
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    save(D/'aa_independent_initial_tests_stdout.raw',output.getvalue().encode())
    assert result.wasSuccessful(),output.getvalue()

    # The actual unchanged saved-reader function consumes this private envelope.
    # Its upstream configuration/physical-contract gate is an explicit synthetic
    # premise here; no target qualification or live host condition is asserted.
    fixture=tests.LaunchTests('test_release_receipt_before_atomic_unchanged_envelope');fixture.setUp()
    try:
        fixture.release_binding()
        with patch.object(fixture.m,'scheduler',side_effect=fixture.scheduler):
            fixture.m.release_once(fixture.publication)
        source=D.parents[1]/'10_jax_exploration_platform/scripts/atari_pixel_campaign/uva_slurm_configured_isolation.py'
        node=next(node for node in ast.parse(source.read_bytes()).body if isinstance(node,ast.FunctionDef) and node.name=='owned_inputs')
        namespace={'Path':Path,'re':re,'require':fixture.m.require,'receipt':fixture.m.reference,
            'read_receipt':fixture.m.checked,'ordinary':lambda path:Path(path),
            'validate_plan':lambda plan:{'owned_inputs_path':str(fixture.child/'owned_allocation_inputs.json')}}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
        envelope=namespace['owned_inputs'](fixture.plan)
        assert envelope==fixture.m.document(fixture.child/'owned_allocation_inputs.json')
        job_command=(fixture.run/'slurm/jobs'/fixture.job/'command.txt').read_bytes()
        packet_command=(fixture.packet/'command.txt').read_bytes()
        actual_held=fixture.m.held_command(fixture.m.resources('reservation'))
        import shlex
        assert job_command==packet_command and shlex.split(job_command.decode())!=actual_held
        mismatch={'job_command_bytes':job_command.decode(),'packet_command_bytes':packet_command.decode(),
            'actual_recorded_held_command':actual_held,'job_script_and_receipts_were_correct':True,
            'synthetic_fixture':True}
    finally:
        fixture.doCleanups()
    for path,expected in freeze['files'].items():assert sha(path)==expected
    record={'schema_version':1,'kind':'independent_aa_profile_qualification_launcher_initial_review',
        'approved':False,'passed':False,'source_freeze':{'path':str(FREEZE),'sha256':EXPECTED},
        'preservation_manifest':{'path':str(archive/'manifest.json'),'sha256':sha(archive/'manifest.json')},
        'independent_author_methods_passed':result.testsRun,
        'unchanged_uva_owned_inputs_function_consumed_private_envelope':True,
        'upstream_validation_premise_explicitly_synthetic':True,
        'findings':[{'kind':'per_job_command_is_preparation_command','detail':mismatch}],
        'source_unchanged':True,'actual_job_id':os.environ['SLURM_JOB_ID'],'actual_step_id':os.environ['SLURM_STEP_ID'],
        'seconds':time.monotonic()-started,'real_scheduler_calls':0,'target_operations':0,'admission_granted':False}
    destination=D/'aa_independent_initial_review.json';save(destination,record)
    print(json.dumps({'review':{'path':str(destination),'sha256':sha(destination)},**record}))


if __name__=='__main__':main()
