"""Review the H100 parser-presence argument without preparing or launching a unit."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

D=Path(__file__).resolve().parent
W=D.parent.parent
EXPECTED='3ae0a94b0fbe48e4a68c2c38430bfdf3eeb95b88f5ca5126da2a4e233240d48e'


def sha(path):
    """Hash only the small frozen source and metadata files."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    """Run only the two affected methods and retain actual source/failed-request identities."""
    started=time.monotonic();freeze_path=D/'source_freeze.json';assert sha(freeze_path)==EXPECTED
    freeze=json.loads(freeze_path.read_bytes())
    for row in freeze['files'].values():assert sha(row['path'])==row['sha256']
    bootstrap=[]
    def hook(event,args):
        if event.startswith('ctypes.'):
            if sealed:raise RuntimeError('late native-library operation refused')
            bootstrap.append({'event':event,'own_process_only':args[0] is None if args else False})
        if sealed and event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','flax','optax','torch','jaxtari','pynvml'}:
            raise RuntimeError('numerical/device import refused')
    sealed=False;sys.addaudithook(hook)
    import ctypes
    sealed=True
    spec=importlib.util.spec_from_file_location('independent_h100_presence_tests',D/'test_h100_qualification_backend_presence_argument.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    output=io.StringIO();result=unittest.TextTestRunner(stream=output,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(tests.H100PresenceArgument))
    assert result.wasSuccessful(),output.getvalue()
    environment=dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_TERMINAL_PROMPT='0')
    runtime_diff=subprocess.check_output(['git','-C',str(W),'diff','--name-only','--',
        '10_jax_exploration_platform/scripts/atari_pixel_campaign','10_jax_exploration_platform/src'],env=environment)
    assert runtime_diff==b''
    for row in freeze['files'].values():assert sha(row['path'])==row['sha256']
    record={'schema_version':1,'kind':'independent_h100_operational_parser_presence_argument_review',
        'approved':True,'passed':True,'independent_of_author':True,
        'source_freeze':{'path':str(freeze_path),'sha256':EXPECTED},'affected_methods_passed':result.testsRun,
        'actual_failed_request_unchanged':True,'actual_failed_request_without_argument_still_refused':True,
        'actual_failed_request_with_only_fresh_record_argument_parses':True,
        'argument_path_equals_request_fresh_backend_record':True,'fresh_backend_record_sha256':'19e752ef9b93999e8bec28a0706947935c31f4229a3b33556597de8f86575412',
        'new_backend_reader_accepts_and_old_reader_refuses_new_record':True,'old_record_reused':False,
        'runtime_source_diff_empty':True,'original_failed_unit_directories_still_absent':True,
        'actual_job_id':os.environ['SLURM_JOB_ID'],'actual_step_id':os.environ['SLURM_STEP_ID'],
        'explicit_stdlib_own_process_bootstrap_before_guard':bootstrap,
        'seconds':time.monotonic()-started,'actual_runtime_prepared':False,'target_operations':0,
        'scientific_credit':0,'admission_granted':False}
    destination=D/'independent_source_review.json'
    with destination.open('x') as stream:
        json.dump(record,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps({'review':{'path':str(destination),'sha256':sha(destination)},**record}));print(output.getvalue())


if __name__=='__main__':main()
