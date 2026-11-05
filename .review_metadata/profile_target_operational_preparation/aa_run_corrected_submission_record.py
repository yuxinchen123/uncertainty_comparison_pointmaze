"""Run one affected private metadata case on the exact owned CPU allocation."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import time
import unittest

p = Path(__file__).resolve().parent
assert os.environ.get('SLURM_JOB_ID') == '6825983' and os.environ.get('SLURM_STEP_ID','').isdecimal()
freeze = p/'aa_held_launcher_source_freeze_2_actual_job_command.json'
files = json.loads(freeze.read_text())['files']
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path,digest in files.items())
spec = importlib.util.spec_from_file_location('aa_affected_submission_record',p/'test_submit_profile_qualification_held_once.py')
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
output = io.StringIO();start = time.monotonic()
result = unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.TestSuite([
    m.LaunchTests('test_all_three_fsyncs_precede_submission_receipt')]))
elapsed = time.monotonic()-start
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path,digest in files.items())
raw = output.getvalue().encode();raw_path = p/'aa_corrected_submission_record_stdout.raw'
with raw_path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
receipt = {'kind':'profile_target_qualification_aa_launcher_corrected_author_metadata_test','passed':result.wasSuccessful(),
    'tests':result.testsRun,'elapsed_seconds':elapsed,'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],
    'hostname':socket.gethostname(),'pid':os.getpid(),'uid':os.getuid(),'source_freeze':{'path':str(freeze),
    'sha256':hashlib.sha256(freeze.read_bytes()).hexdigest()},'frozen_before_after_exact':True,
    'stdout':{'path':str(raw_path),'sha256':hashlib.sha256(raw).hexdigest()},'actual_actions':False,
    'synthetic_private_scheduler_only':True,'prior_14_case_proof_reused':True}
path = p/'aa_corrected_submission_record_author_checks.json'
with path.open('x') as stream:json.dump(receipt,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps(receipt,sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() else 1)
