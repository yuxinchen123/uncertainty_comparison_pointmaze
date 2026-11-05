"""Review the one-block per-job command correction without repeating accepted checks."""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import time
import unittest

D=Path(__file__).resolve().parent
FREEZE=D/'aa_held_launcher_source_freeze_2_actual_job_command.json'
EXPECTED='887e0ea7477770695a9b1c2f6a426b4940331076ce8a58849ed44764c995a74a'
ARCHIVE=D/'preserved_aa_launcher_per_job_command_rejection'


def sha(path):
    """Hash only the frozen source and small prior review metadata."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def functions(path):
    """Compare function syntax across the one approved correction."""
    return {node.name:ast.dump(node,include_attributes=False) for node in ast.walk(ast.parse(path.read_bytes()))
        if isinstance(node,ast.FunctionDef)}


def main():
    """Verify the exact source change and one affected private submission-record case."""
    started=time.monotonic();assert sha(FREEZE)==EXPECTED
    freeze=json.loads(FREEZE.read_bytes())
    for path,expected in freeze['files'].items():assert sha(path)==expected
    name='submit_profile_qualification_held_once.py'
    old=(ARCHIVE/name).read_text();new=(D/name).read_text()
    before="        for name in (*STANDARD,'job.sbatch'):\n            publish(job_directory/name,ordinary_bytes(PACKET/name))\n"
    after="        for name in (*(name for name in STANDARD if name != 'command.txt'),'job.sbatch'):\n            publish(job_directory/name,ordinary_bytes(PACKET/name))\n        publish(job_directory/'command.txt',(shlex.join(command)+'\\n').encode())\n"
    assert old.count(before)==1 and new==old.replace(before,after)
    old_functions=functions(ARCHIVE/'test_submit_profile_qualification_held_once.py')
    new_functions=functions(D/'test_submit_profile_qualification_held_once.py')
    assert old_functions.keys()==new_functions.keys()
    assert {name for name in old_functions if old_functions[name]!=new_functions[name]}=={'test_all_three_fsyncs_precede_submission_receipt'}
    spec=importlib.util.spec_from_file_location('independent_corrected_aa_launcher_tests',D/'test_submit_profile_qualification_held_once.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    output=io.StringIO()
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.TestSuite([
        tests.LaunchTests('test_all_three_fsyncs_precede_submission_receipt')]))
    assert result.wasSuccessful(),output.getvalue()
    for path,expected in freeze['files'].items():assert sha(path)==expected
    prior=D/'aa_independent_initial_review.json'
    record={'schema_version':1,'kind':'independent_aa_profile_target_qualification_launcher_source_review',
        'approved':True,'passed':True,'independent_of_author':True,
        'source_freeze':{'path':str(FREEZE),'sha256':EXPECTED},
        'prior_fourteen_checks_and_unchanged_owned_inputs_schema_proof_reused':{'path':str(prior),'sha256':sha(prior)},
        'only_prior_finding_corrected':True,'per_job_command_records_actual_held_argv':True,
        'all_twelve_packet_parent_child_standard_bytes_unchanged':True,
        'three_ledger_fsyncs_precede_submission_receipt':True,'affected_methods_passed':result.testsRun,
        'actual_job_id':os.environ['SLURM_JOB_ID'],'actual_step_id':os.environ['SLURM_STEP_ID'],
        'seconds':time.monotonic()-started,'actual_prepared_packet_review_still_required':True,
        'fresh_reservation_observation_still_required':True,'actual_prepared_unit_used':False,
        'real_scheduler_calls':0,'target_operations':0,'scientific_credit':0,'admission_granted':False}
    destination=D/'aa_independent_corrected_source_review.json'
    with destination.open('x') as stream:
        json.dump(record,stream,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps({'review':{'path':str(destination),'sha256':sha(destination)},**record}));print(output.getvalue())


if __name__=='__main__':main()
