"""Review only the corrected publication and packet-name metadata boundaries."""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

D = Path(__file__).resolve().parent
FREEZE = D / 'request_and_direct_transport_source_freeze_2.json'
EXPECTED = '094ccd75dfe0f1948d4c1085a2894b28cec0b454b9ba80d2db49f1fdfbc420ee'
ARCHIVE = D / 'preserved_request_transport_local_publication_rejection'


def sha(path):
    """Hash only the small frozen source and metadata files for this review."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def functions(path):
    """Compare actual function syntax without repeating accepted process-control fixtures."""
    return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(path.read_bytes()).body
        if isinstance(node,ast.FunctionDef)}


def main():
    """Check the narrow source changes and exercise real private Git refusal cases."""
    started = time.monotonic()
    assert sha(FREEZE) == EXPECTED
    freeze = json.loads(FREEZE.read_bytes())
    for value in freeze['files'].values():assert sha(value['path']) == value['sha256']
    old_functions = functions(ARCHIVE/'start_direct_profile_qualification_once.py')
    new_functions = functions(D/'start_direct_profile_qualification_once.py')
    assert set(new_functions)-set(old_functions) == {'published_tracking'}
    assert {name for name in old_functions if old_functions[name] != new_functions[name]} == {'verify'}
    old = (ARCHIVE/'start_direct_profile_qualification_once.py').read_text()
    new = (D/'start_direct_profile_qualification_once.py').read_text()
    old_verify = ast.get_source_segment(old,next(node for node in ast.parse(old).body
        if isinstance(node,ast.FunctionDef) and node.name=='verify'))
    new_verify = ast.get_source_segment(new,next(node for node in ast.parse(new).body
        if isinstance(node,ast.FunctionDef) and node.name=='verify'))
    assert new_verify.replace('    published_tracking(root, publication)\n','') == old_verify
    old = (ARCHIVE/'derive_one_copy_qualification_request.py').read_text()
    new = (D/'derive_one_copy_qualification_request.py').read_text()
    assert new.replace("    require(packet.name == args.profile_unit_id, 'the actual packet name must equal the profile capture unit name')\n",'') == old
    for name in ('prepare_direct_transport_configuration.py','four_context_fact_request_actual_refs.json','operational_preparation_recipe.json'):
        assert (D/name).read_bytes() == (ARCHIVE/name).read_bytes()

    specification = importlib.util.spec_from_file_location('independent_corrected_operational_tests',
        D/'test_qualification_request_and_direct_transport.py')
    tests = importlib.util.module_from_spec(specification);specification.loader.exec_module(tests)
    output = io.StringIO()
    selected = ['test_only_actual_origin_tracking_accepts_publication',
        'test_complete_synthetic_transport_checks_and_changed_plan_refuse',
        'test_request_uses_actual_component_plan_not_result_fields']
    suite = unittest.TestSuite(tests.OperationalChecks(name) for name in selected)
    result = unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    assert result.wasSuccessful(),output.getvalue()

    # All commands below address one temporary local repository; no network exists in this test.
    environment = dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_TERMINAL_PROMPT='0',
        GIT_AUTHOR_NAME='independent fixture',GIT_AUTHOR_EMAIL='fixture@example.invalid',
        GIT_COMMITTER_NAME='independent fixture',GIT_COMMITTER_EMAIL='fixture@example.invalid')
    def git(root,*arguments):
        """Create private fixture commits and tracking metadata without fetching."""
        return subprocess.run(['git','-C',str(root),*arguments],check=True,capture_output=True,
            timeout=15,env=environment).stdout
    checks = []
    def refused(root,publication,name):
        """Record refusal only when the actual corrected function raises on bad metadata."""
        try:tests.TRANSPORT.published_tracking(root,publication)
        except (ValueError,subprocess.CalledProcessError):checks.append(name);return
        raise AssertionError('incorrect publication accepted: '+name)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary);git(root,'init','-q')
        (root/'source.py').write_text('# explicit private fixture\n')
        git(root,'add','source.py');git(root,'commit','-qm','first synthetic source')
        parent = git(root,'rev-parse','HEAD').decode().strip()
        refused(root,parent,'missing_origin')
        git(root,'remote','add','origin','git@github.com:yuxinchen123/uncertainty_comparison_pointmaze.git')
        refused(root,parent,'missing_tracking_ref')
        git(root,'update-ref','refs/remotes/origin/merge-atari-cleanrl',parent)
        tests.TRANSPORT.published_tracking(root,parent);checks.append('included_parent_accepted')
        (root/'source.py').write_text('# explicit private second fixture\n')
        git(root,'add','source.py');git(root,'commit','-qm','second synthetic source')
        child = git(root,'rev-parse','HEAD').decode().strip()
        refused(root,child,'local_only_descendant_refused')
        git(root,'update-ref','refs/remotes/origin/merge-atari-cleanrl',child)
        tests.TRANSPORT.published_tracking(root,child);checks.append('fresh_tracking_tip_accepted')
        tests.TRANSPORT.published_tracking(root,parent);checks.append('tracked_ancestor_accepted')
        git(root,'config','remote.origin.url','git@example.invalid:other/repository.git')
        refused(root,child,'wrong_origin_refused')
        git(root,'config','remote.origin.url','git@github.com:yuxinchen123/uncertainty_comparison_pointmaze.git')
        git(root,'config','remote.origin.fetch','+refs/heads/other:refs/remotes/origin/other')
        refused(root,child,'wrong_fetch_mapping_refused')
        git(root,'config','remote.origin.fetch','+refs/heads/*:refs/remotes/origin/*')
        git(root,'config','--add','remote.origin.url','git@example.invalid:additional/repository.git')
        refused(root,child,'multiple_origin_urls_refused')
    for value in freeze['files'].values():assert sha(value['path']) == value['sha256']
    record = {'schema_version':1,'kind':'independent_operational_request_transport_corrected_source_review',
        'approved':True,'passed':True,'source_freeze':{'path':str(FREEZE),'sha256':EXPECTED},
        'independent_of_author':True,'scope':'Only request builder, transport configuration and H root transport; excludes the reviewer-authored fact producer and any later actual fact/output review',
        'existing_seven_source_checks_reused':{'path':str(D/'independent_request_transport_initial_review.json'),
            'sha256':sha(D/'independent_request_transport_initial_review.json')},
        'three_affected_methods_passed':result.testsRun,'actual_private_git_checks':checks,
        'all_prior_custody_functions_unchanged':True,'prior_algorithm_independent_approval_reused':True,
        'new_algorithm_independence_claimed':False,'runtime_helpers_changed':False,
        'actual_packet_review_required_before_launch':True,'actual_target_launch_permitted_by_this_source_review':False,
        'fact_produced':False,'actual_runtime_prepared':False,'target_operations':0,
        'actual_job_id':os.environ['SLURM_JOB_ID'],'actual_step_id':os.environ['SLURM_STEP_ID'],
        'seconds':time.monotonic()-started,'admission_granted':False}
    destination = D/'independent_request_transport_corrected_source_review.json'
    with destination.open('x') as stream:
        json.dump(record,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps({'review':{'path':str(destination),'sha256':sha(destination)},**record}));print(output.getvalue())


if __name__=='__main__':main()
