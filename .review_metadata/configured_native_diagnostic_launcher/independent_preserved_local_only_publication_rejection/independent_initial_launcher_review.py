"""Review frozen one-shot bookkeeping using private Git/files and no real scheduler action."""
import contextlib,hashlib,importlib.util,io,json,os,pathlib,time,unittest
from unittest import mock
D=pathlib.Path(__file__).resolve().parent
F=D/'author_source_freeze_2.json'
freeze=json.loads(F.read_text())
def verify():
    """Require every held source/test/schema byte and mode before and after review."""
    assert hashlib.sha256(F.read_bytes()).hexdigest()=='4f4970fe9de26a128798368893570ab142344eb31e86e53c45add566851f793d'
    for name,row in freeze['files'].items():
        p=D/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],name
        assert oct(p.stat().st_mode & 0o777)==row['mode'],name
verify()
spec=importlib.util.spec_from_file_location('independent_frozen_configured_launcher_tests',D/'test_launch_diagnostic_once.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
selected=['test_default_publication_check_has_no_scheduler_action','test_changed_initial_source_refuses_before_intent',
 'test_all_three_ledger_fsyncs_precede_submission_receipt','test_owned_membership_checked_before_query_even_with_decimal_foreign_id',
 'test_release_receipt_precedes_atomic_sidecar_and_matches_actual_runtime_reader','test_failed_release_retains_attempt_without_false_ownership',
 'test_late_binding_has_only_actual_source_and_plan_and_cannot_be_reused','test_exact_held_owner_and_resources_refuse_substitution']
started=time.monotonic();suite=unittest.TestSuite(module.LauncherTests(name) for name in selected)
stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
(D/'independent_launcher_selected_tests_stdout.raw').write_text(stream.getvalue())
fixture=module.LauncherTests('runTest');fixture.setUp()
try:
    refs=fixture.git('for-each-ref','--format=%(refname)','refs/remotes').decode().splitlines()
    remotes=fixture.git('remote').decode().splitlines()
    assert refs==[] and remotes==[]
    output=io.StringIO()
    with mock.patch.object(fixture.module,'scheduler',side_effect=AssertionError('no scheduler operation during independent review')),contextlib.redirect_stdout(output):
        fixture.module.main(['--publication-commit',fixture.publication])
    observed=json.loads(output.getvalue())
    assert observed['passed'] is True and observed['action_performed'] is False
    publication={'private_local_commit':fixture.publication,'configured_remotes':remotes,'remote_tracking_refs':refs,
        'local_only_commit_accepted_as_publication':True,'actual_scheduler_actions':0,'actual_campaign_files_changed':False}
finally:fixture.doCleanups()
verify()
record={'kind':'independent_configured_native_launcher_initial_review','slurm_job_id':os.environ['SLURM_JOB_ID'],
 'slurm_step_id':os.environ['SLURM_STEP_ID'],'source_freeze':{'path':str(F),'sha256':hashlib.sha256(F.read_bytes()).hexdigest()},
 'selected_author_tests':selected,'tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),
 'selected_tests_passed':result.wasSuccessful(),'publication_reproduction':publication,'elapsed_seconds':time.monotonic()-started,
 'approved':False,'reason':'The publication check accepts an exact local-only commit even when the repository has no configured remote or remote-tracking history. Exact bytes in a local commit alone do not prove publication.',
 'actual_scheduler_actions':0,'target_or_profile_admission':False}
path=D/'independent_initial_launcher_publication_review.json';path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record));print('REVIEW_SHA256='+hashlib.sha256(path.read_bytes()).hexdigest())
raise SystemExit(not result.wasSuccessful())
