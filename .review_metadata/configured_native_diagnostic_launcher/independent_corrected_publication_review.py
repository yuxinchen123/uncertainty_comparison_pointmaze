"""Independently verify the narrow local publication gate, without any network or scheduler action."""
import ast,contextlib,hashlib,importlib.util,io,json,os,pathlib,time,unittest
from unittest import mock
D=pathlib.Path(__file__).resolve().parent
F=D/'author_source_freeze_3.json'
freeze=json.loads(F.read_text())
def verify():
    """Keep the exact corrected source, schema and tests held through this review."""
    assert hashlib.sha256(F.read_bytes()).hexdigest()=='8958f74e258114bc4c833b0e601a6af1f444ba826ad2effd7b4deacb467a386c'
    for name,row in freeze['files'].items():
        path=D/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],name
        assert oct(path.stat().st_mode & 0o777)==row['mode'],name
verify()
old=ast.parse((D/'independent_preserved_local_only_publication_rejection/launch_diagnostic_once.py').read_text())
new=ast.parse((D/'launch_diagnostic_once.py').read_text())
old_functions={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)}
new_functions={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
assert set(new_functions)-set(old_functions)=={'require_published_commit'}
unchanged=[]
for name,previous in old_functions.items():
    current=new_functions[name]
    if name=='check_publication':
        additions=[n for n in current.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='require_published_commit']
        assert len(additions)==1
        current.body=[n for n in current.body if n not in additions]
    assert ast.dump(previous,include_attributes=False)==ast.dump(current,include_attributes=False),name
    unchanged.append(name)
allowed_commands={'config','rev-parse','merge-base'}
source=(D/'launch_diagnostic_once.py').read_text()
assert 'fetch' not in [n.value for n in ast.walk(new_functions['require_published_commit']) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
assert 'git@github.com:yuxinchen123/uncertainty_comparison_pointmaze.git' in source
spec=importlib.util.spec_from_file_location('independent_corrected_configured_launcher_tests',D/'test_launch_diagnostic_once.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
names=['test_origin_tracking_included_commit_allows_metadata_check_without_network','test_absent_origin_refuses_publication',
 'test_absent_origin_tracking_refuses_publication','test_local_only_new_commit_refuses_until_included_in_exact_tracking_history',
 'test_stale_origin_tracking_history_refuses_new_publication','test_different_origin_or_branch_mapping_refuses_publication',
 'test_default_publication_check_has_no_scheduler_action','test_release_receipt_precedes_atomic_sidecar_and_matches_actual_runtime_reader']
started=time.monotonic();stream=io.StringIO();suite=unittest.TestSuite(module.LauncherTests(name) for name in names)
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
(D/'independent_corrected_publication_tests_stdout.raw').write_text(stream.getvalue())
# Independently repeat the exact earlier empty-origin condition; it must now refuse before any scheduler call.
fixture=module.LauncherTests('runTest');fixture.setUp()
try:
    fixture.git('config','--remove-section','remote.origin')
    fixture.git('update-ref','-d',fixture.module.PUBLICATION_REF)
    refused=False
    with mock.patch.object(fixture.module,'scheduler',side_effect=AssertionError('no scheduler query or action')):
        try:fixture.module.main(['--publication-commit',fixture.publication])
        except ValueError:refused=True
    assert refused
finally:fixture.doCleanups()
verify()
record={'kind':'independent_corrected_configured_native_launcher_source_review','slurm_job_id':os.environ['SLURM_JOB_ID'],
 'slurm_step_id':os.environ['SLURM_STEP_ID'],'approved':result.wasSuccessful(),'passed':result.wasSuccessful(),
 'source_freeze':{'path':str(F),'sha256':hashlib.sha256(F.read_bytes()).hexdigest()},'tests_run':result.testsRun,
 'errors':len(result.errors),'failures':len(result.failures),'elapsed_seconds':time.monotonic()-started,
 'existing_function_ASTs_unchanged_except_one_publication_call':unchanged,'initial_exact_missing_origin_reproduction_now_refused':refused,
 'earlier_eight_bookkeeping_checks_reused':'6821437.115','all_action_and_ledger_functions_unchanged':True,
 'actual_network_operations':0,'actual_scheduler_actions':0,'actual_campaign_binding_or_unit_mutation':False,
 'actual_prepared_packet_review_still_required':True,'formal_profile_admission':False}
path=D/'independent_corrected_launcher_source_review.json';path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record));print('REVIEW_SHA256='+hashlib.sha256(path.read_bytes()).hexdigest());raise SystemExit(not result.wasSuccessful())
