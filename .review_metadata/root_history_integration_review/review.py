"""Independently check the frozen future-profile integration on owned CPU only."""
import ast,hashlib,importlib.util,json,os,sys,time,unittest
from pathlib import Path
w=Path(__file__).resolve().parents[2];meta=w/'.review_metadata/future_profile_history_integration';source=w/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
freeze_path=meta/'source_freeze.json';assert hashlib.sha256(freeze_path.read_bytes()).hexdigest()=='562bc51e25645f31113602c082280bff39882ab51bc1c8b593ca2b6363a42d79';freeze=json.loads(freeze_path.read_bytes());started=time.monotonic()
def verify():
    """Rehash every sealed source, metadata artifact and protected numerical file."""
    total=0
    for field in ('source_and_tests','artifacts'):
        for row in freeze[field].values():
            p=Path(row['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'];total+=1
    for name,expected in freeze['protected_13_numerical_files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected;total+=1
    return total
checked=verify();sys.path.insert(0,str(source/'tests'))
spec=importlib.util.spec_from_file_location('root_review_future_profile_tests',source/'tests/test_direct_profile_prerequisites.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(module.ProfileHistoryIntegrationTests))
for name in ('test_helper_union_includes_backend_and_separate_profile_observers','test_explicit_old28_profile_still_reaches_original_raw_gate','test_new29_history_pointer_cannot_open_a_result_path','test_new_and_old_registered_profiles_cannot_bypass_current_admission','test_actual_context_failure_precedes_child_creation_and_scientific_reads','test_parent_rejects_false_child_proof_even_with_zero_return','test_saved_old_schema_cannot_bypass_current_metadata_closure'):
    suite.addTest(module.DirectProfileMetadataTests(name))
result=unittest.TextTestRunner(verbosity=2).run(suite);assert result.wasSuccessful()
# Independently use the genuine metadata with no synthetic authority replacement.
peer=json.loads((w/'.review_metadata/independent_full_profile_adapter_review/independent_dependency_6821437_50.json').read_bytes());ref=peer['profile_pointer']['common_plan'];raw=Path(ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];common=json.loads(raw);plan={**common,'kind':'selected_configuration_profile','unit_id':'root_nonoperational_authority_review'}
admission,provenance=module.profile.original_admission_metadata(plan,common,live=True);assert hashlib.sha256(admission).hexdigest()==module.profile.ORIGINAL_ADMISSION_SHA256;assert provenance['profile_worker_admitted'] is False
# Compare the two exact source-authority functions with published69f syntax.
published=Path(freeze['published_69f_binding']['path']).read_text();current=(source/'serval02_card1_binding.py').read_text()
def function(text,name):
    """Extract only the selected function's complete syntax."""
    return ast.dump(next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef) and x.name==name),include_attributes=False)
for name in ('current_admission_claim','check_live_admission'):assert function(current,name)==function(published,name)
# The real unsupported preparation must still stop before creating history artifacts.
try:module.profile.require_admission()
except ValueError as e:assert 'remain disabled' in str(e)
else:raise AssertionError('formal profile admission unexpectedly open')
assert verify()==checked
record={'kind':'independent_closed_future_profile_history_integration_review','approved_for_closed_source_integration':True,'operational_admission':False,'actual_target_qualification':False,'scientific_credit':0,'source_freeze_sha256':hashlib.sha256(freeze_path.read_bytes()).hexdigest(),'checks':{'test_methods':result.testsRun,'source_and_metadata_hashes_before_after':checked,'genuine_original_admission':True,'published_69f_authority_functions_identical':True,'formal_admission_closed':True},'genuine_common_plan':ref,'fixture_limit':'Preparation tests have explicitly synthetic unavailable upstream reviews/backend prepin; no real profile unit or target preparation is claimed.','job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'wall_seconds':time.monotonic()-started}
path=Path(__file__).parent/'independent_review.json'
with path.open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(record))
