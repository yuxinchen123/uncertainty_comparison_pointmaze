"""Run only frozen configured-helper CPU checks, never a tool or target operation."""
import hashlib,json,os,pathlib,sys,time,unittest
from types import SimpleNamespace
W=pathlib.Path(__file__).resolve().parents[2]
R=W/'.review_metadata/configured_native_process'
F=R/'author_saved_role_source_freeze.json'
freeze=json.loads(F.read_text())
def verify():
    """Compare only the held helper slice, excluding the separate changing profile author slice."""
    for path,record in freeze['files'].items():
        actual=W/path
        assert hashlib.sha256(actual.read_bytes()).hexdigest()==record['sha256'],path
        assert oct(actual.stat().st_mode & 0o777)==record['mode'],path
verify()
sys.path.insert(0,str(W/'10_jax_exploration_platform/scripts/atari_pixel_campaign/tests'))
sys.path.insert(0,str(W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'))
names=['test_configured_native_saved_roles.ConfiguredNativeSavedRoleTests']
suite=unittest.defaultTestLoader.loadTestsFromNames(names)
started=time.monotonic()
result=unittest.TextTestRunner(verbosity=2).run(suite)
verify()
record={'kind':'configured_native_helper_author_cpu_checks','slurm_job_id':os.environ['SLURM_JOB_ID'],'slurm_step_id':os.environ['SLURM_STEP_ID'],
 'source_freeze':{'path':str(F),'sha256':hashlib.sha256(F.read_bytes()).hexdigest()},'tests_run':result.testsRun,
 'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'elapsed_seconds':time.monotonic()-started,
 'nsight_executed':False,'gpu_operations':0,'standing_gpu_descriptor_opened':False,'scientific_credit':0,'target_qualification':False,'profile_admission':False}
output=R/('author_saved_roles_cpu_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']+'.json')
with output.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
print(json.dumps(record));raise SystemExit(not result.wasSuccessful())
