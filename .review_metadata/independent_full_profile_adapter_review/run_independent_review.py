import ast, hashlib, importlib.util, json, os, stat, subprocess, sys, tempfile, time, unittest
from pathlib import Path
W=Path('/p/rlprojects/RND_worktrees/2026-11-05-11-09_future-profile-exact-readers_shared-metadata-seconds-300_work-seconds-240_cleanup-seconds-60_admission-closed')
M=W/'.review_metadata/independent_full_profile_adapter_review';S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
part=sys.argv[1];started=time.monotonic();report={'part':part,'job':os.environ.get('SLURM_JOB_ID'),'step':os.environ.get('SLURM_STEP_ID'),'gpu_operations':0,'NUM_operations':0,'operational_admission':False}
assert report['job']=='6821437' and report['step'].isdecimal()
freeze=json.loads((W/'.review_metadata/future_profile_exact_readers_budget/source_freeze.json').read_bytes())
assert hashlib.sha256((W/'.review_metadata/future_profile_exact_readers_budget/source_freeze.json').read_bytes()).hexdigest()=='b23cf4c912d8207f55b1671ce89b620f30800c62782b2334cbcffb1a040a69f5'
for relative,value in freeze['source_and_tests'].items():
 path=W/relative;assert hashlib.sha256(path.read_bytes()).hexdigest()==value['sha256'] and path.stat().st_mode&0o777==value['mode'];ast.parse(path.read_bytes())
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
if part in ('runtime','adapters'):
 sys.path.insert(0,str(S/'tests'));sys.path.insert(0,str(S));suite=unittest.TestSuite()
 if part=='runtime':
  module=load('independent_profile_runtime_tests',S/'tests/test_profile_metadata_runtime.py');suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
 else:
  first=load('independent_profile_prereq_tests',S/'tests/test_direct_profile_prerequisites.py');second=load('independent_profile_saved_tests',S/'tests/test_direct_profile_saved_consumers.py')
  suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(first.DirectProfileMetadataTests))
  for name,value in vars(second).items():
   if isinstance(value,type) and issubclass(value,unittest.TestCase):suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(value))
 result=unittest.TextTestRunner(verbosity=2).run(suite);report.update(passed=result.wasSuccessful(),methods=result.testsRun,failures=len(result.failures),errors=len(result.errors))
elif part=='baseline':
 total=0
 for relative,expected in freeze['protected_tracked_source_files'].items():
  path=W/relative;raw=path.read_bytes();total+=len(raw);assert hashlib.sha256(raw).hexdigest()==expected,relative
 for path,expected in freeze['protected_original_numerical_files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
 report.update(passed=True,protected_tracked=len(freeze['protected_tracked_source_files']),protected_numerical=len(freeze['protected_original_numerical_files']),protected_source_bytes=total)
elif part=='dependency':
 profile=load('independent_profile_dependency',S/'direct_profile_prerequisites.py')
 helper=Path('/p/rlprojects/RND_worktrees/2026-11-05-04-02_common-timing-helper_source-69f1b507_helpers-29_numerical-source-d28e6735_gpu-count-0')
 binding=load('actual_approved_69f_binding',helper/'10_jax_exploration_platform/scripts/atari_pixel_campaign/serval02_card1_binding.py')
 R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
 P=Path(json.loads((R/'next_actions.json').read_bytes())['h100_common_timing_controller_preparation']['copy_one_request']['packet']);common=json.loads((R/'data'/P.name/'measurement_plan.json').read_bytes())
 plan={'campaign_run_dir':str(R),'unit_id':'nonoperational_profile_claim_source_fixture','source_root':common['source_root'],'source_commit':common['source_commit'],'configuration':common['configuration'],'copy_seeds':[1]}
 history=profile.history_pointer(plan,common)
 with tempfile.TemporaryDirectory() as temporary:
  path=Path(temporary)/'inputs.json';path.write_bytes(profile.encoded(history));plan['direct_binding']={'frozen_inputs':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}
  try:binding.current_admission_claim(plan)
  except KeyError as error:
   assert str(error)=="'admission'";report.update(passed=True,blocking_dependency_reproduced=True,error_type='KeyError',error=str(error),profile_pointer=history,binding_sha256=hashlib.sha256((helper/'10_jax_exploration_platform/scripts/atari_pixel_campaign/serval02_card1_binding.py').read_bytes()).hexdigest(),fixture_only=True,actual_profile_unit_created=False)
  else:raise AssertionError('the expected compatibility failure did not reproduce')
else:raise ValueError(part)
report['elapsed_seconds']=time.monotonic()-started
path=M/('independent_'+part+'_'+report['job']+'_'+report['step']+'.json')
with path.open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(report,sort_keys=True));raise SystemExit(0 if report['passed'] else 1)
