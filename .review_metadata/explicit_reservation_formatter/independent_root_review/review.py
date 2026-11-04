import pathlib,hashlib,json,unittest,importlib.util,os,datetime
w=pathlib.Path('/p/rlprojects/RND_worktrees/2026-11-04-15-06_nsight-process-diagnostic-placement_gpu-count-1_processor-count-1_host-memory-gib-2_time-limit-minutes-5_explicit-reservation-or-unreserved')
f=w/'.review_metadata/explicit_reservation_formatter/source_freeze.json'
assert hashlib.sha256(f.read_bytes()).hexdigest()=='cb01d5c0cffd73f1b02511704dc40d25eb464f79081ad279e46eb30f7f101fd7'
assert os.environ.get('SLURM_JOB_ID')=='6783003'
t=w/'10_jax_exploration_platform/scripts/atari_pixel_campaign/tests/test_nsight_diagnostic_preparation.py'
before={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [f,t,w/'.review_metadata/prepare_nsight_process_diagnostic.py']}
spec=importlib.util.spec_from_file_location('independent_explicit_placement_tests',t);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
suite=unittest.defaultTestLoader.loadTestsFromModule(module);outcome=unittest.TextTestRunner(verbosity=2).run(suite)
assert all(hashlib.sha256(pathlib.Path(x).read_bytes()).hexdigest()==sha for x,sha in before.items())
report={'kind':'independent_explicit_nsight_placement_source_review','passed':outcome.wasSuccessful(),'approved_for_metadata_preparation_only':outcome.wasSuccessful(),'tests_run':outcome.testsRun,'source_freeze_sha256':before[str(f)],'files_unchanged':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'actual_diagnostic_or_gpu_or_network_actions':0}
p=pathlib.Path(__file__).parent/'review.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));raise SystemExit(not outcome.wasSuccessful())
