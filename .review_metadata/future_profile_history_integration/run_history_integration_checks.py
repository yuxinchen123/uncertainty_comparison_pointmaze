"""Affected preparation/shared-reader/source-admission fixtures only; no target or numerical work."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import unittest
M=Path(__file__).resolve().parent;W=M.parents[1];S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
sys.path.insert(0,str(S/'tests'))
spec=importlib.util.spec_from_file_location('profile_history_integration_fixtures',S/'tests/test_direct_profile_prerequisites.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
if len(sys.argv)>1:
    suite=unittest.TestSuite(module.DirectProfileMetadataTests(name) for name in sys.argv[1:])
else:
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(module.ProfileHistoryIntegrationTests))
    suite.addTest(module.DirectProfileMetadataTests('test_helper_union_includes_backend_and_separate_profile_observers'))
    suite.addTest(module.DirectProfileMetadataTests('test_explicit_old28_profile_still_reaches_original_raw_gate'))
started=time.time();result=unittest.TextTestRunner(verbosity=2).run(suite)
target=M/('integration_checks_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']+'.json')
with target.open('x') as handle:
    json.dump({'passed':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'started_at_seconds':started,'finished_at_seconds':time.time(),'synthetic_upstream_review_and_source_premises':True,
        'actual_published_69f_claim_and_live_source_checks_called':True,'genuine_original_admission_sha256':module.profile.ORIGINAL_ADMISSION_SHA256,
        'actual_profile_unit_created':False,'NUM_or_GPU_operations':0,'operational_admission':False},handle,indent=2,sort_keys=True)
    handle.write('\n');handle.flush();os.fsync(handle.fileno())
raise SystemExit(not result.wasSuccessful())
