"""Run affected closed-profile metadata/source fixtures; no actual scientific replay."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import unittest
M=Path(__file__).resolve().parent;S=M.parents[1]/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
sys.path.insert(0,str(S/'tests'))
def load(name):
    spec=importlib.util.spec_from_file_location('_focused_'+name,S/'tests'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
first=load('test_direct_profile_prerequisites');second=load('test_direct_profile_saved_consumers')
suite=unittest.TestSuite()
if len(sys.argv)>1:
    for name in sys.argv[1:]:suite.addTest(first.DirectProfileMetadataTests(name))
else:
    for cls in (first.DirectProfileMetadataTests,second.SourceAndProtocolChecks,second.NewTwentyNineCommonMetadataChecks):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
started=time.time();result=unittest.TextTestRunner(verbosity=2).run(suite)
target=M/('adapter_checks_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']+'.json')
with target.open('x') as handle:
    json.dump({'passed':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'started_at_seconds':started,'finished_at_seconds':time.time(),'actual_scientific_records_replayed':False,
        'NUM_or_GPU_operations':0,'operational_admission':False},handle,indent=2,sort_keys=True)
    handle.write('\n');handle.flush();os.fsync(handle.fileno())
raise SystemExit(not result.wasSuccessful())
