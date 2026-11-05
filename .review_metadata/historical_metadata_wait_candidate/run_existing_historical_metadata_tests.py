import sys,unittest,json,os
from pathlib import Path
w=Path(__file__).resolve().parents[2]
s=w/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
sys.path.insert(0,str(s));sys.path.insert(0,str(s/'tests'))
assert os.environ.get('SLURM_JOB_ID')=='6783003'
import test_direct_historical_consumer
suite=unittest.defaultTestLoader.loadTestsFromModule(test_direct_historical_consumer)
result=unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({'tests':result.testsRun,'passed':result.wasSuccessful(),'actual_job':os.environ['SLURM_JOB_ID'],'actual_step':os.environ.get('SLURM_STEP_ID'),'actual_num_or_gpu_work':False,'production_admission':False}))
raise SystemExit(0 if result.wasSuccessful() else 1)
