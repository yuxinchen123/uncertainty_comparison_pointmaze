"""Use a fresh own CPU child for the irreversible read-only metadata audit hook."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
M=Path(__file__).resolve().parent;S=M.parents[1]/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
spec=importlib.util.spec_from_file_location('_policy_fixture_runtime',S/'profile_metadata_runtime.py')
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
path=next((R/'data').glob('2026-11-05-02-24_common-timing*copies-1_*'))/'measurement_plan.json'
p={'helpers':{'files':{runtime.PREFIX+'nsight_process_identity.py':hashlib.sha256((S/'nsight_process_identity.py').read_bytes()).hexdigest()}}}
observer=runtime.process_observer(p);own=observer.ProcReader().snapshot(os.getpid(),2618919)
owner={k:own[k] for k in ('pid','parent_pid','process_start_time','uids','cgroup')}
owner.update(argv=own['command'],hostname=runtime.socket.gethostname().split('.')[0])
budget=runtime.Budget(runtime.origin(p,owner),p)
payload={'runtime_plan':p,'metadata_origin':budget.origin,'common_plan':str(path)}
with runtime.MetadataScope(p,budget) as scope:
    with runtime.work_alarm(budget):
        result=scope.run(['/usr/bin/python3','-I','-S','-B',str(M/'check_new29_policy.py')],input=json.dumps(payload).encode(),timeout=60)
target=M/('policy_check_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']+'.json')
record={'returncode':result.returncode,'stdout':result.stdout.decode(),'stderr':result.stderr.decode(),
    'actual_scope':scope.proof,'metadata_origin':budget.origin,'passed':result.returncode==0 and result.stderr==b''}
try:
    runtime.validate_scope(scope.proof,budget,scope.observer,scope.owner)
except Exception as error:
    record['scope_validation_error']=repr(error);record['passed']=False
with target.open('x') as handle:
    json.dump(record,handle,indent=2,sort_keys=True);handle.write('\n');handle.flush();os.fsync(handle.fileno())
print(str(target));print(result.stderr.decode())
raise SystemExit(not record['passed'])
