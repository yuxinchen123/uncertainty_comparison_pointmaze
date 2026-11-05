"""Check genuine old26 and current29 source namespaces, not their scientific records."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
M=Path(__file__).resolve().parent;S=M.parents[1]/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
spec=importlib.util.spec_from_file_location('_profile_namespace_runtime_fixture',S/'profile_metadata_runtime.py')
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
common_unit=next((R/'data').glob('2026-11-05-02-24_common-timing*copies-1_*'))
common=json.loads((common_unit/'measurement_plan.json').read_bytes())
paths={'confirmation':common_unit/'measurement_plan.json',
    'qualification':Path(common['direct_qualification_inputs']['result']['path']).parent/'measurement_plan.json',
    'correctness':Path(common['common_correctness_inputs']['result']['path']).parent/'measurement_plan.json'}
p={'helpers':{'files':{runtime.PREFIX+'nsight_process_identity.py':hashlib.sha256((S/'nsight_process_identity.py').read_bytes()).hexdigest()}}}
observer=runtime.process_observer(p);owner=observer.ProcReader().snapshot(os.getpid(),2618919)
simple={k:owner[k] for k in ('pid','parent_pid','process_start_time','uids','cgroup')}
simple.update(argv=owner['command'],hostname=runtime.socket.gethostname().split('.')[0])
budget=runtime.Budget(runtime.origin(p,simple),p);proofs=[]
with runtime.MetadataScope(p,budget) as scope:
    with runtime.work_alarm(budget):
        for role,path in paths.items():
            raw=path.read_bytes();payload={'role':role,'plan_path':str(path),'plan_sha256':hashlib.sha256(raw).hexdigest()}
            done=scope.run(['/usr/bin/python3','-I','-S','-B',str(M/'check_exact_staged_namespace.py')],input=json.dumps(payload).encode(),timeout=30)
            if done.returncode or done.stderr:raise ValueError({'role':role,'returncode':done.returncode,'stderr':done.stderr.decode()})
            proof=json.loads(done.stdout);assert proof['source_count']==(29 if role=='confirmation' else 26)
            proofs.append(proof)
runtime.validate_scope(scope.proof,budget,scope.observer,scope.owner)
target=M/('namespace_checks_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']+'.json')
with target.open('x') as handle:
    json.dump({'passed':True,'proofs':proofs,'actual_closed_CPU_scope':scope.proof,'metadata_origin':budget.origin,
        'original_cases_reexecuted':False,'scientific_credit':0,'operational_admission':False},handle,indent=2,sort_keys=True)
    handle.write('\n');handle.flush();os.fsync(handle.fileno())
print(str(target))
