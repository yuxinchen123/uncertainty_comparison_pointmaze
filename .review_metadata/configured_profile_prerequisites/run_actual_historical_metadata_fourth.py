import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

w = Path("/p/rlprojects/RND_worktrees/2026-11-05-07-34_a4500-profile-preparation_common-gpu-types-2_copies-1-2_reviewed-native-process-contract_admission-closed_gpu-operations-0")
r = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
d = w / '.review_metadata/configured_profile_prerequisites'
freeze = json.loads((d / 'source_freeze_4.json').read_text())

def check():
    for name, value in freeze['paths'].items():
        path = w / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == value['sha256']
        assert path.stat().st_mode & 511 == value['mode']

def reference(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

check()
started = time.monotonic()
source = w / '10_jax_exploration_platform/scripts/atari_pixel_campaign/configured_profile_prerequisites.py'
spec = importlib.util.spec_from_file_location('actual_configured_profile_metadata_check', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
next_actions = json.loads((r / 'next_actions.json').read_text())
reviews = [('jaguar03_rtx_a4500', int(count), value) for count, value in
    next_actions['common_a4500_confirmation_preparation']['independent_reviews'].items()]
h100 = next((r / 'remote_setup').glob('2026-11-05-05-49*/independent_actual_common_timing_review.json'))
reviews.append(('h100', 1, reference(h100)))
first = json.loads(Path(reviews[0][2]['path']).read_text())
first_plan = json.loads(Path(first['plan']['path']).read_text())
proposed = {'campaign_run_dir': str(r), 'configuration': first_plan['configuration'],
    'runtime_environment': first_plan['runtime_environment'], 'asset_record': first_plan['asset_record'], **{key: first_plan[key] for key in ('source_root','core','paired_source')}}
passed = []
for hardware, copies, value in reviews:
    result = module.reviewed_context(module.MetadataFiles(), {'hardware_class': hardware, 'copies': copies, 'review': value}, proposed)
    passed.append({'hardware_class': hardware, 'copies': copies, 'review': value,
        'source_commit': result['helpers']['source_commit'], 'helper_count': len(result['helpers']['files'])})
hreview = json.loads(h100.read_text())
hplan = json.loads(Path(hreview['plan']['path']).read_text())
snapshot = module.initial_snapshot(module.MetadataFiles(), hplan['accepted_evidence_snapshot_inputs'], hplan)
qualified = module.qualification_metadata(first_plan)
check()
result = {'kind': 'actual_historical_metadata_compatibility_check', 'passed': True,
    'job_id': os.environ['SLURM_JOB_ID'], 'step_id': os.environ['SLURM_STEP_ID'],
    'seconds': time.monotonic() - started, 'contexts_checked': passed,
    'initial_snapshot': snapshot, 'raw_result_payloads_opened': False,
    'old_raw_consumers_reexecuted': False, 'historical_payloads_rehashed': False,
    'new_unit_prepared': False, 'GPU_or_device_operations': False,
    'four_context_prerequisites_complete': False, 'missing_context': {'hardware_class': 'h100', 'copies': 2},
    'original_qualification_metadata': {'reference': qualified['qualification_result'], 'package_names': sorted(qualified['installed_backend']['packages']), 'current_software_checked': False},
    'source_freeze_sha256': hashlib.sha256((d / 'source_freeze_4.json').read_bytes()).hexdigest()}
(d / 'actual_historical_metadata_4.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({'passed': True, 'seconds': result['seconds'], 'contexts': len(passed),
    'job_id': result['job_id'], 'step_id': result['step_id'], 'four_context_prerequisites_complete': False}))
