"""Inspect only the small accepted qualification's backend metadata shape."""
import json
import os
from pathlib import Path

r = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
n = json.loads((r / 'next_actions.json').read_text())
review = json.loads(Path(n['common_a4500_confirmation_preparation']['independent_reviews']['1']['path']).read_text())
plan = json.loads(Path(review['plan']['path']).read_text())
path = Path(plan['target_qualification_inputs']['result']['path'])
assert path.stat().st_size < 16 * 1024 * 1024
value = json.loads(path.read_text())
print(json.dumps({'job_id': os.environ['SLURM_JOB_ID'], 'step_id': os.environ['SLURM_STEP_ID'],
    'qualification_result_bytes': path.stat().st_size, 'keys': list(value),
    'installed_backend_keys': list(value['installed_backend']),
    'gpu_libraries_keys': list(value['installed_backend']['gpu_libraries']),
    'jaxtari_source_keys': list(value['installed_backend']['jaxtari_source'])}, indent=2))
