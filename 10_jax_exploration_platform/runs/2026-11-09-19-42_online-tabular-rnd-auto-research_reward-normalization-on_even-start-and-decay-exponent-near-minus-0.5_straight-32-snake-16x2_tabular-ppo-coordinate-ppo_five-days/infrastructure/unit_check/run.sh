#!/bin/bash
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 00:40:00
#SBATCH -J rnd-auto-unit-check
# One short real unit per new kind of idea, through the frozen code copy's train_unit.py, in a scratch run folder.
set -euo pipefail
S=/p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/code/source_9cd855bb4d2c2da5399974396a9183c58fba890a/10_jax_exploration_platform
export PYTHONPATH=$S/src PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false GRID_GPU_CLASS=nvidia_rtx_a4500
W=/p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/infrastructure/unit_check/scratch; rm -rf $W; mkdir -p $W/specifications $W/infrastructure
cp -r /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/infrastructure/packing $W/infrastructure/
/p/rlprojects/RND/.venvs/platform_jax/bin/python - <<'PY'
import json, sys
sys.path.insert(0, '/p/rlprojects/RND/10_jax_exploration_platform/scripts/grid_initialization_campaign')
from auto_research_units import unit_specifications
ideas = json.load(open('/p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/batches/01_diagnostics-even-start-and-decay-components/rl_ideas.json'))
keep = ['unit-start-shrink-target-sgd-2-length-readout', 'unit-start-adagrad-0.1', 'initial-ratio-readout', 'bias-start-0.5', 'separate-adam-constant-learning-rate-1e-4']
for spec in unit_specifications('check', {name: ideas[name] for name in keep}):
    if spec['geometry']['kind'] != 'snake' or spec['settings'][0]['entropy_coefficient'] != 1e-3:
        continue
    spec.update(settings=spec['settings'][:2], seed_count=2, total_timesteps=4096, stop_after_timesteps=2048, name=spec['name'].replace('seeds-0-31', 'check'))
    json.dump(spec, open('/p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/infrastructure/unit_check/scratch/specifications/' + spec['name'] + '.json', 'w'))
PY
for f in $W/specifications/*.json; do /p/rlprojects/RND/.venvs/platform_jax/bin/python $S/scripts/grid_initialization_campaign/train_unit.py --run-dir $W --specification $f; done
/p/rlprojects/RND/.venvs/platform_jax/bin/python - <<'PY'
import json, glob, numpy as np
for path in sorted(glob.glob('/p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/infrastructure/unit_check/scratch/data/research/*/summary.json')):
    s = json.load(open(path)); b = np.load(path.replace('summary.json', 'behavior.npz'))
    print(s['specification']['idea'][:50], s['specification']['policy_kind'], 'budget', s['budget'], 'copies', len(s['copy_records']),
          'initial bonus range', float(b['rnd_error_initial'].min()), float(b['rnd_error_initial'].max()),
          'finite', bool(np.isfinite(b['rnd_error_history']).all()))
PY
