#!/bin/bash
# One count-bonus unit of the fixed-settings study on serval05 (run by the serval05 queue, under its lock).
set -euo pipefail
umask 0002
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
SNAPSHOT=$(/usr/bin/python3 -c "import json;print(json.load(open('/p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-07-20-03_fixed-cleanrl-ppo-rnd_straight-32-snake-16x2_tabular-neural_uniform-alternating-frozen-network-online-rnd_state-and-reward-normalization_learning-rate-1e-4-1e-3_entropy-1e-3-1e-2_steps-16777216_seeds-0-15_confirmation-20000-20127/source.json'))['snapshot'])")
export PYTHONPATH=$SNAPSHOT/10_jax_exploration_platform/src
cd $SNAPSHOT/10_jax_exploration_platform/scripts/grid_initialization_campaign
exec /p/rlprojects/RND/.venvs/platform_jax/bin/python serval05_unit.py --run-dir /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-07-20-03_fixed-cleanrl-ppo-rnd_straight-32-snake-16x2_tabular-neural_uniform-alternating-frozen-network-online-rnd_state-and-reward-normalization_learning-rate-1e-4-1e-3_entropy-1e-3-1e-2_steps-16777216_seeds-0-15_confirmation-20000-20127 --unit-script /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-07-20-03_fixed-cleanrl-ppo-rnd_straight-32-snake-16x2_tabular-neural_uniform-alternating-frozen-network-online-rnd_state-and-reward-normalization_learning-rate-1e-4-1e-3_entropy-1e-3-1e-2_steps-16777216_seeds-0-15_confirmation-20000-20127/infrastructure/serval05/count_unit.sh
