#!/bin/bash
# Final report re-run from the repository code at the current repository commit (the run's snapshot plus the results-table width fix).
set -euo pipefail
umask 0002
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRID_GPU_CLASS=none
export PYTHONPATH=/p/rlprojects/RND/10_jax_exploration_platform/src
cd /p/rlprojects/RND
exec /p/rlprojects/RND/.venvs/platform_jax/bin/python /p/rlprojects/RND/10_jax_exploration_platform/scripts/grid_initialization_campaign/run_report.py --run-dir /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-07-20-03_fixed-cleanrl-ppo-rnd_straight-32-snake-16x2_tabular-neural_uniform-alternating-frozen-network-online-rnd_state-and-reward-normalization_learning-rate-1e-4-1e-3_entropy-1e-3-1e-2_steps-16777216_seeds-0-15_confirmation-20000-20127 --final
