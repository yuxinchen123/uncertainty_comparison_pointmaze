#!/bin/bash
# Packing measurement for the nvidia_h100_nvl class on serval05 (no Slurm; run under the queue lock).
set -euo pipefail
umask 0002
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export PYTHONPATH=/p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-07-20-03_fixed-cleanrl-ppo-rnd_straight-32-snake-16x2_tabular-neural_uniform-alternating-frozen-network-online-rnd_state-and-reward-normalization_learning-rate-1e-4-1e-3_entropy-1e-3-1e-2_steps-16777216_seeds-0-15_confirmation-20000-20127/code/source_05ba02426f82d9f174f5fd90b9e220a88d600158/10_jax_exploration_platform/src
# The benchmark refuses to run without a scheduler allocation (a guard against login nodes);
# serval05 is a compute machine without Slurm, so name the queue unit as the allocation.
export SLURM_JOB_ID=serval05-queue-h100-packing GRID_GPU_CLASS=nvidia_h100_nvl
hostname; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
cd /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-07-20-03_fixed-cleanrl-ppo-rnd_straight-32-snake-16x2_tabular-neural_uniform-alternating-frozen-network-online-rnd_state-and-reward-normalization_learning-rate-1e-4-1e-3_entropy-1e-3-1e-2_steps-16777216_seeds-0-15_confirmation-20000-20127/code/source_05ba02426f82d9f174f5fd90b9e220a88d600158/10_jax_exploration_platform/scripts/grid_initialization_campaign
timeout 2400 /p/rlprojects/RND/.venvs/platform_jax/bin/python fixed_settings_benchmark.py --run-dir /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-07-20-03_fixed-cleanrl-ppo-rnd_straight-32-snake-16x2_tabular-neural_uniform-alternating-frozen-network-online-rnd_state-and-reward-normalization_learning-rate-1e-4-1e-3_entropy-1e-3-1e-2_steps-16777216_seeds-0-15_confirmation-20000-20127 --gpu-class nvidia_h100_nvl
echo finished
