#!/bin/bash
set -euo pipefail
umask 0002
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export GRID_GPU_CLASS=nvidia_a40
export PYTHONPATH=/p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-07-20-03_fixed-cleanrl-ppo-rnd_straight-32-snake-16x2_tabular-neural_uniform-alternating-frozen-network-online-rnd_state-and-reward-normalization_learning-rate-1e-4-1e-3_entropy-1e-3-1e-2_steps-16777216_seeds-0-15_confirmation-20000-20127/code/source_5abf6b2f35965445d4dd42c023fa1728aa058bf0/10_jax_exploration_platform/src
cd /p/rlprojects/RND
/usr/bin/python3 /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-07-20-03_fixed-cleanrl-ppo-rnd_straight-32-snake-16x2_tabular-neural_uniform-alternating-frozen-network-online-rnd_state-and-reward-normalization_learning-rate-1e-4-1e-3_entropy-1e-3-1e-2_steps-16777216_seeds-0-15_confirmation-20000-20127/infrastructure/release_one_held_unit.py /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-07-20-03_fixed-cleanrl-ppo-rnd_straight-32-snake-16x2_tabular-neural_uniform-alternating-frozen-network-online-rnd_state-and-reward-normalization_learning-rate-1e-4-1e-3_entropy-1e-3-1e-2_steps-16777216_seeds-0-15_confirmation-20000-20127
IFS=, read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
test "${#devices[@]}" -eq 1
pids=()
for device in "${devices[@]}"; do
  CUDA_VISIBLE_DEVICES="$device" /p/rlprojects/RND/.venvs/platform_jax/bin/python /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-07-20-03_fixed-cleanrl-ppo-rnd_straight-32-snake-16x2_tabular-neural_uniform-alternating-frozen-network-online-rnd_state-and-reward-normalization_learning-rate-1e-4-1e-3_entropy-1e-3-1e-2_steps-16777216_seeds-0-15_confirmation-20000-20127/code/source_5abf6b2f35965445d4dd42c023fa1728aa058bf0/10_jax_exploration_platform/scripts/grid_initialization_campaign/worker.py --run-dir /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-07-20-03_fixed-cleanrl-ppo-rnd_straight-32-snake-16x2_tabular-neural_uniform-alternating-frozen-network-online-rnd_state-and-reward-normalization_learning-rate-1e-4-1e-3_entropy-1e-3-1e-2_steps-16777216_seeds-0-15_confirmation-20000-20127 --stage research --max-seconds 345480 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
