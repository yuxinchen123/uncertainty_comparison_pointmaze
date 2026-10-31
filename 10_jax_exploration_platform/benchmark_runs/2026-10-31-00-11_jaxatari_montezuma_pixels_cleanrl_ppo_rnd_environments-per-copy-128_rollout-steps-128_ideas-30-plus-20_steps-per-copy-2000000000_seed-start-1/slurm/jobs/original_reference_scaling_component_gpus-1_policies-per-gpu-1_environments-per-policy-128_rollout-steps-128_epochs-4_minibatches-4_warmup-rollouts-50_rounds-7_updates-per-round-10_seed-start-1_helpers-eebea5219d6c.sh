#!/bin/bash
set -euo pipefail
unset XLA_FLAGS
source /p/rlprojects/RND_worktrees/atari-cleanrl-source-d28e673/10_jax_exploration_platform/scripts/atari_pixel_campaign/worker_environment.sh
export OMP_NUM_THREADS=8
export JAX_ENABLE_COMPILATION_CACHE=false
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=''
exec srun --wait=0 --ntasks=1 --cpus-per-task=8 --cpu-bind=cores /usr/bin/python3 /p/rlprojects/RND_worktrees/original-reference-scaling-component-helpers-eebea52/10_jax_exploration_platform/scripts/atari_pixel_campaign/run_original_scaling_component.py --launch-inputs /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-10-31-08-45_original_reference_scaling_components_gpus-1-and-8_policies-per-gpu-1_environments-per-policy-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_rounds-7_updates-per-round-10_seed-start-1/gpu-count-1_launch_inputs.json
