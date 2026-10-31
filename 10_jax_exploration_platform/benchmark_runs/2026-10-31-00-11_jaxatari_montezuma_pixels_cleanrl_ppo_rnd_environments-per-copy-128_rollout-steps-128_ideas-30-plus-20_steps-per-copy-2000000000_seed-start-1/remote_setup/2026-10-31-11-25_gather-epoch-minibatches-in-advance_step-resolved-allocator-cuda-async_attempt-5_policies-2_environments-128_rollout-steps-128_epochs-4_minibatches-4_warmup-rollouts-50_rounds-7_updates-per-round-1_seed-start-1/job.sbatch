#!/bin/bash
set -euo pipefail
source /p/rlprojects/RND_worktrees/atari-cleanrl-source-d28e673/10_jax_exploration_platform/scripts/atari_pixel_campaign/worker_environment.sh
exec srun --ntasks=1 --cpus-per-task=8 --cpu-bind=threads /bin/bash /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-10-31-11-25_gather-epoch-minibatches-in-advance_step-resolved-allocator-cuda-async_attempt-5_policies-2_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_rounds-7_updates-per-round-1_seed-start-1/inside_step.sh
