#!/bin/bash
set -euo pipefail
export TZ=UTC
exec srun --unbuffered --ntasks=1 --cpus-per-task=8 --cpu-bind=threads /usr/bin/python3 -I /p/rlprojects/RND_worktrees/h100-allocation-observation-helpers-bba2278eb57e/10_jax_exploration_platform/scripts/atari_pixel_campaign/h100_nvl_allocation_observation.py capture --directory /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-10-31-17-37_h100-nvl_allocation-observation_hosts-serval06-to-serval09_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-minutes-5_resource-source-job-6779137
