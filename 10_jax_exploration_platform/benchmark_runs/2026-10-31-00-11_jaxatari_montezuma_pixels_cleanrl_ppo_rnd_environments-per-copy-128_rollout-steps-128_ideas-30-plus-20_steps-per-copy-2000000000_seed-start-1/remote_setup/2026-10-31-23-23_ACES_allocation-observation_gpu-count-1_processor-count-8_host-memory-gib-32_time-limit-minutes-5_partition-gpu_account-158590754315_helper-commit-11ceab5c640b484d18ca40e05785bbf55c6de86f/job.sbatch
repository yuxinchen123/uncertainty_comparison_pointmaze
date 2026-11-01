#!/bin/bash
set -euo pipefail
export TZ=UTC
exec srun --unbuffered --ntasks=1 --cpus-per-task=8 --cpu-bind=threads /scratch/user/u.sl359075/envs/rnd_atari_pixels_jax/bin/python -I /scratch/user/u.sl359075/rnd_atari_pixels/git/checkouts/study_helpers_11ceab5c640b/10_jax_exploration_platform/scripts/atari_pixel_campaign/aces_allocation_observation.py capture --directory /scratch/user/u.sl359075/rnd_atari_pixels/runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-10-31-23-23_ACES_allocation-observation_gpu-count-1_processor-count-8_host-memory-gib-32_time-limit-minutes-5_partition-gpu_account-158590754315_helper-commit-11ceab5c640b484d18ca40e05785bbf55c6de86f
