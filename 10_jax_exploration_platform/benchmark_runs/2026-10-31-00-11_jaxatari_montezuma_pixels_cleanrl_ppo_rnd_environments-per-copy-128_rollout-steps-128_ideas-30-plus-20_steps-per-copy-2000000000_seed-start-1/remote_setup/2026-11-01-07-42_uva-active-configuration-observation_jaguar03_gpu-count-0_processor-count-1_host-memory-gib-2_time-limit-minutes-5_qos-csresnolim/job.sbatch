#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?the observer requires its actual owned allocation}"
sha256sum --check --quiet /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-07-42_uva-active-configuration-observation_jaguar03_gpu-count-0_processor-count-1_host-memory-gib-2_time-limit-minutes-5_qos-csresnolim/source_checksums.sha256
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cd /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/data/2026-11-01-07-42_uva-active-configuration-observation_jaguar03_gpu-count-0_processor-count-1_host-memory-gib-2_time-limit-minutes-5_qos-csresnolim
# One zero-GPU numeric Slurm step performs only standard-library metadata reads.
exec srun "--jobid=${SLURM_JOB_ID}" --exact --ntasks=1 --cpus-per-task=1 --gres=none --cpu-bind=threads /usr/bin/python3 -B /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/data/2026-11-01-07-42_uva-active-configuration-observation_jaguar03_gpu-count-0_processor-count-1_host-memory-gib-2_time-limit-minutes-5_qos-csresnolim/code/uva_active_configuration_observation.py run /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/data/2026-11-01-07-42_uva-active-configuration-observation_jaguar03_gpu-count-0_processor-count-1_host-memory-gib-2_time-limit-minutes-5_qos-csresnolim
