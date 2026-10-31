#!/bin/bash
set -euo pipefail
source /p/rlprojects/RND_worktrees/atari-cleanrl-source-d28e673/10_jax_exploration_platform/scripts/atari_pixel_campaign/worker_environment.sh
export JAX_ENABLE_X64=true
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true
export JAX_ENABLE_COMPILATION_CACHE=false
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=''
export JAXTARI_SKIP_SPRITE_UPDATE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CLEANRL_REFERENCE_FILE=/p/rlprojects/RND/08_cleanrl_ppo_rnd/cleanrl/cleanrl/ppo_rnd_envpool.py
exec srun --ntasks=1 --cpus-per-task=8 --cpu-bind=threads /p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python /p/rlprojects/RND_worktrees/atari-h100-nvl-site-validation-helpers-e0dd3cd4fb4b/10_jax_exploration_platform/scripts/atari_pixel_campaign/h100_nvl_slurm_site_validation.py run --directory /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-10-31-14-48_h100-nvl-slurm-site-validation_hosts-serval06-serval07-serval08-serval09_policies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-1
