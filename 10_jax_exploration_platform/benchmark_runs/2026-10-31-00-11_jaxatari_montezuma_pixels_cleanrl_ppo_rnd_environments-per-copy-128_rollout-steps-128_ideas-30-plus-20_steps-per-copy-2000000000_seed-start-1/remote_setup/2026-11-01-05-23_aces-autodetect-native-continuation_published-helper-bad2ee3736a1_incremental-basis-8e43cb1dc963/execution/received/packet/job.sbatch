#!/bin/bash
set -euo pipefail
export TZ=UTC
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
export CLEANRL_REFERENCE_FILE=/scratch/user/u.sl359075/rnd_atari_pixels/remote_setup/2026-11-01-01-16_aces-full-site-inputs_original-source-d28e673_driver-580.178.04/cleanrl_source/cleanrl/ppo_rnd_envpool.py
exec srun --unbuffered --ntasks=1 --cpus-per-task=8 --cpu-bind=threads /scratch/user/u.sl359075/envs/2026-11-01-02-22_rnd-atari-pixels_reference-torch-2.6.0-cpu_base-python-3.12.14_system-site-packages_without-pip_copied-python/bin/python -I /scratch/user/u.sl359075/rnd_atari_pixels/git/checkouts/study_helpers_bad2ee3736a1/10_jax_exploration_platform/scripts/atari_pixel_campaign/aces_slurm_site_validation.py run --directory /scratch/user/u.sl359075/rnd_atari_pixels/runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-05-23_aces-full-site_h100_autodetect-nvml_driver-580.178.04_policies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-1_gpu-count-1_processor-count-8_host-memory-gib-32_time-limit-hours-3
