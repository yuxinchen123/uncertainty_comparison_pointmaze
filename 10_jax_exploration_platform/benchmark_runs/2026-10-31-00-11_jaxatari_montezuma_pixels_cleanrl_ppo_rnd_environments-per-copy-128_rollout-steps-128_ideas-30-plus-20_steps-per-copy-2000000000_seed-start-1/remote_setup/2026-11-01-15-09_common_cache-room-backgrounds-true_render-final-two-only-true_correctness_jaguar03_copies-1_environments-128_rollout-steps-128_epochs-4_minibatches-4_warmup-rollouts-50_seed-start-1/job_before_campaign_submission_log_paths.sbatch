#!/bin/bash
#SBATCH --job-name=pixel-common-correctness
#SBATCH --output=/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-15-09_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_jaguar03_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1/slurm-%j.out
#SBATCH --error=/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-15-09_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_jaguar03_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1/slurm-%j.err
set -euo pipefail
umask 077
cd /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=true
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true
export JAXTARI_SKIP_SPRITE_UPDATE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XDG_DATA_HOME=/p/rlprojects/RND/reference_repos/jaxtari_pixel_study_assets
export SLURM_CONF=/etc/slurm/slurm.conf
export JAX_ENABLE_COMPILATION_CACHE=false
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=''
unset JAX_COMPILATION_CACHE_DIR JAX_COMPILATION_CACHE_MAX_SIZE JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES
unset PYTEST_ADDOPTS PYTEST_PLUGINS PYTEST_CURRENT_TEST PYTHONOPTIMIZE CLEANRL_REFERENCE_FILE
export LD_LIBRARY_PATH="/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
sha256sum --check --strict /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-15-09_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_jaguar03_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1/source_checksums.sha256
/usr/bin/python3 -I -S -B /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-15-09_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_jaguar03_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1/wait_for_owned_inputs.py --unit /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/data/2026-11-01-15-09_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_jaguar03_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1 --plan-sha256 4acda94f98d53f2a907505cc788da2f8251acf452e0b4c48971a68bfada67354 --helper-commit e5afbf6b51a3d1666b46e71c0631a45d47080cd8
exec srun --nodes=1 --ntasks=1 --cpus-per-task=8 --cpu-bind=cores --gres=gpu:nvidia_rtx_a4500:1 --kill-on-bad-exit=1 /p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python -B /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/data/2026-11-01-15-09_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_jaguar03_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1/code/measure_selected.py --prepared-unit /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/data/2026-11-01-15-09_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_jaguar03_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1
