#!/bin/bash
set -euo pipefail
# Run only inside the actual owned allocation; preserve Slurm's CUDA selector.
: "${SLURM_JOB_ID:?the diagnostic requires its actual Slurm allocation}"
sha256sum --check --quiet /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-06-53_device-memory-capture-diagnostic_jaguar03_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-minutes-20_qos-csresnolim_blas-threads-1_fixture-rounds-7_calls-per-round-10_ownership-cgroup-v2_process-uids-4/source_checksums.sha256
# Remove inherited persistent cache settings before setting the exact reviewed runtime.
for cache_variable_name in ${!JAX_COMPILATION_CACHE@} ${!JAX_PERSISTENT_CACHE@}; do
  unset "$cache_variable_name"
done
export JAX_ENABLE_X64=true
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export JAX_ENABLE_COMPILATION_CACHE=false
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=''
export JAXTARI_SKIP_SPRITE_UPDATE=1
export XDG_DATA_HOME=/p/rlprojects/RND/reference_repos/jaxtari_pixel_study_assets
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
cd /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/data/2026-11-01-06-53_device-memory-capture-diagnostic_jaguar03_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-minutes-20_qos-csresnolim_blas-threads-1_fixture-rounds-7_calls-per-round-10_ownership-cgroup-v2_process-uids-4
# One numeric Slurm step owns the status wrapper and both fresh diagnostic children.
exec srun "--jobid=${SLURM_JOB_ID}" --exact --ntasks=1 --cpus-per-task=8 --gres=gpu:nvidia_rtx_a4500:1 --cpu-bind=threads /p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-06-53_device-memory-capture-diagnostic_jaguar03_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-minutes-20_qos-csresnolim_blas-threads-1_fixture-rounds-7_calls-per-round-10_ownership-cgroup-v2_process-uids-4/wrapper/run_checks.py --run-dir /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1 --unit-id 2026-11-01-06-53_device-memory-capture-diagnostic_jaguar03_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-minutes-20_qos-csresnolim_blas-threads-1_fixture-rounds-7_calls-per-round-10_ownership-cgroup-v2_process-uids-4 --source-commit 474907ef8623aa50f2979954ef2466e14173c05f -- /p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/data/2026-11-01-06-53_device-memory-capture-diagnostic_jaguar03_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-minutes-20_qos-csresnolim_blas-threads-1_fixture-rounds-7_calls-per-round-10_ownership-cgroup-v2_process-uids-4/code/device_memory_capture_regression.py run /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/data/2026-11-01-06-53_device-memory-capture-diagnostic_jaguar03_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-minutes-20_qos-csresnolim_blas-threads-1_fixture-rounds-7_calls-per-round-10_ownership-cgroup-v2_process-uids-4
