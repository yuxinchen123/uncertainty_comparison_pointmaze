#!/bin/bash
# Keep the pixel study's precision, assets and dependencies explicit.
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=true
export JAX_DEFAULT_MATMUL_PRECISION=highest
# Original CleanRL requests deterministic operations; apply that choice to XLA too.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_exclude_nondeterministic_ops=true"
export JAXTARI_SKIP_SPRITE_UPDATE=1
export XDG_DATA_HOME="${ATARI_PIXEL_ASSETS:-/p/rlprojects/RND/reference_repos/jaxtari_pixel_study_assets}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
