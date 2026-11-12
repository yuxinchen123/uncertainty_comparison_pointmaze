#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -J maze-trainer-tests
# Run platform tests on a compute node's processors (no card). Arguments: pytest arguments,
# paths relative to 10_jax_exploration_platform. Numerical tests never run on a login node.
set -euo pipefail
P=/p/rlprojects/RND/10_jax_exploration_platform
cd $P
# The platform_jax environment holds no pytest. The folder below links only pytest's own pure-Python
# packages (pytest, _pytest, pluggy, iniconfig, py) from the atari_pixels_jax environment, so jax,
# numpy and mujoco still come from platform_jax. Nothing is written into either environment:
# no bytecode files, and no pytest cache folder.
PYTEST_PACKAGES="$P/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/pytest_packages_linked_from_atari_pixels_jax"
export PYTHONPATH=$P/src:$PYTEST_PACKAGES PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "host $(hostname) cpus ${SLURM_CPUS_PER_TASK:-?} started $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')"
/p/rlprojects/RND/.venvs/platform_jax/bin/python -m pytest -q -p no:cacheprovider "$@"
