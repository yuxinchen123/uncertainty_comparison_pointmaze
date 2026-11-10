#!/bin/bash
#SBATCH -p nolim
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J rnd-extension-tests
# Run the grid-initialization tests on a compute node's processors (no card needed).
set -euo pipefail
P=/p/rlprojects/RND/10_jax_exploration_platform
export PYTHONPATH=$P/src PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=true XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
hostname
/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python -m pytest -q -x $P/tests/grid_initialization "$@"
