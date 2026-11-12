#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -J maze-trainer-tests
# Run platform tests on a compute node's processors (no card). Arguments: pytest arguments,
# paths relative to 10_jax_exploration_platform. Numerical tests never run on a login node.
set -euo pipefail
P=/p/rlprojects/RND/10_jax_exploration_platform
cd $P
export PYTHONPATH=$P/src PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "host $(hostname) cpus ${SLURM_CPUS_PER_TASK:-?} started $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')"
/p/rlprojects/RND/.venvs/platform_jax/bin/python -m pytest -q "$@"
