#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -J divisor-first-arrival-check
# Run one script of this profile with the platform_jax interpreter on a compute node's processors (no card):
#   run_script_cpu.sh <script> <its arguments ...>
set -euo pipefail
P=/p/rlprojects/RND/10_jax_exploration_platform
cd $P
export PYTHONPATH=$P/src PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "host $(hostname) cpus ${SLURM_CPUS_PER_TASK:-?} started $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')"
/p/rlprojects/RND/.venvs/platform_jax/bin/python -u "$@"
