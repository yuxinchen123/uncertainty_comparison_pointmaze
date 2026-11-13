#!/bin/bash
#SBATCH -t 01:30:00
#SBATCH -J ant-motion-check-c-mujoco
# Run one Python script of the motion check with the platform_jax_warp interpreter on a compute node's
# processors (no card). Arguments: the script and its arguments.
#   sbatch -p <partition or reservation options> -c <processors> --mem=<memory> -o <log> processor_job.sh <script> [arguments]
set -uo pipefail
P=/p/rlprojects/RND/10_jax_exploration_platform
cd $P
export PYTHONPATH=$P/src PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# every worker process runs one MuJoCo thread; the pool has one process per allotted processor
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host $(hostname) job ${SLURM_JOB_ID:-none} processors ${SLURM_CPUS_PER_TASK:-?} started $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')"
/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python -u "$@"
rc=$?
echo "ended $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') exit code $rc"
exit $rc
