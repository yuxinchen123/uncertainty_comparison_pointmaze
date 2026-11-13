#!/bin/bash
#SBATCH --job-name=rnd-maze-warp-import-check
#SBATCH -p cpu
#SBATCH -A dsimchil
#SBATCH -q standby
#SBATCH -N1
#SBATCH -n1
#SBATCH -c2
#SBATCH -t 00:20:00
# Import check of the Warp environment on Gautschi, on a processor-only node under standby (not charged): imports
# only, no device, no training. The group's two H100 cards were both held by another session's jobs until about
# 2026-11-13 20:07 PT, so the card part of the check is left to the first worker job, which checks the card itself
# before it claims a unit.
#   sbatch --parsable -o <setup folder>/import_check_%j.out import_check_processor_job_gautschi.sh <setup folder>
set -euo pipefail
HERE=${1:?give the folder that holds import_check_rnd_platform_jax_mujoco_warp.py}
PY=/scratch/gautschi/liu4869/envs/rnd_platform_jax_mujoco_warp/bin/python
export PYTHONNOUSERSITE=1
unset LD_LIBRARY_PATH
echo "cluster gautschi, job ${SLURM_JOB_ID:-none}, host $(hostname), start $(date -Is)"
JAX_PLATFORMS=cpu "$PY" -I "$HERE/import_check_rnd_platform_jax_mujoco_warp.py" processor
echo "end $(date -Is)"
