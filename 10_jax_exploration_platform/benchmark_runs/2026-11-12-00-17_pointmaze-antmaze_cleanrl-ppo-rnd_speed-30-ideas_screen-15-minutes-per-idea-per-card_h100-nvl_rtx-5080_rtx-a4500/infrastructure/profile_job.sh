#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J maze-speed-profile
# One speed profile on one card: profile_job.sh <python script, path relative to the platform folder> <its arguments ...>
# The partition, the node, the card, the processors, the memory and the 15-minute limit are given to sbatch
# by submit_profile.sh. The card belongs to this job alone, so no other process can disturb the timing.
set -uo pipefail
P=/p/rlprojects/RND/10_jax_exploration_platform
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# as the training units run: the card's memory is taken as needed, not all at the start
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH=/usr/lib/nvidia:${LD_LIBRARY_PATH:-}
echo "host $(hostname) job $SLURM_JOB_ID started $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
cd $P
script=$1; shift
/p/rlprojects/RND/.venvs/platform_jax/bin/python "$script" "$@"
rc=$?
echo "ended with exit code $rc at $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')"
exit $rc
