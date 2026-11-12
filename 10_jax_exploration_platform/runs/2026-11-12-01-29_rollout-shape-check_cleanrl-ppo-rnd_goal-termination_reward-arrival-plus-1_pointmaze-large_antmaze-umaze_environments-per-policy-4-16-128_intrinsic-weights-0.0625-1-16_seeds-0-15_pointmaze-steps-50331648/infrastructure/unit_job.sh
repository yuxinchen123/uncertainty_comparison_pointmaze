#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH --gpus-per-node=1
#SBATCH -J maze-rollout-shape
#SBATCH --requeue
#SBATCH --open-mode=append
# One unit of the rollout-shape check on one card: unit_job.sh <unit id> <configuration file> <more options ...>
# The unit saves a checkpoint every 20 minutes and continues from it when the job starts again.
set -uo pipefail
P=/p/rlprojects/RND/10_jax_exploration_platform
RUN=$(cd "$(dirname "$(scontrol show job "$SLURM_JOB_ID" | grep -oP 'Command=\K\S+')")/.." && pwd)
unit=$1; config=$2; shift 2
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH=/usr/lib/nvidia:${LD_LIBRARY_PATH:-}
echo "host $(hostname) job $SLURM_JOB_ID unit $unit started $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
cd $P
/p/rlprojects/RND/.venvs/platform_jax/bin/python scripts/run_training.py --run-dir "$RUN" --unit-id "$unit" \
    --configuration-file "$config" --checkpoint-seconds 1200 "$@"
rc=$?
echo "unit $unit ended with exit code $rc at $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')"
exit $rc
