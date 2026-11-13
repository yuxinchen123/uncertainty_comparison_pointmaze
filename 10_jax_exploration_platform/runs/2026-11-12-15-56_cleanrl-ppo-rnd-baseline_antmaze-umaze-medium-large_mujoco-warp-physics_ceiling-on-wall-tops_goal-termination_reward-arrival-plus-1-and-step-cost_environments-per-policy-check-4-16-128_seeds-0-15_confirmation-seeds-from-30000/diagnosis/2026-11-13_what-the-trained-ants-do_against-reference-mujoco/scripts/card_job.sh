#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 00:45:00
#SBATCH -J ant-motion-check-rollout
# Roll the policies of one checkpoint out in the Warp AntMaze on one card (scripts/roll_out_and_record.py).
#   sbatch -p <partition> -w <node> --gres=gpu:1 -o <log> card_job.sh <checkpoint> <unit configuration file> <out without extension> [more options]
set -uo pipefail
P=/p/rlprojects/RND/10_jax_exploration_platform
HERE=$P/runs/2026-11-12-15-56_cleanrl-ppo-rnd-baseline_antmaze-umaze-medium-large_mujoco-warp-physics_ceiling-on-wall-tops_goal-termination_reward-arrival-plus-1-and-step-cost_environments-per-policy-check-4-16-128_seeds-0-15_confirmation-seeds-from-30000/diagnosis/2026-11-13_what-the-trained-ants-do_against-reference-mujoco
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
if [ -d /localtmp ] && mkdir -p /localtmp/$USER/warp_kernel_cache 2>/dev/null; then
    export WARP_CACHE_PATH=/localtmp/$USER/warp_kernel_cache
else
    mkdir -p /tmp/$USER/warp_kernel_cache; export WARP_CACHE_PATH=/tmp/$USER/warp_kernel_cache
fi
echo "host $(hostname) job ${SLURM_JOB_ID:-none} started $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
CHECKPOINT=$1; CONFIGURATION=$2; OUT=$3; shift 3
/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python -u $HERE/scripts/roll_out_and_record.py --checkpoint "$CHECKPOINT" --configuration-file "$CONFIGURATION" --out "$OUT" "$@"
rc=$?
echo "ended $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') exit code $rc"
exit $rc
