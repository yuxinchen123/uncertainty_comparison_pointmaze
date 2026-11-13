#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:30:00
#SBATCH -J ant-integrator-rollout
# Roll one checkpoint's policies out with each named physics setting in turn (scripts/roll_out_with_setting.py).
#   sbatch -p gpu -w <node> --gres=gpu:<type>:1 -o <log> rollout_job.sh <checkpoint> <unit configuration file> \
#       <environment steps per environment> <record name> <setting slug> [...]
# Records: records/rollout_<record name>_<setting slug>_summary.json and _episodes.npz.
set -uo pipefail
P=/p/rlprojects/RND/10_jax_exploration_platform
HERE=$P/runs/2026-11-12-15-56_cleanrl-ppo-rnd-baseline_antmaze-umaze-medium-large_mujoco-warp-physics_ceiling-on-wall-tops_goal-termination_reward-arrival-plus-1-and-step-cost_environments-per-policy-check-4-16-128_seeds-0-15_confirmation-seeds-from-30000/diagnosis/2026-11-13_ant-integrators-in-mujoco-warp
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
if [ -d /localtmp ] && mkdir -p /localtmp/$USER/warp_kernel_cache 2>/dev/null; then
    export WARP_CACHE_PATH=/localtmp/$USER/warp_kernel_cache
else
    mkdir -p /tmp/$USER/warp_kernel_cache; export WARP_CACHE_PATH=/tmp/$USER/warp_kernel_cache
fi
echo "host $(hostname) job ${SLURM_JOB_ID:-none} started $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
CHECKPOINT=$1; CONFIGURATION=$2; STEPS=$3; NAME=$4; shift 4
rc=0
for slug in "$@"; do
    echo "== setting $slug, $(TZ=America/Los_Angeles date '+%H:%M:%S PT')"
    /p/rlprojects/RND/.venvs/platform_jax_warp/bin/python -u $HERE/scripts/roll_out_with_setting.py --checkpoint "$CHECKPOINT" \
        --configuration-file "$CONFIGURATION" --setting "$slug" --steps "$STEPS" \
        --out $HERE/records/rollout_${NAME}_${slug} || rc=$?
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
done
echo "ended $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') exit code $rc"
exit $rc
