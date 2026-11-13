#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:40:00
#SBATCH -J antmaze-medium-large-profile
# Time AntMaze unit shapes on one card (scripts/profile_unit_shapes.py), one or more groups in a row.
#   sbatch -p gpu -w <node> --gres=gpu:<type>:1 -o <log> card_job.sh <budget minutes> <group> [<group> ...]
# A group is map:environments-per-policy:copies, e.g. medium:16:64,128,256 (the map's stable-physics
# configuration file under configs/experiments/ is used). Records go to records/profile_<card>_job-<id>.jsonl.
set -uo pipefail
P=/p/rlprojects/RND/10_jax_exploration_platform
HERE=$P/runs/2026-11-12-15-56_cleanrl-ppo-rnd-baseline_antmaze-umaze-medium-large_mujoco-warp-physics_ceiling-on-wall-tops_goal-termination_reward-arrival-plus-1-and-step-cost_environments-per-policy-check-4-16-128_seeds-0-15_confirmation-seeds-from-30000/diagnosis/2026-11-13_medium-and-large-readiness
TAIL=warp-physics_time-step-0.005_frame-skip-10_fault-guard-30-metres-and-100-radians-per-second_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-204800.toml
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
if [ -d /localtmp ] && mkdir -p /localtmp/$USER/warp_kernel_cache 2>/dev/null; then
    export WARP_CACHE_PATH=/localtmp/$USER/warp_kernel_cache
else
    mkdir -p /tmp/$USER/warp_kernel_cache; export WARP_CACHE_PATH=/tmp/$USER/warp_kernel_cache
fi
echo "host $(hostname) job ${SLURM_JOB_ID:-none} started $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
CARD=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-')
BUDGET=$1; shift
START=$(date +%s)
rc=0
for group in "$@"; do
    IFS=: read -r MAP ENVS COPIES <<< "$group"
    # the budget is for the whole job: each group gets what is left of it
    LEFT=$(( BUDGET * 60 - ($(date +%s) - START) ))
    [ "$LEFT" -gt 60 ] || { echo "budget used: group $group not started"; break; }
    echo "== group $group, $LEFT s of the budget left, $(TZ=America/Los_Angeles date '+%H:%M:%S PT')"
    /p/rlprojects/RND/.venvs/platform_jax_warp/bin/python -u $HERE/scripts/profile_unit_shapes.py \
        --configuration-file $P/configs/experiments/antmaze_${MAP}_$TAIL --map "$MAP" \
        --environments-per-policy "$ENVS" --copies "$COPIES" --budget-minutes "$(( LEFT / 60 ))" \
        --out $HERE/records/profile_${CARD}_job-${SLURM_JOB_ID:-none}.jsonl || rc=$?
done
echo "ended $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') exit code $rc"
exit $rc
