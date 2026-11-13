#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH -J ant-integrator-speed
# Time whole training iterations of the U-maze trainer for each named physics setting (scripts/profile_integrators.py),
# 16 environments per policy, the policy counts given, on one card. Records: records/speed_<card>_job-<id>.jsonl.
#   sbatch -p gpu -w <node> --gres=gpu:<type>:1 -o <log> speed_job.sh <policy counts, e.g. 128,512> <setting slug> [...]
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
CARD=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-')
COPIES=$1; shift
rc=0
for slug in "$@"; do
    echo "== setting $slug, $(TZ=America/Los_Angeles date '+%H:%M:%S PT')"
    /p/rlprojects/RND/.venvs/platform_jax_warp/bin/python -u $HERE/scripts/profile_integrators.py \
        --configuration-file $HERE/configurations/antmaze_umaze_${slug}_guard-30-and-100.toml --map umaze \
        --environments-per-policy 16 --copies "$COPIES" --budget-minutes 15 \
        --out $HERE/records/speed_${CARD}_job-${SLURM_JOB_ID:-none}.jsonl || rc=$?
done
# processes of other users on this job's card during the job (a timing with one is not clean)
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "ended $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') exit code $rc"
exit $rc
