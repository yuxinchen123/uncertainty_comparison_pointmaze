#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -J ant-integrator-processors
# Run one Python script of this folder on a compute node's processors (MuJoCo Warp on the processor device, C MuJoCo
# in double precision). Arguments: the script and its arguments.
#   sbatch --reservation=<name> -w puma01 -o <log> processor_job.sh <script> [arguments]
set -uo pipefail
P=/p/rlprojects/RND/10_jax_exploration_platform
HERE=$P/runs/2026-11-12-15-56_cleanrl-ppo-rnd-baseline_antmaze-umaze-medium-large_mujoco-warp-physics_ceiling-on-wall-tops_goal-termination_reward-arrival-plus-1-and-step-cost_environments-per-policy-check-4-16-128_seeds-0-15_confirmation-seeds-from-30000/diagnosis/2026-11-13_ant-integrators-in-mujoco-warp
cd $P
# pytest is not in platform_jax_warp: the speed study links its pure-Python packages (the fault-guard test module is imported)
PYTEST_PACKAGES=$P/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/pytest_packages_linked_from_atari_pixels_jax
export PYTHONPATH=$P/src:$PYTEST_PACKAGES PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
if [ -d /localtmp ] && mkdir -p /localtmp/$USER/warp_kernel_cache 2>/dev/null; then
    export WARP_CACHE_PATH=/localtmp/$USER/warp_kernel_cache
else
    mkdir -p /tmp/$USER/warp_kernel_cache; export WARP_CACHE_PATH=/tmp/$USER/warp_kernel_cache
fi
echo "host $(hostname) job ${SLURM_JOB_ID:-none} processors ${SLURM_CPUS_PER_TASK:-?} started $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')"
/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python -u "$@"
rc=$?
echo "ended $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') exit code $rc"
exit $rc
