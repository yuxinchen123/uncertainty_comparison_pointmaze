#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -J antmaze-warp-tests
# Run the Warp AntMaze checks on a compute node's processors (MuJoCo Warp on the processor device, no card).
# Arguments: pytest arguments, paths relative to 10_jax_exploration_platform.
# The interpreter is platform_jax_warp (platform_jax plus warp-lang). It holds no pytest, so the folder of
# links to pytest's own pure-Python packages is added to the path, as in infrastructure/run_tests_cpu.sh.
# Warp writes its compiled kernels to a node-local folder, not to the home folder's ~/.cache/warp.
set -uo pipefail
P=/p/rlprojects/RND/10_jax_exploration_platform
cd $P
PYTEST_PACKAGES="$P/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/pytest_packages_linked_from_atari_pixels_jax"
export PYTHONPATH=$P/src:$PYTEST_PACKAGES PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
# GOAL: a node-local kernel cache; /localtmp when the node has it, else the node's /tmp.
if [ -d /localtmp ] && mkdir -p /localtmp/$USER/warp_kernel_cache 2>/dev/null; then
    export WARP_CACHE_PATH=/localtmp/$USER/warp_kernel_cache
else
    mkdir -p /tmp/$USER/warp_kernel_cache; export WARP_CACHE_PATH=/tmp/$USER/warp_kernel_cache
fi
echo "host $(hostname) cpus ${SLURM_CPUS_PER_TASK:-?} started $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')"
echo "warp kernel cache: $WARP_CACHE_PATH ($(find $WARP_CACHE_PATH -maxdepth 2 -type d | wc -l) folders before the run)"
/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python -m pytest -q -s -p no:cacheprovider --durations=0 "$@"
rc=$?
echo "warp kernel cache after the run: $(du -sh $WARP_CACHE_PATH | cut -f1), $(find $WARP_CACHE_PATH -maxdepth 2 -type d | wc -l) folders"
echo "ended $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') exit code $rc"
exit $rc
