#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -J antmaze-warp-processor-checks
# Run the Warp environment's correctness checks again on a compute node's processors, after the environment's
# non-finite rule changed (2026-11-12), and write the environment profile's document again:
# profiles/antmaze_warp/antmaze_warp_checks_processors.json and profiles/antmaze_warp/antmaze_warp.md.
# The card's measurements in antmaze_warp.json are not touched.
set -uo pipefail
P=/p/rlprojects/RND/10_jax_exploration_platform
RECORD=/p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp/antmaze_warp.json
cd $P
export PYTHONPATH=$P/src PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python
echo "host $(hostname) started $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')"
cp -p "${RECORD%.json}_checks_processors.json" "${RECORD%.json}_checks_processors_before-the-non-finite-rule-change.json"
$PY benchmarks/env/profile_antmaze_warp.py --out "$RECORD" --processor-checks || exit 1
$PY benchmarks/env/profile_antmaze_warp.py --out "$RECORD" --render
rc=$?
echo "ended $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') exit code $rc"
exit $rc
