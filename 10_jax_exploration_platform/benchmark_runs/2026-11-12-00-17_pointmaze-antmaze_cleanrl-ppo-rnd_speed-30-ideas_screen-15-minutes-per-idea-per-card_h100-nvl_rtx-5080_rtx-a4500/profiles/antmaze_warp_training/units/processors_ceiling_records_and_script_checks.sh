#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -J antmaze-ceiling-processor-checks
# On a compute node's processors (no card), with the platform_jax_warp interpreter:
#   1. the measurement scripts of the card units at a tiny size, so that a fault in a script is found before
#      a 15-minute card unit runs it: the training-iteration script's own test, and the random-torque script
#      with both physics (its output goes to the node's temporary folder and is not kept);
#   2. the record of the nine ants, the thrown ant and the periodic torque patterns
#      (units/diagnose_ordinary_and_thrown_ants.py -> records_ceiling/ordinary-and-thrown-ants_processors.json);
#   3. the Markdown document again.
# With the argument record-only, part 1 is left out (job 6890149 ran all three parts, with 128 ants in the
# record's C MuJoCo random-torque part; the record was written again with 1,024 ants).
set -uo pipefail
RECORD_ONLY=${1:-}
P=/p/rlprojects/RND/10_jax_exploration_platform
OUT=$P/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_training
PYTEST_PACKAGES="$P/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/infrastructure/pytest_packages_linked_from_atari_pixels_jax"
cd $P
export PYTHONPATH=$P/src:$PYTEST_PACKAGES PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
if [ -d /localtmp ] && mkdir -p /localtmp/$USER/warp_kernel_cache 2>/dev/null; then
    export WARP_CACHE_PATH=/localtmp/$USER/warp_kernel_cache
else
    mkdir -p /tmp/$USER/warp_kernel_cache; export WARP_CACHE_PATH=/tmp/$USER/warp_kernel_cache
fi
PY=/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python
pt() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
echo "host $(hostname) cpus ${SLURM_CPUS_PER_TASK:-?} started $(pt)"
worst=0
SCRATCH=$(mktemp -d)
[ "$RECORD_ONLY" = record-only ] && PHYSICS_TO_CHECK="" || PHYSICS_TO_CHECK="warp jax"
for physics in $PHYSICS_TO_CHECK; do
    $PY $OUT/units/diagnose_arrivals_with_random_torques.py --physics $physics --environments 16 --steps 200 \
        --goal-on-the-start-cell --out $SCRATCH/arrivals_$physics.json | grep -v "^   {" | tail -3
    rc=${PIPESTATUS[0]}; echo "$(pt) random-torque script, physics $physics, goal on the start cell: exit code $rc"; [ $rc -gt $worst ] && worst=$rc
    $PY $OUT/units/diagnose_arrivals_with_random_torques.py --physics $physics --environments 64 --steps 800 \
        --out $SCRATCH/torques_$physics.json | grep -v "^   {" | tail -3
    rc=${PIPESTATUS[0]}; echo "$(pt) random-torque script, physics $physics: exit code $rc"; [ $rc -gt $worst ] && worst=$rc
    $PY -c "
import json
for name in ('arrivals', 'torques'):
    r = json.load(open('$SCRATCH/' + name + '_$physics.json'))
    print(name, {k: v for k, v in r.items() if k not in ('arrivals', 'pool')}, 'first arrival:', r['arrivals'][:1])
"
done
rm -rf $SCRATCH
if [ "$RECORD_ONLY" != record-only ]; then
    $PY -m pytest -q -p no:cacheprovider benchmarks/env/test_profile_antmaze_warp_training.py
    rc=$?; echo "$(pt) the training-iteration script's test: exit code $rc"; [ $rc -gt $worst ] && worst=$rc
fi
$PY $OUT/units/diagnose_ordinary_and_thrown_ants.py --out $OUT/records_ceiling/ordinary-and-thrown-ants_processors.json
rc=$?; echo "$(pt) the nine ants' record: exit code $rc"; [ $rc -gt $worst ] && worst=$rc
$PY benchmarks/env/profile_antmaze_warp_training.py --render --records $OUT/records --markdown $OUT/antmaze_warp_training.md
rc=$?; echo "$(pt) the document: exit code $rc"; [ $rc -gt $worst ] && worst=$rc
echo "ended $(pt) exit code $worst"
exit $worst
