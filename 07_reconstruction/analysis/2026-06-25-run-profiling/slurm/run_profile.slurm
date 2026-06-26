#!/bin/bash
# Baseline profiling on ONE node: cProfile hotspots + clean fps for the 3 run-2 algorithms.
# Single process (ntasks=1, 2 cpus) so the fps is a clean per-worker measurement; pinned to one node
# by the launcher so a later baseline-vs-improved run lands on the SAME node.
#SBATCH --job-name=prof-base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00

set -e
PROF=/p/rlprojects/RND/07_reconstruction/analysis/2026-06-25-run-profiling
CODE=$PROF/analysis/code
DATA=$PROF/data
PY=/u/sl5nw/.conda/envs/exploration/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
NODE=${SLURMD_NODENAME:-$(hostname -s)}

# run <algorithm> <beta> <extra-args>: profile/bench one algorithm, tag output by node
run() {
  $PY $CODE/profile_train.py --algorithm "$1" --beta "$2" --threads 2 --seed 0 \
    --bench_steps 8000 --warmup 1500 $3 \
    --out "$DATA/baseline_${1}_${NODE}.json"
}

echo "[run_profile] node=$NODE starting $(date)"
# rnd_state carries the cProfile hotspot pass (the RND intrinsic path); the other two are fps-only
run rnd_state 100 "--profile --steps 6000"
run gt_position_velocity 1 ""
run rnd_elliptical 0.01 ""
echo "[run_profile] node=$NODE done $(date)"
