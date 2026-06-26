#!/bin/bash
# Bold-trainer benchmark on ONE node: (1) validate the SAC core learns on Pendulum-v1, (2) fast-trainer fps
# on PointMaze for all 3 algorithms, (3) SB3 baseline fps on the SAME node (for the fast/SB3 ratio).
#SBATCH --job-name=fast-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00

set -e
PROF=/p/rlprojects/RND/07_reconstruction/analysis/2026-06-25-run-profiling
FAST=$PROF/experiments/2026-06-25-19-55-fast-trainer
CODE=$PROF/analysis/code
DATA=$PROF/data
PY=/u/sl5nw/.conda/envs/exploration/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
NODE=${SLURMD_NODENAME:-$(hostname -s)}

echo "[fast-bench] node=$NODE start $(date)"

# 1. SAC-core validation: Pendulum-v1 should rise from ~-1200 toward ~-300 within ~12k steps
$PY $FAST/fast_sac_rnd.py --env Pendulum-v1 --steps 10000 --warmup 3000 --threads 2 --seed 0 --validate \
  --out $DATA/fast_pendulum_${NODE}.json

# 2+3. PointMaze: fast trainer fps AND SB3 baseline fps on the SAME node, per algorithm
for AB in "rnd_state 100" "gt_position_velocity 1" "rnd_elliptical 0.01"; do
  set -- $AB
  $PY $FAST/fast_sac_rnd.py --env pointmaze --algorithm "$1" --beta "$2" --steps 6000 --warmup 1000 \
    --threads 2 --seed 0 --out $DATA/fast_${1}_${NODE}.json
  $PY $CODE/profile_train.py --condition sb3_baseline --algorithm "$1" --beta "$2" --threads 2 \
    --bench_steps 6000 --warmup 1000 --out $DATA/sb3base_${1}_${NODE}.json
done

echo "[fast-bench] node=$NODE done $(date)"
