#!/bin/bash
# JAX vs torch A/B on the EXACT run-2 task (PointMaze_Large + RND), one node, single clean process.
# sb3 = exploration env (torch 2.10, the run-2 stack); sbx = shared jax venv (JAX SAC + torch RND).
# 3 seeds of the with-RND A/B (the full run-2 task) + 1 extrinsic-only pair (the pure-SAC speedup).
#SBATCH --job-name=jax-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00

set -e
PROF=/p/rlprojects/RND/07_reconstruction/analysis/2026-06-25-run-profiling
SCRIPT=$PROF/experiments/2026-06-25-21-40-jax-ab/sac_pointmaze_bench.py
DATA=$PROF/data
VENVPY=/p/rlprojects/RND/.venvs/jax_bench/bin/python
EXPLPY=/u/sl5nw/.conda/envs/exploration/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
NODE=${SLURMD_NODENAME:-$(hostname -s)}
echo "[jax-ab] node=$NODE start $(date)"

# full run-2 task (PointMaze + step-time RND): SB3 (torch) vs sbx (JAX), 3 seeds, same node
for s in 0 1 2; do
  $EXPLPY $SCRIPT --backend sb3 --rnd --steps 15000 --warmup 2000 --threads 2 --seed $s --out $DATA/jaxab_sb3_rnd_${NODE}_s${s}.json
  $VENVPY $SCRIPT --backend sbx --rnd --steps 15000 --warmup 2000 --threads 2 --seed $s --out $DATA/jaxab_sbx_rnd_${NODE}_s${s}.json
done
# extrinsic-only (pure SAC speedup, no RND), seed 0
$EXPLPY $SCRIPT --backend sb3 --steps 15000 --warmup 2000 --threads 2 --seed 0 --out $DATA/jaxab_sb3_ext_${NODE}.json
$VENVPY $SCRIPT --backend sbx --steps 15000 --warmup 2000 --threads 2 --seed 0 --out $DATA/jaxab_sbx_ext_${NODE}.json
echo "[jax-ab] node=$NODE done $(date)"
