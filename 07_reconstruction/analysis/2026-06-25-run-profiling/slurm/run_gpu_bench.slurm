#!/bin/bash
# GPU bold-run: on ONE GPU node, measure the SAME workload device=cpu vs device=cuda (same-node, so the
# CPU-vs-GPU comparison is clean). Confirms whether GPU helps these tiny 256x256 nets today.
#SBATCH --job-name=gpu-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:30:00

set -e
PROF=/p/rlprojects/RND/07_reconstruction/analysis/2026-06-25-run-profiling
CODE=$PROF/analysis/code
DATA=$PROF/data
PY=/u/sl5nw/.conda/envs/exploration/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
NODE=${SLURMD_NODENAME:-$(hostname -s)}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_')
echo "[gpu-bench] node=$NODE gpu=$GPU start $(date)"

for AB in "rnd_state 100" "gt_position_velocity 1" "rnd_elliptical 0.01"; do
  set -- $AB
  # CPU on this GPU node (baseline for the same-node ratio)
  $PY $CODE/profile_train.py --condition gpunode_cpu --device cpu --algorithm "$1" --beta "$2" \
    --threads 2 --bench_steps 5000 --warmup 1000 --out $DATA/gpunode_cpu_${1}_${NODE}.json || echo "CPU $1 failed"
  # CUDA on this GPU node
  $PY $CODE/profile_train.py --condition gpunode_cuda --device cuda --algorithm "$1" --beta "$2" \
    --threads 2 --bench_steps 5000 --warmup 1000 --out $DATA/gpunode_cuda_${1}_${NODE}.json || echo "CUDA $1 failed"
done
echo "[gpu-bench] node=$NODE done $(date)"
