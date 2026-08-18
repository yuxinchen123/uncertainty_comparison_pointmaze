#!/bin/bash
# One 11_decay_rate GPU experiment (image domains): 1 GPU, 4 worker processes sharing it
# (the conv distillation net is tiny; 4 cells fit one card comfortably). Node/partition and
# --time come from the sbatch command line per the GPU submission skill.
#SBATCH --job-name=11decay_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
set -euo pipefail
OUT="$1"; shift
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export METHOD_DEVICE=cuda
export CELL_TIME_CAP=3600
cd /p/rlprojects/RND/11_decay_rate/code
exec /p/rlprojects/RND/.venvs/exploration/bin/python -u run_experiment.py \
    --out "$OUT" --procs 4 "$@"
