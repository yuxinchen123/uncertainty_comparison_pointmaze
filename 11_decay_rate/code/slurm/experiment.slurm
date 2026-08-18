#!/bin/bash
# One 11_decay_rate experiment as a Slurm job. Usage (via submit_experiment.sh):
#   sbatch experiment.slurm <out_dir> [extra run_experiment.py args...]
# 20 hardware threads on jaguar03 hold the 20 (env x seed) cells, one single-thread torch
# worker per thread (the measured-good packing: two independent workers per core).
#SBATCH --job-name=11decay_exp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=30G
#SBATCH --gpus-per-node=0
#SBATCH --time=02:00:00

set -euo pipefail
OUT="$1"; shift
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
cd /p/rlprojects/RND/11_decay_rate/code
exec /p/rlprojects/RND/.venvs/exploration/bin/python -u run_experiment.py \
    --out "$OUT" --procs 20 "$@"
