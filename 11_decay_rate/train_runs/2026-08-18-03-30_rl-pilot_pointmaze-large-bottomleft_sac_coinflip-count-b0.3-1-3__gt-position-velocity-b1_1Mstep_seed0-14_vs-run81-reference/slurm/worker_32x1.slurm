#!/bin/bash
# RL-pilot CPU worker job = 32 single-cpu worker tasks (shape and flags per the shared
# uva-submit-cpu-sweep skill: --ntasks-per-core=2 packs tasks onto sibling threads so AllocCPUS
# equals the ask; srun --wait=0 is mandatory because workers exit at different times as the
# queue empties and WaitTime=3600 would kill the survivors an hour after the first exit).
# Partition, --time, --job-name and --export=ALL,SWEEP_ID=<id> come from the sbatch command line.
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=2
#SBATCH --mem=64G
#SBATCH --output=/p/rlprojects/RND/11_decay_rate/train_runs/2026-08-18-03-30_rl-pilot_pointmaze-large-bottomleft_sac_coinflip-count-b0.3-1-3__gt-position-velocity-b1_1Mstep_seed0-14_vs-run81-reference/logs/%x_%j.log
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export TMPDIR=/tmp
export WORKER_DEVICE=cpu
export PROJ_DIR="/p/rlprojects/RND/07_reconstruction"
export RUN_DIR="/p/rlprojects/RND/11_decay_rate/train_runs/2026-08-18-03-30_rl-pilot_pointmaze-large-bottomleft_sac_coinflip-count-b0.3-1-3__gt-position-velocity-b1_1Mstep_seed0-14_vs-run81-reference"
export SWEEP_ID="${SWEEP_ID:?SWEEP_ID must be passed via sbatch --export=ALL,SWEEP_ID=<id>}"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"
srun --wait=0 "$PY" "$RUN_DIR/slurm/worker.py"
