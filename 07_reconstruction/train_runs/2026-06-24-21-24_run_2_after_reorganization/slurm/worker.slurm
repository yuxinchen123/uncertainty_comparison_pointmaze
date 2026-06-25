#!/bin/bash
# One work-queue job = 8 workers (srun ntasks=8). Each worker (worker.py) pulls configs from the local
# file queue and runs train.py. 64 such jobs => 512 workers but only 64 slurm IDs in squeue.
# NO `wandb agent` — config distribution is the local queue. The launcher sets all resources
# (partition, nodelist, reservation, job-name, ntasks=8, cpus-per-task=2, --mem-per-cpu, output, chdir).
# Convention: the launching shell must already have the `exploration` conda env active (so `python` is
# the env's python on every task).
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export WANDB__SERVICE_WAIT=300
export TMPDIR=/tmp
export WANDB_DIR="/tmp/wandb_${SLURM_JOB_ID:-$$}"
mkdir -p "$WANDB_DIR"
export PROJ_DIR="/p/rlprojects/RND/07_reconstruction"
export RUN_DIR="$PROJ_DIR/train_runs/2026-06-24-21-24_run_2_after_reorganization"
srun python "$RUN_DIR/slurm/worker.py"
