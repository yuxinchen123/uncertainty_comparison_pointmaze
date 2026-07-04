#!/bin/bash
# Run 3.2.1 worker job = 32 workers (srun ntasks=32, cpus-per-task=1; sbatch adds --ntasks-per-core=2
# so tasks pack onto sibling hardware threads and AllocCPUS equals the ask — slurm-submission.md §7).
# Every thread pool capped at 1 (train.py also sets torch threads from OMP_NUM_THREADS), so 32
# single-cpu workers never oversubscribe the 32 allocated hardware threads.
# SWEEP_ID is injected via `sbatch --export=ALL,SWEEP_ID=...`.
# srun --wait=0 is MANDATORY (slurm-submission.md §9, verified 2026-07-04): in this sweep workers
# exit at DIFFERENT times by design (the optuna controller prunes pending work), and without
# --wait=0 srun kills every remaining worker 3600 s after the first one exits — mid-flight
# training runs included. 0 = unlimited wait / never kill.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR=/tmp
export PROJ_DIR="/p/rlprojects/RND/07_reconstruction"
export RUN_DIR="$PROJ_DIR/train_runs/2026-07-04-17-41_run_3_2_1_rnd-next-state_adam-l2_adagrad-mse-l2_sgd1t-mse-l2_eta0-1e-3-3e-3-1e-2-3e-2-1e-1_t0-1e3-1e4_beta-1e-3-to-1e4_1Mstep_seed0-49_optuna-bar-36.95_32x1"
export SWEEP_ID="${SWEEP_ID:?SWEEP_ID must be passed via sbatch --export=ALL,SWEEP_ID=<id>}"
PY="/u/sl5nw/.conda/envs/exploration/bin/python"
srun --wait=0 "$PY" "$RUN_DIR/slurm/worker.py"
