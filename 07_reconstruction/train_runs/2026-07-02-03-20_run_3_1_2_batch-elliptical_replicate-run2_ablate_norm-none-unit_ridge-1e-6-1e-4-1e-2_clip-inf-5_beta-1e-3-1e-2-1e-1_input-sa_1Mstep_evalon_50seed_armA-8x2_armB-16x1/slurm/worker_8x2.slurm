#!/bin/bash
# ARM A job = 8 workers (srun ntasks=8, cpus-per-task=2). Each worker pulls configs from THIS arm's queue
# (queue/<SWEEP_ID>/) and runs train.py under the exploration env, logging under data/<SWEEP_ID>/local/.
# Thread pools capped at 2 to match cpus-per-task (train.py also calls torch.set_num_threads from
# OMP_NUM_THREADS). SWEEP_ID is injected via `sbatch --export=ALL,SWEEP_ID=...`.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TMPDIR=/tmp
export PROJ_DIR="/p/rlprojects/RND/07_reconstruction"
export RUN_DIR="$PROJ_DIR/train_runs/2026-07-02-03-20_run_3_1_2_batch-elliptical_replicate-run2_ablate_norm-none-unit_ridge-1e-6-1e-4-1e-2_clip-inf-5_beta-1e-3-1e-2-1e-1_input-sa_1Mstep_evalon_50seed_armA-8x2_armB-16x1"
export SWEEP_ID="${SWEEP_ID:?SWEEP_ID must be passed via sbatch --export=ALL,SWEEP_ID=<id>}"
PY="/u/sl5nw/.conda/envs/exploration/bin/python"
srun "$PY" "$RUN_DIR/slurm/worker.py"
