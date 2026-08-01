#!/bin/bash
# Run 8.1.2 task-S CPU worker (pool pending_1m) job = 14 single-cpu worker tasks (for a ~16-core node class, leaving 2 cores of
# headroom; the 16-core adriatic/affogato class rejects 16x1 outright per cluster-slurm.md, so 14 fits).
# cpus-per-task=1 + --ntasks-per-core=2 packs the 14 tasks onto sibling hardware threads so AllocCPUS
# equals the ask. --mem sized ~2G per worker (PointMaze ~630 MB avg / 765 MB peak RSS, AntMaze slightly
# larger) => 14 * 2G = 28G.
# srun --wait=0 is MANDATORY: workers exit at different times as the queue empties, and the default
# WaitTime=3600 would otherwise kill the survivors 1 h after the first worker finishes.
# Partition, --time, --job-name and --export=...,SWEEP_ID=<id> are supplied on the sbatch command line
# (see launch_queue.sh). No --nodelist: this shape fills whatever ~16-core node class is free.
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=2
#SBATCH --mem=28G
#SBATCH --output=/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-01-01-44_run_8_1_2_antmaze-umaze-medium-bottom-left_sac__rnd-origsmall-beta1e4-3e3-10M-100seed__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1M-prune30-race100-frozenbar_gamma0.99/logs/%x_%j.log
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR=/tmp
export WORKER_DEVICE=cpu
export WORKER_POOLS="pending_1m"
export PROJ_DIR="/p/rlprojects/RND/07_reconstruction"
export RUN_DIR="/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-01-01-44_run_8_1_2_antmaze-umaze-medium-bottom-left_sac__rnd-origsmall-beta1e4-3e3-10M-100seed__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1M-prune30-race100-frozenbar_gamma0.99"
export SWEEP_ID="${SWEEP_ID:?SWEEP_ID must be passed via sbatch --export=ALL,SWEEP_ID=<id>}"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"
srun --wait=0 "$PY" "$RUN_DIR/slurm/worker.py"
