#!/bin/bash
# CPU worker job for the run-6 4M extension sweep (ext96h): N worker tasks on ONE node, each
# claiming resumable 96-hour fresh runs from queue/<SWEEP_ID>/pending/ until it is empty.
#
# No --ntasks, --mem, --cpus-per-task, --ntasks-per-core, --partition, --time, --job-name or
# --nodelist here: ext96h_plan_jobs.py sizes them per node from live scontrol numbers (puma01 gets
# 2 CPUs per worker per the 2026-08-13 skill rule) and passes them on the sbatch command line.
#
# srun --wait=0 is MANDATORY (WaitTime=3600 would kill the survivors an hour after the first
# worker exits).
#SBATCH --nodes=1
#SBATCH --output=/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-08-02-00_run_6_pointmaze-large-topright__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1Mstep_seed1200-1499_prune20-race100-winner300_bar-run5-origrnd_cpu-nolim-resv/logs/%x_%j.log
# thread budget follows the per-task CPU grant (1 on most nodes, 2 on puma01)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TMPDIR=/tmp
# PYTHONNOUSERSITE=1 is load-bearing: the owner's ~/.local torch 2.10+cu128 would otherwise shadow
# the shared env's 2.5.1+cu124 for the owner's workers only
export PYTHONNOUSERSITE=1
export PROJ_DIR="/p/rlprojects/RND/07_reconstruction"
export RUN_DIR="/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-08-02-00_run_6_pointmaze-large-topright__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1Mstep_seed1200-1499_prune20-race100-winner300_bar-run5-origrnd_cpu-nolim-resv"
export SWEEP_ID="${SWEEP_ID:?SWEEP_ID must be passed via sbatch --export=ALL,SWEEP_ID=<id>}"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"
srun --wait=0 "$PY" "$RUN_DIR/slurm/ext96h_worker.py"
