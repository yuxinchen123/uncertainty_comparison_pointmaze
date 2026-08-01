#!/bin/bash
# Run 8.1.2 gnolim CPU worker job = 14 single-cpu tasks on a gnolim node (16-core / 32-thread class;
# --gpus-per-node=0 because gnolim's Pascal/Maxwell GPUs cannot run torch 2.10+cu128).
# CLAIM ORDER (the user rule): pending_10m FIRST, pending_1m as fallback — worker.py's walltime
# guard requires >= 170 h of remaining job walltime for a cpu 10M claim (a 10M cpu run is ~154 h),
# so a job submitted with a short --time (before a maintenance window) automatically serves task S,
# and a 20-day job takes task-R 10M runs first, then drains into task S.
# srun --wait=0 is MANDATORY. Partition (gnolim), --time, --job-name and
# --export=ALL,SWEEP_ID=<id> come on the sbatch command line.
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=2
#SBATCH --gpus-per-node=0
#SBATCH --mem=28G
#SBATCH --output=/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-01-01-44_run_8_1_2_antmaze-umaze-medium-bottom-left_sac__rnd-origsmall-beta1e4-3e3-10M-100seed__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1M-prune30-race100-frozenbar_gamma0.99/logs/%x_%j.log
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR=/tmp
export WORKER_DEVICE=cpu
export WORKER_POOLS="pending_10m pending_1m"
export PROJ_DIR="/p/rlprojects/RND/07_reconstruction"
export RUN_DIR="/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-01-01-44_run_8_1_2_antmaze-umaze-medium-bottom-left_sac__rnd-origsmall-beta1e4-3e3-10M-100seed__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1M-prune30-race100-frozenbar_gamma0.99"
export SWEEP_ID="${SWEEP_ID:?SWEEP_ID must be passed via sbatch --export=ALL,SWEEP_ID=<id>}"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"
srun --wait=0 "$PY" "$RUN_DIR/slurm/worker.py"
