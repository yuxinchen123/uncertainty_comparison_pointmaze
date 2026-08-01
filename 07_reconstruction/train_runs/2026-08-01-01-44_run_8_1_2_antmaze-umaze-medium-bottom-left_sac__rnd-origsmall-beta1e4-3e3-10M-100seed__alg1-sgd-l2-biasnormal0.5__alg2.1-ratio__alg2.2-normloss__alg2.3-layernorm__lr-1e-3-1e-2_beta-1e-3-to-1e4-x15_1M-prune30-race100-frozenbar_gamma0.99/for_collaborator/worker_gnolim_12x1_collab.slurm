#!/bin/bash
# Collaborator gnolim worker job for Run 8.1.2 = 12 single-cpu worker tasks on a gnolim node of the
# 12-core / 24-thread class (jinx01, jinx02, titanx03 — 22 allocatable threads). sbatch validates
# --ntasks against the node's PHYSICAL CORE count, so 12 is the largest shape this class accepts.
# --gpus-per-node=0 is MANDATORY: gnolim's Pascal/Maxwell GPUs cannot run torch 2.10+cu128, so for
# this run gnolim is a CPU pool. WORKER_DEVICE=cpu — never a GPU-using job.
#
# CLAIM ORDER: WORKER_POOLS="pending_10m pending_1m" — the 10M task-R baseline runs FIRST, the 1M
# task-S runs as fallback; the run's own worker.py allows a 10M claim only with >= 170 h of REMAINING
# job walltime (a 10M cpu run measures ~154 h; no checkpoints in this run). A short-walltime job
# therefore serves task S automatically. See worker_gnolim_14x1_collab.slurm for the full note.
#
# --cpus-per-task=1 + --ntasks-per-core=2 packs the 12 tasks onto sibling hardware threads so
# AllocCPUS equals the ask. --mem-per-cpu=2G => 12 * 2G = 24G.
# packet_env.sh is sourced by its ABSOLUTE hardcoded path — NEVER via dirname "$0" (common_problems #1).
# srun --wait=0 is MANDATORY (workers exit at different times as the pools drain / the owner prunes).
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=2
#SBATCH --gpus-per-node=0
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-01-01-44_run_8_1_2_antmaze-umaze-medium-bottom-left_sac__rnd-origsmall-beta1e4-3e3-10M-100seed__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1M-prune30-race100-frozenbar_gamma0.99/for_collaborator/logs/%x_%j.log
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR=/tmp
export WORKER_DEVICE=cpu
export WORKER_POOLS="pending_10m pending_1m"
source "/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-01-01-44_run_8_1_2_antmaze-umaze-medium-bottom-left_sac__rnd-origsmall-beta1e4-3e3-10M-100seed__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1M-prune30-race100-frozenbar_gamma0.99/for_collaborator/packet_env.sh"
export RUN_DIR PROJ_DIR
export SWEEP_ID
srun --wait=0 "$PY" "$FC/worker_collab.py"
