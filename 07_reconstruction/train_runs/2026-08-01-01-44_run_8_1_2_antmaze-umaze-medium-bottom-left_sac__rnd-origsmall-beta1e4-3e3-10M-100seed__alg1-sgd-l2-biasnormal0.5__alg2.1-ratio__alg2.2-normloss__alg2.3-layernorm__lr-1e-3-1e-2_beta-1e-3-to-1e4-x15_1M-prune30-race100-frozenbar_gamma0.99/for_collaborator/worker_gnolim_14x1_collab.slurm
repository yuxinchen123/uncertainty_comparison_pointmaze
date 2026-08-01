#!/bin/bash
# Collaborator gnolim worker job for Run 8.1.2 = 14 single-cpu worker tasks on a gnolim node of the
# 16-core / 32-thread class (ai05, ai07, ai08, ai09, ai10 — 30 allocatable threads), leaving headroom.
# --gpus-per-node=0 is MANDATORY: gnolim's Pascal/Maxwell GPUs cannot run torch 2.10+cu128, so for
# this run gnolim is a CPU pool. WORKER_DEVICE=cpu — never a GPU-using job.
#
# CLAIM ORDER: WORKER_POOLS="pending_10m pending_1m" — the 10M task-R baseline runs FIRST, the 1M
# task-S runs as fallback. Nothing to configure: the run's own worker.py holds a walltime guard and
# allows a 10M claim only when the job has >= 170 h of walltime LEFT (a 10M cpu run measures ~154 h,
# and this run has no checkpoints — a run that cannot finish inside its job must never start). So:
#   - a job submitted with a SHORT --time (e.g. before the 2026-08-05 maintenance window) skips the
#     10M pool automatically and serves task S from pending_1m;
#   - a 20-day job submitted after the maintenance takes 10M runs first, then drains into task S.
# The launcher sets --time for you; you never set it by hand.
#
# --cpus-per-task=1 + --ntasks-per-core=2 packs the 14 tasks onto sibling hardware threads so
# AllocCPUS equals the ask. --mem-per-cpu=2G => 14 * 2G = 28G.
# packet_env.sh is sourced by its ABSOLUTE hardcoded path — NEVER via dirname "$0", which under sbatch
# resolves to the /var/spool/slurmd/job<id>/ spool copy and kills the job at t=1s (common_problems #1).
# srun --wait=0 is MANDATORY (workers exit at different times as the pools drain / the owner prunes).
#SBATCH --nodes=1
#SBATCH --ntasks=14
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
