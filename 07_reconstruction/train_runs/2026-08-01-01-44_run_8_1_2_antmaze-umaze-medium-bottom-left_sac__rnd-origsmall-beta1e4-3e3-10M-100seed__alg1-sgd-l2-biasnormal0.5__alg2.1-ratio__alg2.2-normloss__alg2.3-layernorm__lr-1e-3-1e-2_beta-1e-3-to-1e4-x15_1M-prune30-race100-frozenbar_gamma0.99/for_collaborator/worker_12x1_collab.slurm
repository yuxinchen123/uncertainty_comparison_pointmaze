#!/bin/bash
# Collaborator TASK-S worker job for Run 8.1.2 = 12 single-cpu worker tasks (pool pending_1m only).
# Shape 12x1 targets a ~14-core node class (2 cores of headroom), or a small fragment during the
# fragment-conversion procedure (submit-cpu-sweep section 10 / common_problems #6). --cpus-per-task=1
# + --ntasks-per-core=2 packs the 12 tasks onto sibling hardware threads so AllocCPUS equals the ask.
# --mem-per-cpu=2G => 12 * 2G = 24G.
# Same shared env + fail-fast wrapper + WORKER_DEVICE=cpu + WORKER_POOLS="pending_1m" as
# worker_30x1_collab.slurm; see that file for the full notes.
# packet_env.sh is sourced by its ABSOLUTE hardcoded path — NEVER via dirname "$0" (common_problems #1).
# srun --wait=0 is MANDATORY (workers exit at different times as the pool drains / the owner prunes).
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=2
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-01-01-44_run_8_1_2_antmaze-umaze-medium-bottom-left_sac__rnd-origsmall-beta1e4-3e3-10M-100seed__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1M-prune30-race100-frozenbar_gamma0.99/for_collaborator/logs/%x_%j.log
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR=/tmp
export WORKER_DEVICE=cpu
export WORKER_POOLS="pending_1m"
source "/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-01-01-44_run_8_1_2_antmaze-umaze-medium-bottom-left_sac__rnd-origsmall-beta1e4-3e3-10M-100seed__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1M-prune30-race100-frozenbar_gamma0.99/for_collaborator/packet_env.sh"
export RUN_DIR PROJ_DIR
export SWEEP_ID
srun --wait=0 "$PY" "$FC/worker_collab.py"
