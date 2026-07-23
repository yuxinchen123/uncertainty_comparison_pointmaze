#!/bin/bash
# Collaborator worker job for Run 8.1 = 14 single-cpu worker tasks. Shape 14x1 targets a ~16-core
# node class (2 cores of headroom); the 16-core adriatic/affogato class REJECTS 16x1 outright
# (cluster-slurm.md), so 14 is the largest that schedules there. --cpus-per-task=1 +
# --ntasks-per-core=2 packs the 14 tasks onto sibling hardware threads so AllocCPUS equals the ask.
# --mem-per-cpu=2G => 14 * 2G = 28G. Same shared env + fail-fast wrapper + WORKER_DEVICE=cpu as
# worker_30x1_collab.slurm; see that file for the full notes.
# packet_env.sh is sourced by its ABSOLUTE hardcoded path — NEVER via dirname "$0" (common_problems #1).
# srun --wait=0 is MANDATORY (workers exit at different times as the queue drains / owner prunes).
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=2
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/p/rlprojects/RND/07_reconstruction/train_runs/2026-07-23-01-35_run_8_1_pointmaze-antmaze-8env-start-bottom-left_sac__rnd-origsmall__gt-position-velocity__gt-position-maze-cell__gt-position-1m__beta-1e-3-to-1e4-x15_prune20-winner100-max300_1Mstep_default-max-episode_reward-shift-1_gamma0.99/for_collaborator/logs/%x_%j.log
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR=/tmp
export WORKER_DEVICE=cpu
source "/p/rlprojects/RND/07_reconstruction/train_runs/2026-07-23-01-35_run_8_1_pointmaze-antmaze-8env-start-bottom-left_sac__rnd-origsmall__gt-position-velocity__gt-position-maze-cell__gt-position-1m__beta-1e-3-to-1e4-x15_prune20-winner100-max300_1Mstep_default-max-episode_reward-shift-1_gamma0.99/for_collaborator/packet_env.sh"
export RUN_DIR PROJ_DIR
export SWEEP_ID
srun --wait=0 "$PY" "$FC/worker_collab.py"
