#!/bin/bash
# Collaborator worker job for the 16-core/32-thread node class (adriatic/affogato/ai): sbatch sets
# --ntasks to 16 or 14 (one 16x1 + one 14x1 fills a node's 30 allocatable threads; ntasks is capped
# by PHYSICAL cores, so 32 tasks are rejected on this class). Same shared env + fail-fast wrapper as
# worker_32x1_collab.slurm; srun --wait=0 is MANDATORY (see that file).
#
# packet_env.sh is sourced by its ABSOLUTE hardcoded path — NEVER via dirname "$0": under sbatch,
# $0 is the spooled copy /var/spool/slurmd/job<id>/slurm_script, so dirname "$0" would point at the
# spool dir and the source would fail at t=1s (collab-handbook common problem #1, 2026-07-16).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR=/tmp
source "/p/rlprojects/RND/07_reconstruction/train_runs/2026-07-19-03-29_convergence_run_1_rnd-next-state-distill_env-center1x1-topright1x1-cellmid9x12_init-zero-ptbias-ptfull-normal0.5_adam-lr1e-3-1e-4-1e-5_sgd1t-eta0-1e-3-1e-2-1e-1_t0-1e2-1e3-1e4_4096step_seed0-29_32x1/for_collaborator/packet_env.sh"
export RUN_DIR PROJ_DIR
export SWEEP_ID
srun --wait=0 "$PY" "$FC/worker_collab.py"
