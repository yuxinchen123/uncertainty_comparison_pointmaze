#!/bin/bash
# CPU worker job for the Adam learning-rate 1e-3 addendum: N single-cpu worker tasks on ONE node,
# each claiming 1,000,000-step runs from queue/<SWEEP_ID>/pending/ until it is empty.
#
# This script deliberately carries NO --ntasks, --mem, --ntasks-per-core, --partition, --time,
# --job-name or --nodelist: plan_jobs.py sizes every one of those per node from the live
# scontrol numbers (physical cores, free allocatable threads, real memory, the per-user pool cap)
# and passes them on the sbatch command line, which overrides any #SBATCH directive. One script,
# every shape.
#
# srun --wait=0 is MANDATORY: workers exit at different times as the queue empties, and this
# cluster's WaitTime=3600 would otherwise kill the survivors one hour after the first worker
# finishes (94 mid-flight runs were lost that way on an earlier sweep).
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-05-16-05_run_5_8_1_2_addendum_rnd-origsmall-adam-lr1e-3__pointmaze-large-topright-seed900-999__antmaze-umaze-medium-bottom-left-seed0-99__beta-1e-3-to-1e4-x15_1Mstep-100seed_prune30-race100-frozenbar-vs-adam-lr1e-4_cpu-nolim/logs/%x_%j.log
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR=/tmp
# PYTHONNOUSERSITE=1 is load-bearing, not hygiene: the owner's ~/.local/lib/python3.11/site-packages
# sits AHEAD of the shared env on sys.path and supplies torch 2.10.0+cu128, while a collaborator's
# uid has no such directory and gets the env's own torch 2.5.1+cu124. Without this line the owner's
# workers and a collaborator's workers in the SAME sweep run different torch builds. With it, every
# worker of this sweep runs the env's 2.5.1+cu124 whoever submitted it.
export PYTHONNOUSERSITE=1
export PROJ_DIR="/p/rlprojects/RND/07_reconstruction"
export RUN_DIR="/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-05-16-05_run_5_8_1_2_addendum_rnd-origsmall-adam-lr1e-3__pointmaze-large-topright-seed900-999__antmaze-umaze-medium-bottom-left-seed0-99__beta-1e-3-to-1e4-x15_1Mstep-100seed_prune30-race100-frozenbar-vs-adam-lr1e-4_cpu-nolim"
export SWEEP_ID="${SWEEP_ID:?SWEEP_ID must be passed via sbatch --export=ALL,SWEEP_ID=<id>}"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"
srun --wait=0 "$PY" "$RUN_DIR/slurm/worker.py"
