#!/bin/bash
# Collaborator CPU worker job: N single-cpu worker tasks on ONE node, each claiming 1,000,000-step
# runs from the sweep's single pending pool until it is empty.
#
# --ntasks, --mem, --ntasks-per-core, --partition, --time, --nodelist, --job-name and --comment are
# all supplied on the sbatch command line by launch_workers_collaborator.sh, which sizes them per
# node from the live cluster state. One script, every shape.
#
# packet_env.sh is sourced by its ABSOLUTE hardcoded path — NEVER via dirname "$0", which under
# sbatch resolves to the Slurm spool copy /var/spool/slurmd/job<id>/ and kills every job one second
# in (a real incident on an earlier packet).
#
# srun --wait=0 is MANDATORY: workers exit at different times as the pool drains and as the owner's
# controller truncates configurations, and this cluster's WaitTime=3600 would otherwise kill the
# survivors one hour after the first worker finishes.
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-05-16-05_run_5_8_1_2_addendum_rnd-origsmall-adam-lr1e-3__pointmaze-large-topright-seed900-999__antmaze-umaze-medium-bottom-left-seed0-99__beta-1e-3-to-1e4-x15_1Mstep-100seed_prune30-race100-frozenbar-vs-adam-lr1e-4_cpu-nolim/for_collaborator/logs/%x_%j.log
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR=/tmp
# every worker of this sweep must run the SHARED env's torch, whoever submitted it (a user site
# directory ahead of the env on sys.path would otherwise supply a different torch build)
export PYTHONNOUSERSITE=1
source "/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-05-16-05_run_5_8_1_2_addendum_rnd-origsmall-adam-lr1e-3__pointmaze-large-topright-seed900-999__antmaze-umaze-medium-bottom-left-seed0-99__beta-1e-3-to-1e4-x15_1Mstep-100seed_prune30-race100-frozenbar-vs-adam-lr1e-4_cpu-nolim/for_collaborator/packet_env.sh"
export RUN_DIR PROJ_DIR
export SWEEP_ID
srun --wait=0 "$PY" "$FC/worker_collab.py"
