#!/bin/bash
# Collaborator worker job for Run 8.1 = 30 single-cpu worker tasks. Shape 30x1 targets a ~32-core
# node (2 cores of headroom); --cpus-per-task=1 + --ntasks-per-core=2 packs the 30 tasks onto sibling
# hardware threads so AllocCPUS equals the ask (see cluster-slurm.md hyperthreading note).
# --mem-per-cpu=2G => 30 workers * 2G = 60G (PointMaze ~630 MB avg / 765 MB peak RSS; AntMaze larger).
# Differences from the owner's slurm/worker_30x1.slurm: the SHARED env python + the fail-fast wrapper
# worker_collab.py (reports a broken environment instead of silently failing configs), and an explicit
# WORKER_DEVICE=cpu (collaborators run CPU-only — never a GPU job).
# packet_env.sh is sourced by its ABSOLUTE hardcoded path — NEVER via dirname "$0", which under sbatch
# resolves to the /var/spool/slurmd/job<id>/ spool copy and kills the job at t=1s (common_problems #1).
# srun --wait=0 is MANDATORY: workers exit at DIFFERENT times as the queue drains / the owner prunes,
# and the default WaitTime=3600 would otherwise kill the survivors 1 h after the first worker exits.
# launch_workers_collaborator.sh supplies --job-name/--partition/--time/--comment/--export and may
# OVERRIDE --ntasks (fragment conversion) and --output on the command line; the #SBATCH defaults below
# also let a bare `sbatch worker_30x1_collab.slurm --partition=cpu --job-name=<prefix>` work.
#SBATCH --nodes=1
#SBATCH --ntasks=30
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
