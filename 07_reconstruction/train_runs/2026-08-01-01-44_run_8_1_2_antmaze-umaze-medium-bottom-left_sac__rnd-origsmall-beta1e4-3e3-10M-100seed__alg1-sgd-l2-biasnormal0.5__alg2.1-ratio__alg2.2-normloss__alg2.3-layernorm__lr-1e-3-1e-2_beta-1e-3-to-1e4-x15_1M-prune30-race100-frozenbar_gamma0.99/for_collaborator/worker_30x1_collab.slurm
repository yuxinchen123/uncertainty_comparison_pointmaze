#!/bin/bash
# Collaborator TASK-S worker job for Run 8.1.2 = 30 single-cpu worker tasks (pool pending_1m only).
# Shape 30x1 targets a ~32-core node (2 cores of headroom); --cpus-per-task=1 + --ntasks-per-core=2
# packs the 30 tasks onto sibling hardware threads so AllocCPUS equals the ask (see cluster-slurm.md
# hyperthreading note). --mem-per-cpu=2G => 30 workers * 2G = 60G (measured ~1.0-1.2 GiB per worker
# at 1M steps on AntMaze).
# WORKER_POOLS="pending_1m": this shape claims ONLY the 1M-step task-S runs. The 10M task-R pool is
# served by the gnolim shapes (and by the owner's cuda jobs) — a cpu job on a 4-day walltime could
# never finish a 10M run (~154 h), and this run has no checkpoints.
# WORKER_DEVICE=cpu: collaborators never submit a GPU-using job.
# packet_env.sh is sourced by its ABSOLUTE hardcoded path — NEVER via dirname "$0", which under sbatch
# resolves to the /var/spool/slurmd/job<id>/ spool copy and kills the job at t=1s (common_problems #1).
# srun --wait=0 is MANDATORY: workers exit at DIFFERENT times as the pool drains / the owner prunes,
# and the default WaitTime=3600 would otherwise kill the survivors 1 h after the first worker exits.
# launch_workers_collaborator.sh supplies --job-name/--partition/--time/--comment/--export and may
# OVERRIDE --ntasks (fragment conversion) on the command line; the #SBATCH defaults below also let a
# bare submission work — options must come BEFORE the script:
#   sbatch --partition=cpu --job-name=<prefix> --time=<t> --export=ALL,SWEEP_ID=<id> worker_30x1_collab.slurm
#SBATCH --nodes=1
#SBATCH --ntasks=30
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
