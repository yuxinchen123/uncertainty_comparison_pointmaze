#!/bin/bash
# COLLABORATOR worker job for the run-6 96-hour extension sweep (ext96h). Identical to the owner's
# slurm/ext96h_worker_cpu.slurm except the log location (your for_collaborator/logs/) — the worker
# code itself is shared. Submit through ext96h_launch_workers_collaborator.sh, never by hand.
#SBATCH --nodes=1
#SBATCH --output=/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-08-02-00_run_6_pointmaze-large-topright__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1Mstep_seed1200-1499_prune20-race100-winner300_bar-run5-origrnd_cpu-nolim-resv/for_collaborator/logs/%x_%j.log
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export TMPDIR=/tmp
export PYTHONNOUSERSITE=1
export PROJ_DIR="/p/rlprojects/RND/07_reconstruction"
export RUN_DIR="/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-08-02-00_run_6_pointmaze-large-topright__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1Mstep_seed1200-1499_prune20-race100-winner300_bar-run5-origrnd_cpu-nolim-resv"
export SWEEP_ID="${SWEEP_ID:?SWEEP_ID must be passed via sbatch --export=ALL,SWEEP_ID=<id>}"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"
srun --wait=0 "$PY" "$RUN_DIR/slurm/ext96h_worker.py"
