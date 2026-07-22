#!/bin/bash
# Collaborator worker job = 32 workers (srun ntasks=32, cpus-per-task=1; sbatch adds
# --ntasks-per-core=2 so tasks pack onto sibling hardware threads and AllocCPUS equals the ask).
# Differences from the owner's worker_32x1.slurm: the SHARED env python (packet_env.sh), and the
# fail-fast wrapper worker_collab.py (reports a broken env instead of silently failing configs).
# srun --wait=0 is MANDATORY: workers exit at DIFFERENT times by design (the controller prunes
# pending work), and without it srun kills every remaining worker 3600 s after the first exits.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR=/tmp
source "/p/rlprojects/RND/07_reconstruction/train_runs/2026-07-11-00-27_run_3_2_3_rnd-next-state_rewardnorm-adam-mse-gamma0.99_beta-1e-4-to-1e4__bias-pytorch-default-normal0.5_adam-l2_sgd1t-l2_eta0-1e-3-1e-2-1e-1_t0-1e3-1e4_beta-1e-5-to-1e4_1Mstep_seed0-99_optuna-bar-34.50_32x1/for_collaborator/packet_env.sh"
export RUN_DIR PROJ_DIR
export SWEEP_ID
srun --wait=0 "$PY" "$FC/worker_collab.py"
