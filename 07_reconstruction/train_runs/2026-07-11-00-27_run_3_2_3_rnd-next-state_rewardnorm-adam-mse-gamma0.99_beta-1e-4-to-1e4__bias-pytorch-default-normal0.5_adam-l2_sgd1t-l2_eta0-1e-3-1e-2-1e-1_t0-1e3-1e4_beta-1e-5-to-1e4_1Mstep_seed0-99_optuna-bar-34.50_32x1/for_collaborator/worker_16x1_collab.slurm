#!/bin/bash
# Collaborator worker job for the 16-core/32-thread node class (adriatic/affogato/ai): sbatch sets
# --ntasks to 16 or 14 (one 16x1 + one 14x1 fills a node's 30 allocatable threads; ntasks is
# capped by PHYSICAL cores, so 32 tasks are rejected on this class). Same shared env + fail-fast
# wrapper as worker_32x1_collab.slurm; srun --wait=0 is MANDATORY (see that file).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR=/tmp
source "/p/rlprojects/RND/07_reconstruction/train_runs/2026-07-11-00-27_run_3_2_3_rnd-next-state_rewardnorm-adam-mse-gamma0.99_beta-1e-4-to-1e4__bias-pytorch-default-normal0.5_adam-l2_sgd1t-l2_eta0-1e-3-1e-2-1e-1_t0-1e3-1e4_beta-1e-5-to-1e4_1Mstep_seed0-99_optuna-bar-34.50_32x1/for_collaborator/packet_env.sh"
export RUN_DIR PROJ_DIR
export SWEEP_ID
srun --wait=0 "$PY" "$FC/worker_collab.py"
