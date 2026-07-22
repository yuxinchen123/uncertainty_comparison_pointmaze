#!/bin/bash
# Collaborator worker job for the 16-core/32-thread node class (adriatic/affogato/ai): sbatch sets
# --ntasks to 16 or 14 (one 16x1 + one 14x1 fills a node's 30 allocatable threads; ntasks is
# capped by PHYSICAL cores, so 32 tasks are rejected on this class). Same shared env + fail-fast
# wrapper as worker_32x1_collab.slurm; srun --wait=0 is MANDATORY (see that file).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR=/tmp
source "/p/rlprojects/RND/07_reconstruction/train_runs/2026-07-20-16-44_run_5_baseline_rnd-next-state__origsmall-mse-mean-lr1e-4-out128-predextra1-leaky0.2-warmupenv6400-rewardnorm-beta1e-2to1e4+0.5-prune30__C2-adam-mse-b100__N1-rewardnorm-b1e4__seed600-899_1Mstep_cpu-nolim/for_collaborator/packet_env.sh"
export RUN_DIR PROJ_DIR
export SWEEP_ID
srun --wait=0 "$PY" "$FC/worker_collab.py"
