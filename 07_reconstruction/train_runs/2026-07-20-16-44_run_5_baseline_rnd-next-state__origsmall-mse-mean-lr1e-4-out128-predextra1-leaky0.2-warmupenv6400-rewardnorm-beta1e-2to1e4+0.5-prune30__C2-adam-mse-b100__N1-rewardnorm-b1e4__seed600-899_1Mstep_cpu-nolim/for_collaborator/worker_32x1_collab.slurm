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
source "/p/rlprojects/RND/07_reconstruction/train_runs/2026-07-20-16-44_run_5_baseline_rnd-next-state__origsmall-mse-mean-lr1e-4-out128-predextra1-leaky0.2-warmupenv6400-rewardnorm-beta1e-2to1e4+0.5-prune30__C2-adam-mse-b100__N1-rewardnorm-b1e4__seed600-899_1Mstep_cpu-nolim/for_collaborator/packet_env.sh"
export RUN_DIR PROJ_DIR
export SWEEP_ID
srun --wait=0 "$PY" "$FC/worker_collab.py"
