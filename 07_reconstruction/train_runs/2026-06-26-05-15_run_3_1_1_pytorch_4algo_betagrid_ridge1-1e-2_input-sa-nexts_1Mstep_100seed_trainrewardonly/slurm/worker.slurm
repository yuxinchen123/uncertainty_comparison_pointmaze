#!/bin/bash
# One Train-run-3.1.1 work-queue job = 8 workers (srun ntasks=8). Each worker (worker.py) pulls configs from
# THIS sweep's queue (queue/<SWEEP_ID>/) and runs the live PyTorch trainer train.py (SB3 SAC), logging under
# data/<SWEEP_ID>/local/. SWEEP_ID is injected by the launcher via `sbatch --export=ALL,SWEEP_ID=...`. The
# trainer runs under the exploration conda env (explicit interpreter path, so it does not depend on the
# launching shell's active env). Each worker is pinned to 2 CPUs; cap every thread pool to 2 so 8 workers/node
# do not oversubscribe (RND/SAC training is CPU single-core bound).
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TMPDIR=/tmp
export PROJ_DIR="/p/rlprojects/RND/07_reconstruction"
export RUN_DIR="$PROJ_DIR/train_runs/2026-06-26-05-15_run_3_1_1_pytorch_4algo_betagrid_ridge1-1e-2_input-sa-nexts_1Mstep_100seed_trainrewardonly"
export SWEEP_ID="${SWEEP_ID:?SWEEP_ID must be passed via sbatch --export=ALL,SWEEP_ID=<id>}"
PY="/u/sl5nw/.conda/envs/exploration/bin/python"
srun "$PY" "$RUN_DIR/slurm/worker.py"
