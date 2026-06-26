#!/bin/bash
# One Train-run-4 work-queue job = 8 workers (srun ntasks=8). Each worker (worker.py) pulls configs from THIS
# sweep's queue (queue/<SWEEP_ID>/) and runs run4_train.py (fully-JAX/sbx, sample-time RND), logging under
# data/<SWEEP_ID>/local/. SWEEP_ID is injected by the launcher via `sbatch --export=ALL,SWEEP_ID=...`. The JAX
# trainer runs under the dedicated JAX venv (no conda). Each worker is pinned to 2 CPUs; cap every thread pool
# to 2 and force single-threaded XLA:CPU so 8 workers/node don't oversubscribe.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export JAX_PLATFORMS=cpu
export TMPDIR=/tmp
export RUN_DIR="/p/rlprojects/RND/07_reconstruction/train_runs/2026-06-25-21-54_run_4_fully_jax_sampletime_500K_3algo_200seed"
export RUN4_TRAIN="$RUN_DIR/code/run4_train.py"
export SWEEP_ID="${SWEEP_ID:?SWEEP_ID must be passed via sbatch --export=ALL,SWEEP_ID=<id>}"
VENVPY="/p/rlprojects/RND/.venvs/jax_bench/bin/python"
srun "$VENVPY" "$RUN_DIR/slurm/worker.py"
