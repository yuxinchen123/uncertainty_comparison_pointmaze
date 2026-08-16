#!/bin/bash
# The settings the copies-per-worker sweep does not cover, measured the same way, so that every
# row of the report's processor tables is a sustained measurement and none has to be marked as an
# opening burst.
#
# Two groups. The independent-process table has two worker counts the sweep leaves out (8 and 32
# workers, one copy each). The thread table is entirely five-iteration measurements; one process
# holding every copy has no worker-start stagger, so only the clock-boost half of the correction
# applies there, but it applies, and 150 iterations removes it.
#SBATCH --job-name=rnd09_cpu_plateau_gaps
#SBATCH --partition=gpu
#SBATCH --nodelist=jaguar03
#SBATCH --exclusive
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=112
#SBATCH --ntasks=1
#SBATCH --mem=960G
#SBATCH --time=4:00:00
#SBATCH --output=/p/rlprojects/RND/09_parallelization/analysis/2026-08-15-17-05_jaguar03-copies-per-worker-plateau/slurm/gaps_%j.out
set -euo pipefail

RUN=/p/rlprojects/RND/09_parallelization/analysis/2026-08-15-17-05_jaguar03-copies-per-worker-plateau
PY=/p/rlprojects/RND/.venvs/exploration/bin/python
LADDER=$RUN/code/bench_copies_per_worker.py
BENCH=/p/rlprojects/RND/09_parallelization/benchmarks/bench_train_cpu.py
export PYTHONNOUSERSITE=1

echo "host $(hostname), $(nproc) processors visible, started $(date)"

# --- the two worker counts the sweep leaves out, measured with the same barrier and the same
# --- ninety-second window as every rung of the sweep
$PY -u $LADDER --copies 1 --procs-list 8 32 --styles full_batch epoch_minibatch \
  --tag _gaps_j3

# --- the thread table, sixteen updates per batch, at both thread settings
$PY -u $BENCH --mode threads --copies 1 2 4 8 16 32 --threads 8 --iters 150 --warmup 5 \
  --tag _sustained_j3_t8
$PY -u $BENCH --mode threads --copies 64 128 --threads 8 --iters 60 --warmup 3 \
  --tag _sustained_j3_t8_big
$PY -u $BENCH --mode threads --copies 1 2 4 8 16 32 --threads 112 --iters 150 --warmup 5 \
  --tag _sustained_j3_t112
$PY -u $BENCH --mode threads --copies 64 128 --threads 112 --iters 60 --warmup 3 \
  --tag _sustained_j3_t112_big

# --- and the thread settings the best-setup table draws on, one update per batch
$PY -u $BENCH --mode threads --copies 1 8 32 64 --threads 8 --style full_batch \
  --iters 100 --warmup 5 --tag _sustained_j3_t8_styleA
$PY -u $BENCH --mode threads --copies 128 --threads 8 --style full_batch \
  --iters 60 --warmup 3 --tag _sustained_j3_t8_styleA_big

echo "CPU_PLATEAU_GAPS_DONE $(date)"
