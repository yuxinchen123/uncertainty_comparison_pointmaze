#!/bin/bash
# The measurements that say WHAT ran out at the plateau, plus repeats of the rungs the answer
# turns on. The sweep job measured the ladder itself; this one runs the rest in order of what the
# report needs first, so that a job cut short still leaves the explanation measured.
#
#   1. the memory system: what it delivers alone, and what it has left under the training load
#   2. one worker alone on the empty node: the per-copy cost with nothing competing
#   3. repeats of the four peak rungs, to show the spread between runs of the same setting
#SBATCH --job-name=rnd09_cpu_plateau_extras
#SBATCH --partition=gpu
#SBATCH --nodelist=jaguar03
#SBATCH --exclusive
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=112
#SBATCH --ntasks=1
#SBATCH --mem=960G
#SBATCH --time=4:00:00
#SBATCH --output=/p/rlprojects/RND/09_parallelization/analysis/2026-08-15-17-05_jaguar03-copies-per-worker-plateau/slurm/extras_%j.out
set -euo pipefail

RUN=/p/rlprojects/RND/09_parallelization/analysis/2026-08-15-17-05_jaguar03-copies-per-worker-plateau
PY=/p/rlprojects/RND/.venvs/exploration/bin/python
LADDER=$RUN/code/bench_copies_per_worker.py
PRESSURE=$RUN/code/memory_pressure.py
export PYTHONNOUSERSITE=1

echo "host $(hostname), $(nproc) processors visible, started $(date)"

# --- the memory system's own rate, and what a stream of memory traffic gets while the training
# --- load runs beside it at a small, a middling and a large copies-per-worker setting
$PY -u $PRESSURE --corun --train-procs 216 --probe-procs 8 --copies 1 16 64 \
  --style full_batch --iters 90 45 25 --seconds 20 --delay 30

# --- one worker alone on the whole node: the same copies with nothing to compete against
$PY -u $LADDER --copies 1 16 64 128 --procs-list 1 --styles full_batch epoch_minibatch \
  --tag _alone_j3 --write-results 0

# --- the peak rungs a second time, so the report can state the spread between repeats
$PY -u $LADDER --copies 64 --procs-list 112 224 --styles full_batch --tag _repeat_j3
$PY -u $LADDER --copies 16 --procs-list 112 224 --styles epoch_minibatch --tag _repeat_j3

echo "CPU_PLATEAU_EXTRAS_DONE $(date)"
