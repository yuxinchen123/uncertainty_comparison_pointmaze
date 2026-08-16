#!/bin/bash
# How far the copies per worker go before total throughput stops rising, on jaguar03 held
# exclusively so no other job shares the machine.
#
# The report's processor sweep stops at sixteen copies per worker, where total throughput is
# still climbing, so the machine's ceiling has not been shown. This job walks the copies-per-
# worker ladder at a fixed 224 workers for both update conventions, measures the peak memory of
# every worker at every rung, and then measures what runs out: the memory system's rate, the
# effect of loading both hardware threads of every core, and the cost that is there even with a
# single process on an otherwise empty node.
#
# The order is cheapest first and every point is written to disk as it finishes, so a job that
# is killed at its largest point keeps every point below it.
#SBATCH --job-name=rnd09_cpu_plateau
#SBATCH --partition=gpu
#SBATCH --nodelist=jaguar03
#SBATCH --exclusive
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=112
#SBATCH --ntasks=1
#SBATCH --mem=960G
#SBATCH --time=8:00:00
#SBATCH --output=/p/rlprojects/RND/09_parallelization/analysis/2026-08-15-17-05_jaguar03-copies-per-worker-plateau/slurm/plateau_%j.out
set -euo pipefail

RUN=/p/rlprojects/RND/09_parallelization/analysis/2026-08-15-17-05_jaguar03-copies-per-worker-plateau
PY=/p/rlprojects/RND/.venvs/exploration/bin/python
LADDER=$RUN/code/bench_copies_per_worker.py
PRESSURE=$RUN/code/memory_pressure.py
export PYTHONNOUSERSITE=1

echo "host $(hostname), $(nproc) processors visible, started $(date)"

# --- what the machine says about itself, so every hardware number in the report traces to a
# --- command run on this node rather than to recollection
{
  echo "===== hostname / date ====="; hostname; date -Iseconds
  echo "===== nproc ====="; nproc
  echo "===== lscpu ====="; lscpu
  echo "===== lscpu -C (cache) ====="; lscpu -C || echo "lscpu -C unavailable"
  echo "===== numactl -H ====="; numactl -H || echo "numactl unavailable"
  echo "===== getconf cache sizes ====="
  for v in LEVEL1_DCACHE_SIZE LEVEL2_CACHE_SIZE LEVEL3_CACHE_SIZE; do
    echo "$v = $(getconf $v)"
  done
  echo "===== free -g ====="; free -g
  echo "===== transparent huge pages ====="
  cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo "not readable"
  echo "===== perf availability ====="
  cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "no perf_event_paranoid"
  perf stat -e cache-misses,cache-references true 2>&1 | tail -12 || echo "perf unavailable"
} > "$RUN/logs/hardware_jaguar03.txt" 2>&1

# --- the memory model first: one process at a time, so the ladder's guard knows what each rung
# --- will cost before it runs one. Both update conventions, because the update that works on the
# --- whole batch at once holds more at once.
$PY -u $LADDER --probe-memory 1 16 64 128 256 512 --style full_batch
$PY -u $LADDER --probe-memory 1 16 64 128 256 512 --style epoch_minibatch

# --- the ladder itself, 224 workers throughout, cheapest rung first. Rungs at or below 16 copies
# --- repeat settings the report already has a file for, so they stay in this run folder; they are
# --- measured anyway, because a plateau read off rungs timed different lengths is not a plateau.
$PY -u $LADDER --ladder 1 4 16 32 64 128 256 512 --procs 224 --style full_batch \
  --tag _plateau_j3_styleA --results-min-copies 32
$PY -u $LADDER --ladder 1 4 16 32 64 128 256 512 --procs 224 --style epoch_minibatch \
  --tag _plateau_j3 --results-min-copies 32

# --- what ran out. Three separations, all in this run folder only, because none of them is a
# --- throughput setting anyone would run:
# ---   one worker alone on the whole node: the per-copy cost with no competition at all
# ---   112 workers: one per physical core, so neither hardware thread of a core is shared
# ---   224 workers: both threads of every core, which the ladder above already measured
$PY -u $LADDER --ladder 16 64 128 --procs 1 --style full_batch \
  --tag _alone_j3_styleA --results-min-copies 999999
$PY -u $LADDER --ladder 16 64 128 --procs 112 --style full_batch \
  --tag _onecore_j3_styleA --results-min-copies 999999

# --- the memory system: how much it delivers as more processes ask for it, and how much is left
# --- for a stream of memory traffic while the training load runs at a small and a large setting
$PY -u $PRESSURE --saturation 1 8 28 56 112 224 --seconds 15
$PY -u $PRESSURE --corun --train-procs 216 --probe-procs 8 --copies 1 16 128 \
  --style full_batch --iters 40 40 12 --seconds 20 --delay 25

echo "CPU_PLATEAU_JAGUAR03_DONE $(date)"
