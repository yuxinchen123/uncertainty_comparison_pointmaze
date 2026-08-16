#!/bin/bash
# Where total throughput stops rising as the copies per worker grow, on jaguar03 held exclusively.
#
# The report's processor sweep stops at 224 workers holding 16 copies each and is still climbing,
# so the machine's ceiling has not been shown. Workers cannot be added — 224 is the logical
# processor count — but each worker can hold more copies, so this job walks that knob.
#
# It also repairs the method the earlier sweep used. Those numbers time five iterations, about two
# seconds, which reads the seconds during which this processor runs above its sustained clock; and
# they compute the machine's rate as the sum of each process's own rate, which over so short a
# window describes a load that never existed at one time, because hundreds of processes are still
# starting. Here every worker warms up for about twenty seconds, waits at a barrier, and the
# aggregate is the work done inside the window in which every worker was running. Both the old
# figures are kept beside the new one so the difference is visible.
#
# The worker count is measured as well as the copies, because 112 workers (one per physical core)
# beat 224 (both hardware threads of every core) at one copy each, and whether that holds at
# larger copy counts is not known.
#
# The order is cheapest first and every point is written to disk as it finishes, so a job killed
# at its largest point keeps every point below it.
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
BENCH=/p/rlprojects/RND/09_parallelization/benchmarks/bench_train_cpu.py
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
} > "$RUN/logs/hardware_jaguar03.txt" 2>&1

# --- the five-iteration measurement of two settings, run exactly as the earlier sweep ran it, so
# --- the correction has a genuine before number measured on this same node in this same job
$PY -u $BENCH --mode processes --procs 224 --copies-per-proc 16 --style full_batch \
  --iters 5 --warmup 2 --tag _burst_before_j3_p16_styleA
$PY -u $BENCH --mode processes --procs 224 --copies-per-proc 16 \
  --iters 5 --warmup 2 --tag _burst_before_j3_p16

# --- the memory model: one process at a time, so the ladder's guard knows what each rung costs
# --- before it runs one. Both conventions, since the update that works on the whole batch at once
# --- holds more at once. (Written already by the cancelled first attempt; cheap to repeat.)
$PY -u $LADDER --probe-memory 1 16 64 128 256 512 --style full_batch
$PY -u $LADDER --probe-memory 1 16 64 128 256 512 --style epoch_minibatch

# --- the sweep: copies per worker outermost so the cheap rungs of every series finish first;
# --- both worker counts and both update conventions at every rung
$PY -u $LADDER --copies 1 4 16 32 64 128 256 512 --procs-list 112 224 \
  --styles full_batch epoch_minibatch --tag _plateau_j3

# --- the decisive rungs a second time, to show the spread between repeats
$PY -u $LADDER --copies 16 64 128 256 --procs-list 112 224 \
  --styles full_batch epoch_minibatch --tag _repeat_j3

# --- what ran out. One worker alone on the whole node is the per-copy cost with no competition
# --- at all; the 112-worker and 224-worker series above are the same work under competition.
$PY -u $LADDER --copies 1 16 64 128 --procs-list 1 --styles full_batch epoch_minibatch \
  --tag _alone_j3 --write-results 0

# --- the memory system: how much it delivers as more processes ask for it, and how much is left
# --- for a stream of memory traffic while the training load runs at a small and a large setting
$PY -u $PRESSURE --saturation 1 8 28 56 112 224 --seconds 15
$PY -u $PRESSURE --corun --train-procs 216 --probe-procs 8 --copies 1 16 128 \
  --style full_batch --iters 90 60 15 --seconds 20 --delay 30

echo "CPU_PLATEAU_JAGUAR03_DONE $(date)"
