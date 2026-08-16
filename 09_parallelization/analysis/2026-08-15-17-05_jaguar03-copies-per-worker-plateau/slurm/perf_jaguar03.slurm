#!/bin/bash
# Direct counter evidence for what runs out at the top of the copies-per-worker ladder.
#
# The performance counters are readable on this node without privileges (perf_event_paranoid is
# 0), so the traffic the training load puts on the memory system can be counted rather than
# inferred. Each window below runs the load, waits for it to reach its steady state, then counts
# for thirty seconds across the whole machine, which is exclusively this job's.
#
# What the counters say: `cache-misses` on this processor counts last-level misses, and every one
# of them moves a 64-byte line, so misses per second times 64 is the traffic the memory system
# carried. Compared against what the stream measurement says the memory system can deliver, that
# answers whether the load has filled it.
#SBATCH --job-name=rnd09_cpu_plateau_perf
#SBATCH --partition=gpu
#SBATCH --nodelist=jaguar03
#SBATCH --exclusive
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=112
#SBATCH --ntasks=1
#SBATCH --mem=960G
#SBATCH --time=2:00:00
#SBATCH --output=/p/rlprojects/RND/09_parallelization/analysis/2026-08-15-17-05_jaguar03-copies-per-worker-plateau/slurm/perf_%j.out
set -euo pipefail

RUN=/p/rlprojects/RND/09_parallelization/analysis/2026-08-15-17-05_jaguar03-copies-per-worker-plateau
PY=/p/rlprojects/RND/.venvs/exploration/bin/python
LADDER=$RUN/code/bench_copies_per_worker.py
OUT=$RUN/logs/perf_counters.txt
export PYTHONNOUSERSITE=1

echo "host $(hostname), $(nproc) processors visible, started $(date)"

# --- which memory-traffic events this kernel exposes, so the numbers below can be traced to a
# --- named event rather than to an assumption about what the counter means
{
  echo "===== perf_event_paranoid ====="; cat /proc/sys/kernel/perf_event_paranoid
  echo "===== perf list, cache and memory events ====="
  perf list 2>/dev/null | grep -iE "cache|l3|dram|memory|umc|data.fabric" | head -60 || true
} > "$RUN/logs/perf_events.txt" 2>&1

# --- one counting window per configuration: start the load, let it settle, count for 30 seconds.
# --- Two counting passes with different event groups, so an event this kernel does not expose
# --- costs only its own group rather than the whole measurement. The second group is the address
# --- translation caches, which are the suspect once the memory system has been ruled out: a
# --- worker holding hundreds of megabytes has far more pages than the translation cache holds.
window () {
  # goal: run one training configuration and count the whole machine while it is in steady state.
  # The fourth argument is the seconds-per-iteration estimate the sweep uses to choose how many
  # iterations to time; it is set per setting here so that every load lasts comfortably longer
  # than the ninety seconds of counting that follow it.
  local procs="$1" copies="$2" style="$3" est="$4"
  echo "===== procs=$procs copies=$copies style=$style est=$est $(date -Iseconds) =====" >> "$OUT"
  $PY -u $LADDER --copies "$copies" --procs-list "$procs" --styles "$style" \
    --tag "_perfload_p${procs}_c${copies}" --write-results 0 \
    --est-first "$est" > "$RUN/logs/perfload_p${procs}_c${copies}_${style}.log" 2>&1 &
  local load=$!
  sleep 30
  echo "--- group 1: cache and instruction counts ---" >> "$OUT"
  perf stat -a -e cache-misses,cache-references,instructions,cycles,page-faults \
    -- sleep 20 >> "$OUT" 2>&1 || echo "group 1 unavailable" >> "$OUT"
  echo "--- group 2: address translation ---" >> "$OUT"
  perf stat -a -e dTLB-load-misses,dTLB-loads,L1-dcache-load-misses,L1-dcache-loads \
    -- sleep 20 >> "$OUT" 2>&1 || echo "group 2 unavailable" >> "$OUT"
  echo "--- group 3: stalls ---" >> "$OUT"
  perf stat -a -e stalled-cycles-backend,stalled-cycles-frontend,cycles,instructions \
    -- sleep 20 >> "$OUT" 2>&1 || echo "group 3 unavailable" >> "$OUT"
  wait $load || echo "load exited non-zero for procs=$procs copies=$copies" >> "$OUT"
}

# --- one worker alone on the node: what one worker does with nothing competing
window 1 64 full_batch 0.6

# --- the full machine at three rungs of the ladder, one update per batch: below the peak, at it,
# --- and past it, so the counters can be compared across the turn
window 112 16 full_batch 0.4
window 112 64 full_batch 0.6
window 112 256 full_batch 7.5

echo "CPU_PLATEAU_PERF_DONE $(date)"
