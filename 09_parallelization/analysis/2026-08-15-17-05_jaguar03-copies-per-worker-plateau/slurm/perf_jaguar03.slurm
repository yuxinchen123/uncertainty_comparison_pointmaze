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
  perf list 2>/dev/null | grep -iE "cache|l3|dram|memory|umc|data.fabric" | head -60
} > "$RUN/logs/perf_events.txt" 2>&1

# --- one counting window per configuration: start the load, let it settle, count for 30 seconds
window () {
  # goal: run one training configuration and count the whole machine while it is in steady state
  local procs="$1" copies="$2" style="$3" iters="$4"
  echo "===== procs=$procs copies=$copies style=$style iters=$iters $(date -Iseconds) =====" \
    >> "$OUT"
  $PY -u $LADDER --ladder "$copies" --procs "$procs" --style "$style" \
    --tag "_perfload_p${procs}_c${copies}" --results-min-copies 999999 \
    --est-first 2.0 > "$RUN/logs/perfload_p${procs}_c${copies}_${style}.log" 2>&1 &
  local load=$!
  sleep 45
  perf stat -a -e cache-misses,cache-references,instructions,cycles,page-faults \
    -- sleep 30 >> "$OUT" 2>&1 || echo "perf failed for procs=$procs copies=$copies" >> "$OUT"
  wait $load || echo "load exited non-zero for procs=$procs copies=$copies" >> "$OUT"
}

# --- one worker alone on the node: the traffic one worker makes with nothing competing
window 1 1 full_batch 60
window 1 16 full_batch 60
window 1 128 full_batch 20

# --- the full machine at three rungs of the ladder, one update per batch
window 224 1 full_batch 90
window 224 16 full_batch 60
window 224 128 full_batch 20

# --- the same full machine with sixteen updates per batch, which touches the parameters sixteen
# --- times per iteration instead of once
window 224 16 epoch_minibatch 40

echo "CPU_PLATEAU_PERF_DONE $(date)"
