#!/bin/bash
# The counter windows again, taken later into the load.
#
# The first counter job began counting thirty seconds after starting the load. At 256 copies a
# worker takes about a minute to build its copies and prime its statistics, so part of that
# window was of the load starting rather than of the load running. Here the counting begins 150
# seconds in, which is inside the timed region at both settings, and the two settings are the
# ones the argument turns on: the rung where throughput peaks and the rung well past it.
#SBATCH --job-name=rnd09_cpu_plateau_perf2
#SBATCH --partition=gpu
#SBATCH --nodelist=jaguar03
#SBATCH --exclusive
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=112
#SBATCH --ntasks=1
#SBATCH --mem=960G
#SBATCH --time=2:00:00
#SBATCH --output=/p/rlprojects/RND/09_parallelization/analysis/2026-08-15-17-05_jaguar03-copies-per-worker-plateau/slurm/perf_settled_%j.out
set -euo pipefail

RUN=/p/rlprojects/RND/09_parallelization/analysis/2026-08-15-17-05_jaguar03-copies-per-worker-plateau
PY=/p/rlprojects/RND/.venvs/exploration/bin/python
LADDER=$RUN/code/bench_copies_per_worker.py
OUT=$RUN/logs/perf_counters_settled.txt
export PYTHONNOUSERSITE=1

echo "host $(hostname), $(nproc) processors visible, started $(date)"

window () {
  # goal: count the whole machine while one configuration is well into its timed region
  local procs="$1" copies="$2" style="$3" est="$4"
  echo "===== procs=$procs copies=$copies style=$style est=$est $(date -Iseconds) =====" >> "$OUT"
  $PY -u $LADDER --copies "$copies" --procs-list "$procs" --styles "$style" \
    --tag "_perfload2_p${procs}_c${copies}" --write-results 0 \
    --est-first "$est" > "$RUN/logs/perfload2_p${procs}_c${copies}_${style}.log" 2>&1 &
  local load=$!
  sleep 150
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

window 112 16 full_batch 0.4
window 112 64 full_batch 0.6
window 112 256 full_batch 7.5

echo "CPU_PLATEAU_PERF_SETTLED_DONE $(date)"
