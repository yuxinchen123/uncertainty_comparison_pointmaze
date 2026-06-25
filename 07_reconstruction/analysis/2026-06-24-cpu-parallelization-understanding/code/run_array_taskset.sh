#!/bin/bash
# Robust per-node concurrency + memory experiment using taskset pinning inside a
# single 16-cpu allocation (no srun sub-steps, which hung at cpus-per-task=1).
# Reproduces what srun --ntasks=4 --cpus-per-task=N does: 4 worker processes, each
# pinned to N distinct cpus, with OMP_NUM_THREADS=N — vs a single isolated run.
set -uo pipefail
OUTDIR="$1"
CODE="$OUTDIR/code"
PY=/u/sl5nw/.conda/envs/exploration/bin/python
NODE="$(hostname -s)"
LOG="$OUTDIR/logs/array_$NODE"
mkdir -p "$LOG"
export RND_PROFILE=1 PYTHONUNBUFFERED=1
# the 16 cpus this allocation owns, sorted
CPUS=($("$PY" -c "import os;print(' '.join(map(str,sorted(os.sched_getaffinity(0)))))"))
NCPU=${#CPUS[@]}
echo "node=$NODE ncpus=$NCPU cpus=${CPUS[*]} mem=${SLURM_MEM_PER_NODE:-NA}MB $(date -u +%FT%TZ)" | tee "$LOG/run.log"

run_e2e () {  # tag cpt cpuset proc
  local tag=$1 cpt=$2 cs=$3 proc=$4
  OMP_NUM_THREADS=$cpt MKL_NUM_THREADS=$cpt OPENBLAS_NUM_THREADS=$cpt \
  NUMEXPR_NUM_THREADS=$cpt VECLIB_MAXIMUM_THREADS=$cpt \
  taskset -c "$cs" "$PY" "$CODE/profile_e2e.py" --algorithm rnd_linear_next_state \
    --device cpu --n_threads "$cpt" --total_timesteps 1500 --warmup_steps 500 \
    --out "$LOG/${tag}_cpt${cpt}_proc${proc}.json" >> "$LOG/py.log" 2>&1
}

# (1) CO-LOCATED: 4 workers, each pinned to cpt=N distinct cpus (uses first 4N cpus).
for N in 4 2 1; do
  echo "[colocated] cpt=$N x4 -> $((4*N)) cpus $(date -u +%T)" | tee -a "$LOG/run.log"
  pids=()
  for i in 0 1 2 3; do
    start=$((i*N)); cs=$(IFS=,; echo "${CPUS[*]:start:N}")
    run_e2e colocated "$N" "$cs" "$i" & pids+=($!)
  done
  wait "${pids[@]}" && echo "  ok" >> "$LOG/run.log" || echo "  colocated N=$N partial" | tee -a "$LOG/run.log"
done

# (2) ISOLATED: 1 worker alone, pinned to N cpus (rest idle).
for N in 4 2 1; do
  echo "[isolated] cpt=$N $(date -u +%T)" | tee -a "$LOG/run.log"
  cs=$(IFS=,; echo "${CPUS[*]:0:N}")
  run_e2e isolated "$N" "$cs" 0 && echo "  ok" >> "$LOG/run.log" || echo "  isolated N=$N FAIL" | tee -a "$LOG/run.log"
done

# (3) MEMORY BANDWIDTH: 1 vs 4 concurrent triad streams (each pinned to 4 cpus).
for K in 1 4; do
  echo "[bandwidth] K=$K $(date -u +%T)" | tee -a "$LOG/run.log"
  pids=()
  for ((i=0;i<K;i++)); do
    start=$((i*4)); cs=$(IFS=,; echo "${CPUS[*]:start:4}")
    taskset -c "$cs" "$PY" "$CODE/microbench.py" --n_threads 1 --ops mem_triad \
      --seconds 3 --warmup 0.5 --out "$LOG/bw_K${K}_proc${i}.json" >> "$LOG/py.log" 2>&1 & pids+=($!)
  done
  wait "${pids[@]}"
done

touch "$LOG/_DONE"
echo "DONE node=$NODE $(date -u +%FT%TZ)" | tee -a "$LOG/run.log"
