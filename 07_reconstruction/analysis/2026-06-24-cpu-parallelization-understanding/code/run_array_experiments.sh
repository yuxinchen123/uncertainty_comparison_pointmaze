#!/bin/bash
# Per-node experiments for the "ntasks vs cpus-per-task / concurrency / memory" section.
# Submitted as ONE sbatch with --ntasks=4 --cpus-per-task=4 --mem=20G (the real 03_run_cpu.slurm shape);
# runs sub-experiments as srun job steps inside that allocation.
set -uo pipefail
OUTDIR="$1"
CODE="$OUTDIR/code"
PY=/u/sl5nw/.conda/envs/exploration/bin/python
NODE="$(hostname -s)"
LOG="$OUTDIR/logs/array_$NODE"
mkdir -p "$LOG"
export RND_PROFILE=1
echo "node=$NODE alloc ntasks=$SLURM_NTASKS cpt=$SLURM_CPUS_PER_TASK mem=${SLURM_MEM_PER_NODE}MB $(date -u +%FT%TZ)" | tee "$LOG/run.log"

# (1) CO-LOCATED: 4 tasks share this node, each bound to cpt=N cpus (4N cpus used of 16).
#     This is exactly what `srun` with --ntasks=4 does (the user's current method).
for N in 4 2 1; do
  echo "[colocated] ntasks=4 cpt=$N -> $((4*N)) cpus used $(date -u +%T)" | tee -a "$LOG/run.log"
  srun --ntasks=4 --cpus-per-task=$N --cpu-bind=cores --exact \
    bash "$CODE/run_one_e2e.sh" colocated $N "$LOG" >> "$LOG/srun.log" 2>&1 \
    && echo "  ok" >> "$LOG/run.log" || echo "  colocated N=$N FAIL" | tee -a "$LOG/run.log"
done

# (2) ISOLATED: 1 task alone with cpt=N cpus (the rest of the node idle) — proxy for one
#     job-array task / one separate sbatch that lands alone on a node.
for N in 4 2 1; do
  echo "[isolated] ntasks=1 cpt=$N $(date -u +%T)" | tee -a "$LOG/run.log"
  srun --ntasks=1 --cpus-per-task=$N --cpu-bind=cores --exact \
    bash "$CODE/run_one_e2e.sh" isolated $N "$LOG" >> "$LOG/srun.log" 2>&1 \
    && echo "  ok" >> "$LOG/run.log" || echo "  isolated N=$N FAIL" | tee -a "$LOG/run.log"
done

# (3) MEMORY BANDWIDTH: 1 vs 4 concurrent triad streams (each task cpt=4). Per-copy
#     throughput dropping at K=4 is direct evidence of shared-memory-bandwidth contention.
for K in 1 4; do
  echo "[bandwidth] K=$K concurrent triad streams $(date -u +%T)" | tee -a "$LOG/run.log"
  srun --ntasks=$K --cpus-per-task=4 --cpu-bind=cores --exact \
    bash -c "$PY $CODE/microbench.py --n_threads 1 --ops mem_triad --seconds 3 --warmup 0.5 --out $LOG/bw_K${K}_proc\${SLURM_PROCID:-0}.json" \
    >> "$LOG/srun.log" 2>&1 \
    && echo "  ok" >> "$LOG/run.log" || echo "  bw K=$K FAIL" | tee -a "$LOG/run.log"
done

touch "$LOG/_DONE"
echo "DONE node=$NODE $(date -u +%FT%TZ)" | tee -a "$LOG/run.log"
