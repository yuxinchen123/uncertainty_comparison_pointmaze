#!/bin/bash
# Augmented microbench-only pass (original + gap-closing primitives) for ONE node.
# Writes to logs/mb2_<node>/ so it is picked up by aggregate.py as an extra replicate.
set -uo pipefail
OUTDIR="$1"
CODE="$OUTDIR/code"
CONDA="/sw/ubuntu2204/ebu082024/software/common/core/miniforge/24.7.1-py3.11/condabin/conda"
RUN="$CONDA run -n exploration python"
NODE="$(hostname -s)"
LOG="$OUTDIR/logs/mb2_$NODE"
mkdir -p "$LOG"
{ echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} cpus=${SLURM_CPUS_PER_TASK:-NA} date=$(date -u +%FT%TZ)"; \
  grep -m1 'model name' /proc/cpuinfo; } > "$LOG/node_meta.txt" 2>&1
lscpu > "$LOG/lscpu.txt" 2>&1
for N in 1 4 8 16; do
  export OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N \
         NUMEXPR_NUM_THREADS=$N VECLIB_MAXIMUM_THREADS=$N
  $RUN "$CODE/microbench.py" --n_threads "$N" --seconds 2.5 --warmup 0.8 --device cpu \
       --out "$LOG/microbench_dev-cpu_threads-${N}.json" >> "$LOG/mb.log" 2>&1 \
    && echo "n=$N ok" || echo "n=$N FAILED"
done
touch "$LOG/_DONE"
echo "mb2 done on $NODE"
