#!/bin/bash
set -uo pipefail
OUTDIR="$1"; CODE="$OUTDIR/code"
CONDA="/sw/ubuntu2204/ebu082024/software/common/core/miniforge/24.7.1-py3.11/condabin/conda"
RUN="$CONDA run -n exploration python"
NODE="$(hostname -s)"; LOG="$OUTDIR/logs/twocheck_$NODE"; mkdir -p "$LOG"
for ALGO in no_exploration rnd_linear_next_state; do
  for N in 1 2 4 8; do
    export OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N NUMEXPR_NUM_THREADS=$N VECLIB_MAXIMUM_THREADS=$N RND_PROFILE=1
    $RUN "$CODE/profile_e2e.py" --algorithm "$ALGO" --device cpu --n_threads "$N" \
       --total_timesteps 4000 --warmup_steps 1200 \
       --out "$LOG/e2e_${ALGO}_dev-cpu_threads-${N}.json" >> "$LOG/log.txt" 2>&1
  done
done
touch "$LOG/_DONE"
