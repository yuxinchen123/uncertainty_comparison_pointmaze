#!/bin/bash
# Repeated, same-node measurement of threads {1,2,4} for the 2-core column.
# 4 algorithms x {1,2,4} threads x REPS reps. One node; many nodes run in parallel.
set -uo pipefail
OUTDIR="$1"; TT="${2:-1500}"; WU="${3:-500}"; REPS="${4:-3}"
CODE="$OUTDIR/code"
CONDA="/sw/ubuntu2204/ebu082024/software/common/core/miniforge/24.7.1-py3.11/condabin/conda"
RUN="$CONDA run -n exploration python"
NODE="$(hostname -s)"; LOG="$OUTDIR/logs/twocore_$NODE"; mkdir -p "$LOG"
echo "start $NODE $(date -u +%FT%TZ) TT=$TT REPS=$REPS" > "$LOG/run.log"
for ALGO in no_exploration rnd_linear_next_state gt_position rnd_elliptical; do
  for N in 1 2 4; do
    for r in $(seq 1 "$REPS"); do
      export OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N \
             NUMEXPR_NUM_THREADS=$N VECLIB_MAXIMUM_THREADS=$N RND_PROFILE=1
      $RUN "$CODE/profile_e2e.py" --algorithm "$ALGO" --device cpu --n_threads "$N" \
           --total_timesteps "$TT" --warmup_steps "$WU" \
           --out "$LOG/e2e_${ALGO}_threads-${N}_rep-${r}.json" >> "$LOG/log.txt" 2>&1 \
        && echo "$ALGO n=$N rep=$r ok" >> "$LOG/run.log" || echo "$ALGO n=$N rep=$r FAIL" >> "$LOG/run.log"
    done
  done
  echo "  $ALGO done $(date -u +%T)" >> "$LOG/run.log"
done
touch "$LOG/_DONE"
echo "DONE $NODE $(date -u +%FT%TZ)" >> "$LOG/run.log"
