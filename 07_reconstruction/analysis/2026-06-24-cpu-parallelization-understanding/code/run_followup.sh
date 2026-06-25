#!/bin/bash
# Follow-up measurements closing methodology-review gaps. One node.
# Usage: run_followup.sh <OUTDIR_ABS>
set -uo pipefail
OUTDIR="$1"
CODE="$OUTDIR/code"
PROJ="$(dirname "$(dirname "$OUTDIR")")"   # 07_reconstruction
CONDA="/sw/ubuntu2204/ebu082024/software/common/core/miniforge/24.7.1-py3.11/condabin/conda"
RUN="$CONDA run -n exploration python"
NODE="$(hostname -s)"
LOG="$OUTDIR/logs/followup_$NODE"
mkdir -p "$LOG"
NOENV="env -u OMP_NUM_THREADS -u MKL_NUM_THREADS -u OPENBLAS_NUM_THREADS -u NUMEXPR_NUM_THREADS -u VECLIB_MAXIMUM_THREADS"
set_threads(){ export OMP_NUM_THREADS=$1 MKL_NUM_THREADS=$1 OPENBLAS_NUM_THREADS=$1 \
               NUMEXPR_NUM_THREADS=$1 VECLIB_MAXIMUM_THREADS=$1 RND_PROFILE=1; }
echo "followup start $NODE $(date -u +%FT%TZ)" > "$LOG/run.log"

# (A) M1: torch's DEFAULT thread count under the Slurm cgroup, with NO *_NUM_THREADS set.
$NOENV $RUN - > "$LOG/A_torch_default_threads.txt" 2>&1 <<'PY'
import os, torch, numpy as np
print("SLURM_CPUS_PER_TASK =", os.environ.get("SLURM_CPUS_PER_TASK"))
print("len(sched_getaffinity) =", len(os.sched_getaffinity(0)))
print("OMP_NUM_THREADS env   =", os.environ.get("OMP_NUM_THREADS"))
print("torch.get_num_threads =", torch.get_num_threads())
try:
    import threadpoolctl
    for d in threadpoolctl.threadpool_info():
        print("backend", d.get("internal_api"), d.get("prefix"), "num_threads=", d.get("num_threads"))
except Exception as e:
    print("threadpoolctl:", e)
PY

# (B) m12: verify the cap actually reaches each backend (numpy/OpenBLAS AND torch/MKL).
for N in 1 8; do
  set_threads $N
  $RUN - >> "$LOG/B_threadpool_info.txt" 2>&1 <<PY
import threadpoolctl, numpy, torch
print("=== requested $N threads ===")
for d in threadpoolctl.threadpool_info():
    print(" ", d.get("internal_api"), d.get("prefix"), "num_threads=", d.get("num_threads"))
print("  torch.get_num_threads =", torch.get_num_threads())
PY
done

# (C) M1: uncapped real-04 under /usr/bin/time (NO thread env) -> the DEFAULT effective cores.
cd "$PROJ" || exit 0
$NOENV /usr/bin/time -v $RUN "$PROJ/04_many_exploration_method.py" \
    --device cpu --use_wandb False --algorithm rnd_linear_next_state \
    --total_timesteps 3000 --eval_freq 100000000 --n_eval_episodes 1 --n_eval_episodes_final 1 \
    > "$LOG/C_real04_uncapped_default.txt" 2>&1 && echo "C ok" >> "$LOG/run.log" || echo "C FAIL" >> "$LOG/run.log"

# (D) M2: production-like real-04 WITH eval+callbacks enabled, scaling at 1/4/8 threads.
for N in 1 4 8; do
  set_threads $N
  /usr/bin/time -v $RUN "$PROJ/04_many_exploration_method.py" \
      --device cpu --use_wandb False --algorithm rnd_linear_next_state \
      --total_timesteps 3000 --eval_freq 500 --n_eval_episodes 5 --n_eval_episodes_final 5 \
      > "$LOG/D_real04_witheval_threads-${N}.txt" 2>&1 && echo "D n=$N ok" >> "$LOG/run.log" || echo "D n=$N FAIL" >> "$LOG/run.log"
done

# (E) m5: end-to-end at 2 threads for all algorithms (fills the missing 2-core point).
for ALGO in no_exploration rnd_linear_next_state gt_position rnd_elliptical; do
  set_threads 2
  $RUN "$CODE/profile_e2e.py" --algorithm "$ALGO" --device cpu --n_threads 2 \
       --total_timesteps 4000 --warmup_steps 1200 \
       --out "$LOG/e2e_${ALGO}_dev-cpu_threads-2.json" >> "$LOG/E.log" 2>&1 \
    && echo "E $ALGO ok" >> "$LOG/run.log" || echo "E $ALGO FAIL" >> "$LOG/run.log"
done

touch "$LOG/_DONE"
echo "followup DONE $NODE $(date -u +%FT%TZ)" >> "$LOG/run.log"
