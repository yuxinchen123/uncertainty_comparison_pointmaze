#!/bin/bash
# Full controlled CPU thread-scaling sweep for ONE node.
# Each node runs this serially so thread-cap comparisons are apples-to-apples on
# identical hardware; many nodes run it in parallel (separate sbatch jobs) to get
# independent replicate sets and resilience if a node becomes unavailable.
#
# Usage: run_node_sweep.sh <OUTDIR_ABS> <DEVICE> [TOTAL_TIMESTEPS] [WARMUP]
set -uo pipefail

OUTDIR="$1"          # absolute path to the analysis output dir
DEVICE="${2:-cpu}"
TOTAL="${3:-4000}"
WARMUP="${4:-1200}"

CODE="$OUTDIR/code"
CONDA="/sw/ubuntu2204/ebu082024/software/common/core/miniforge/24.7.1-py3.11/condabin/conda"
PROJ="$(dirname "$(dirname "$OUTDIR")")"   # 07_reconstruction
RUN="$CONDA run -n exploration python"

NODE="$(hostname -s)"
LOG="$OUTDIR/logs/$NODE"
mkdir -p "$LOG"

THREADS="1 4 8 16"
ALGOS="no_exploration rnd_linear_next_state gt_position rnd_elliptical"

set_threads () {  # cap every math backend to $1
  export OMP_NUM_THREADS=$1 MKL_NUM_THREADS=$1 OPENBLAS_NUM_THREADS=$1 \
         NUMEXPR_NUM_THREADS=$1 VECLIB_MAXIMUM_THREADS=$1 RND_PROFILE=1
}

echo "=== node sweep start: node=$NODE device=$DEVICE total=$TOTAL $(date -u +%FT%TZ) ===" | tee "$LOG/run.log"

# ---- hardware / allocation facts ----
{
  echo "host=$(hostname)"
  echo "date_utc=$(date -u +%FT%TZ)"
  echo "nproc=$(nproc)"
  echo "slurm_job_id=${SLURM_JOB_ID:-NA}"
  echo "slurm_cpus_on_node=${SLURM_CPUS_ON_NODE:-NA}"
  echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-NA}"
  echo "cpuset=$(cat /sys/fs/cgroup/cpuset.cpus 2>/dev/null || cat /sys/fs/cgroup/cpuset/cpuset.cpus 2>/dev/null || echo NA)"
} > "$LOG/node_meta.txt" 2>&1
lscpu > "$LOG/lscpu.txt" 2>&1
grep -m1 "model name" /proc/cpuinfo >> "$LOG/node_meta.txt" 2>&1

# ---- microbenchmarks: per-primitive thread scaling ----
for N in $THREADS; do
  set_threads "$N"
  echo "--- microbench device=$DEVICE threads=$N ---" | tee -a "$LOG/run.log"
  $RUN "$CODE/microbench.py" --n_threads "$N" --seconds 2.5 --warmup 0.8 --device "$DEVICE" \
       --out "$LOG/microbench_dev-${DEVICE}_threads-${N}.json" \
       >> "$LOG/microbench_dev-${DEVICE}.log" 2>&1 \
    && echo "  microbench n=$N ok" | tee -a "$LOG/run.log" \
    || echo "  microbench n=$N FAILED" | tee -a "$LOG/run.log"
done

# ---- end-to-end real training, steady-state throughput ----
for ALGO in $ALGOS; do
  for N in $THREADS; do
    set_threads "$N"
    echo "--- e2e $ALGO device=$DEVICE threads=$N $(date -u +%T) ---" | tee -a "$LOG/run.log"
    $RUN "$CODE/profile_e2e.py" --algorithm "$ALGO" --device "$DEVICE" --n_threads "$N" \
         --total_timesteps "$TOTAL" --warmup_steps "$WARMUP" \
         --out "$LOG/e2e_${ALGO}_dev-${DEVICE}_threads-${N}.json" \
         >> "$LOG/e2e_${ALGO}_dev-${DEVICE}.log" 2>&1 \
      && echo "  e2e $ALGO n=$N ok" | tee -a "$LOG/run.log" \
      || echo "  e2e $ALGO n=$N FAILED" | tee -a "$LOG/run.log"
  done
done

# ---- cross-check: UNMODIFIED 04_many_exploration_method.py under /usr/bin/time -v ----
# (CPU device only; verifies the snapshot driver matches the real script)
if [ "$DEVICE" = "cpu" ]; then
  cd "$PROJ" || exit 0
  for N in 1 8; do
    set_threads "$N"
    echo "--- crosscheck real-04 rnd_linear threads=$N ---" | tee -a "$LOG/run.log"
    /usr/bin/time -v $RUN "$PROJ/04_many_exploration_method.py" \
        --device cpu --use_wandb False --algorithm rnd_linear_next_state \
        --total_timesteps "$TOTAL" --eval_freq 100000000 \
        --n_eval_episodes 1 --n_eval_episodes_final 1 \
        > "$LOG/crosscheck_real04_threads-${N}.log" 2>&1 \
      && echo "  crosscheck n=$N ok" | tee -a "$LOG/run.log" \
      || echo "  crosscheck n=$N FAILED" | tee -a "$LOG/run.log"
  done
fi

echo "=== node sweep DONE: node=$NODE $(date -u +%FT%TZ) ===" | tee -a "$LOG/run.log"
touch "$LOG/_DONE"
