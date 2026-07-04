#!/bin/bash
# Launch the run-3.1.2 FOLLOW-UP sweep (unit norm, ridge 1e-8, no clip, beta 5 decades, 50 seeds).
# Usage:  TIME=4-00:00:00 bash launch_queue_unit_ridge_1e8.sh
#
# 250 runs, each 1 CPU. Job shape: 16 tasks x 1 cpu with --ntasks-per-core=2 (project rule
# .claude/rules/slurm-submission.md step 7: without it AllocCPUS doubles and tasks double-bind on one
# hardware thread; verified in run 3.1.2). 16 jobs = 256 workers >= 250 runs, so every run starts in
# one wave. OPEN PARTITIONS ONLY — the sl5nw reservation (jaguar03/puma01) is deliberately left free
# for other sweeps (user instruction 2026-07-03):
#   cpu    8 jobs (128 cpus; partition idle at launch, 0/400 per-user cap used)
#   gpu    5 jobs, one idle ALLOWLIST node per job (--gpus-per-node=0), --nodelist set per job
#   gnolim 3 jobs (48 of the 80-cpu per-user cap; all gnolim nodes idle at launch)
#   nolim  SKIPPED (its per-user memory pool read 1048576/1048576 = full at launch)
# Ids go to submitted_jobids_<sweep>.txt as they are returned (hard rule: scancel only ids from that file).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$(dirname "$HERE")"
PROJ=/p/rlprojects/RND/07_reconstruction
LOGDIR="$RUN_DIR/logs"; mkdir -p "$LOGDIR"
PY="/u/sl5nw/.conda/envs/exploration/bin/python"

TAG="unit_ridge-1e-8_clip-inf_beta-1e-4-1e-3-1e-2-1e-1-1e0_50seed_16x1"
TS="$(date +%Y-%m-%d-%H-%M)"
SWEEP="${TS}_${TAG}"
JOBIDS="$HERE/submitted_jobids_${SWEEP}.txt"
TIME="${TIME:-4-00:00:00}"
echo "[sweep] $SWEEP  TIME=$TIME"

# build the 250-config queue + manifest row
"$PY" "$HERE/build_queue_unit_ridge_1e8.py" --sweep_id "$SWEEP" || { echo "build_queue failed"; exit 1; }

submitted=0
sub() {  # sub <jobname> <extra sbatch args...>
  local name="$1"; shift
  local out id
  out=$(sbatch --job-name="$name" --chdir="$PROJ" --nodes=1 --ntasks=16 --cpus-per-task=1 \
        --ntasks-per-core=2 --mem-per-cpu=2G --output="$LOGDIR/${name}_%j.log" --time="$TIME" \
        --export=ALL,SWEEP_ID="$SWEEP" "$@" "$HERE/worker_16x1.slurm")
  id=$(grep -oP '[0-9]+$' <<< "$out"); [[ -n "$id" ]] && echo "$id" >> "$JOBIDS"
  echo "  [$name] $out"
  submitted=$((submitted+1))
  if (( submitted % 10 == 0 )); then echo "[throttle] $submitted submitted; sleep 30"; sleep 30; fi
}

# cpu partition: 8 jobs, any node (puma01 is reservation-protected, normal jobs cannot land there)
for i in 1 2 3 4 5 6 7 8; do
  sub tab-bench --partition=cpu
done
# gpu allowlist: one idle low-tier node per job (include-list, never --exclude; no reserved node here)
GPU_NODES=(lynx01 lynx02 lynx03 lynx04 lynx05)
for n in "${GPU_NODES[@]}"; do
  sub meta-icl --partition=gpu --nodelist="$n" --gpus-per-node=0
done
# gnolim: 3 jobs (old-tier gpu nodes, CPU-only ask)
for i in 1 2 3; do
  sub causal-rep --partition=gnolim --gpus-per-node=0
done

echo "[done] submitted $submitted jobs; ids in $JOBIDS"
echo "[done] queue: queue/$SWEEP/   data: data/$SWEEP/local/"
