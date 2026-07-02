#!/bin/bash
# Launch Train run 3.1.2: BOTH Slurm-shape arms at once. Usage:  TIME=4-00:00:00 bash launch_queue.sh [tag]
#
# Builds two sweeps sharing the 36-config grid, seeds split by parity:
#   arm A = 8 tasks x 2 cpus  (--mem-per-cpu=2G -> job = 16 cpus + 32G), even seeds, worker_8x2.slurm
#   arm B = 16 tasks x 1 cpu  (--mem-per-cpu=2G -> job = 16 cpus + 32G), odd seeds,  worker_16x1.slurm
# Job-level ask is IDENTICAL (16 cpus + 32G) so no node class can accept one arm and reject the other.
#
# Two arenas (.claude/rules/slurm-submission.md + the run-3.1.2 red-team review):
#   EVIDENCE (reservation, co-located same-silicon pairs): jaguar03 7A+7B, puma01 3A+3B (memory-capped:
#     246G / 32G-per-job -> 7 jobs max; arms kept even at 3+3). Speed/error comparisons use ONLY these.
#   BULK DRAIN (open partitions, secondary): cpu 8A+8B; gpu allowlist 5A+5B one node per job; nolim is
#     SKIPPED (80-cpu per-user cap, DenyOnLimit, currently filled by run-3.1.1's jobs).
# Submissions interleave A,B,A,B under the 10-jobs-then-sleep-30s throttle so both arms ramp together.
# Ids go to submitted_jobids_<sweep>.txt per arm (hard rule: scancel only ids from these files).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$(dirname "$HERE")"
PROJ=/p/rlprojects/RND/07_reconstruction
LOGDIR="$RUN_DIR/logs"; mkdir -p "$LOGDIR"
PY="/u/sl5nw/.conda/envs/exploration/bin/python"

TAG="${1:-ablation}"
TS="$(date +%Y-%m-%d-%H-%M)"
SWEEP_A="${TS}_armA-8tasks-2cpu_${TAG}"
SWEEP_B="${TS}_armB-16tasks-1cpu_${TAG}"
JOBIDS_A="$HERE/submitted_jobids_${SWEEP_A}.txt"
JOBIDS_B="$HERE/submitted_jobids_${SWEEP_B}.txt"
TIME="${TIME:-4-00:00:00}"
echo "[sweeps] A=$SWEEP_A  B=$SWEEP_B  TIME=$TIME"

# build both arms' queues (900 configs each) + manifest rows
"$PY" "$HERE/build_queue.py" --sweep_id "$SWEEP_A" --arm A || { echo "build_queue A failed"; exit 1; }
"$PY" "$HERE/build_queue.py" --sweep_id "$SWEEP_B" --arm B || { echo "build_queue B failed"; exit 1; }

submitted=0
sub() {  # sub <arm A|B> <jobname> <extra sbatch args...>
  local arm="$1" name="$2"; shift 2
  local worker ntasks cpt jobids out id
  if [[ "$arm" == "A" ]]; then worker="$HERE/worker_8x2.slurm";  ntasks=8;  cpt=2; jobids="$JOBIDS_A"; sweep="$SWEEP_A";
  else                          worker="$HERE/worker_16x1.slurm"; ntasks=16; cpt=1; jobids="$JOBIDS_B"; sweep="$SWEEP_B"; fi
  out=$(sbatch --job-name="$name" --chdir="$PROJ" --nodes=1 --ntasks="$ntasks" --cpus-per-task="$cpt" \
        --mem-per-cpu=2G --output="$LOGDIR/${name}_%j.log" --time="$TIME" \
        --export=ALL,SWEEP_ID="$sweep" "$@" "$worker")
  id=$(grep -oP '[0-9]+$' <<< "$out"); [[ -n "$id" ]] && echo "$id" >> "$jobids"
  submitted=$((submitted+1))
  if (( submitted % 10 == 0 )); then echo "[throttle] $submitted submitted; sleep 30"; sleep 30; fi
}

# ---- EVIDENCE arena: the reservation (dynamic discovery; fail loud if absent — the A/B needs it) ----
RES=$(scontrol show reservation -o 2>/dev/null | grep -i 'Users=.*sl5nw' | grep -oP 'ReservationName=\K\S+' | head -1)
if [[ -z "${RES:-}" ]]; then
  echo "ERROR: no active sl5nw reservation found — the A/B evidence arena requires jaguar03/puma01."
  echo "Submit only the bulk arena by hand, or re-run when a reservation exists."
  exit 1
fi
echo "[reservation] using $RES"
# jaguar03: 224 cpus, but a run-3.1.1 straggler holds 16 until ~2026-07-04 -> 6 even pairs (192 cpus) so
# neither arm ever pends alone; puma01: allocatable memory 248G / 32G jobs -> 3 pairs (96 of 160 cpus).
for i in 1 2 3 4 5 6; do
  sub A graph-prune --partition=gpu --reservation="$RES" --nodelist=jaguar03 --qos=csresnolim --gpus-per-node=0
  sub B spec-decode --partition=gpu --reservation="$RES" --nodelist=jaguar03 --qos=csresnolim --gpus-per-node=0
done
for i in 1 2 3; do
  sub A kv-cache  --partition=cpu --reservation="$RES" --nodelist=puma01 --qos=csresnolim
  sub B lr-warmup --partition=cpu --reservation="$RES" --nodelist=puma01 --qos=csresnolim
done

# ---- BULK arena: open partitions (nolim SKIPPED: 80-cpu cap is DenyOnLimit and currently full) ----
for i in 1 2 3 4 5 6 7 8; do
  sub A tok-merge --partition=cpu
  sub B ffn-gate  --partition=cpu
done
GPU_ALLOW=(adriatic01 adriatic02 adriatic03 adriatic04 adriatic05 adriatic06 affogato11 affogato13 affogato14 affogato15)
# NOTE (launch 2026-07-02): these nodes have 16 physical cores (2 sockets x 8, 2 threads/core), and
# Slurm REJECTS ntasks=16 x cpus-per-task=1 there at submit time ("Requested node configuration is not
# available") unless tasks may share a core's two threads. Arm B therefore needs --ntasks-per-core=2 on
# this node class — which matches what arm B already gets on the many-core nodes (16 hardware threads on
# 8 cores), i.e. the same footprint arm A's 8x2 occupies. The 5 arm-B jobs below initially failed and
# were resubmitted by hand with --ntasks-per-core=2 (ids appended to the arm-B file).
for i in 0 1 2 3 4; do
  sub A attn-sink  --partition=gpu --nodelist="${GPU_ALLOW[$((2*i))]}"   --gpus-per-node=0
  sub B rope-scale --partition=gpu --nodelist="${GPU_ALLOW[$((2*i+1))]}" --ntasks-per-core=2 --gpus-per-node=0
done

echo "[done] submitted $submitted jobs: arm A ids in $JOBIDS_A, arm B ids in $JOBIDS_B"
echo "[done] queues: queue/$SWEEP_A/ + queue/$SWEEP_B/   data: data/$SWEEP_A/local/ + data/$SWEEP_B/local/"
