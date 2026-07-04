#!/bin/bash
# Launch Train run 3.2.1: 32-tasks-x-1-CPU worker jobs over the seed-outermost queue + the 1-CPU
# optuna controller. Usage:
#   fresh launch :  TIME=4-00:00:00 bash launch_queue.sh [tag]
#   refill wave  :  SWEEP_ID=<existing sweep id> bash launch_queue.sh     (reuses the queue; only
#                   submits worker jobs — run when jobs hit the 4-day walltime with pending left)
#
# Job shape (every worker job): --nodes=1 --ntasks=32 --cpus-per-task=1 --ntasks-per-core=2
# --mem-per-cpu=2G (= 64G per job, 2G per worker; ai01-04 exception 1900M — those nodes have 64G
# total) --time=$TIME; worker_32x1.slurm runs `srun --wait=0` (slurm-submission.md §9).
# Order: controller first, then OPEN partitions, reserved nodes LAST (§8). Throttle: 10 submits
# then sleep 30 s (§6). User headroom (§10): at most 6 jobs on jaguar03 (leaves >= 16 CPUs).
# After the FIRST submit of this new shape, verify sacct AllocCPUS=32 before trusting capacity math.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$(dirname "$HERE")"
PROJ=/p/rlprojects/RND/07_reconstruction
LOGDIR="$RUN_DIR/logs"; mkdir -p "$LOGDIR"
PY="/u/sl5nw/.conda/envs/exploration/bin/python"
TIME="${TIME:-4-00:00:00}"

# resolve the sweep id: env override = refill wave (queue must exist); else build a fresh sweep
if [[ -n "${SWEEP_ID:-}" ]]; then
  [[ -d "$RUN_DIR/queue/$SWEEP_ID/pending" ]] || { echo "ERROR: no queue for SWEEP_ID=$SWEEP_ID"; exit 1; }
  echo "[refill] reusing sweep $SWEEP_ID ($(ls "$RUN_DIR/queue/$SWEEP_ID/pending" | wc -l) pending)"
else
  TAG="${1:-optimizer-variants-bar}"
  SWEEP_ID="$(date +%Y-%m-%d-%H-%M)_${TAG}"
  "$PY" "$HERE/build_queue.py" --sweep_id "$SWEEP_ID" || { echo "build_queue failed"; exit 1; }
fi
JOBIDS="$HERE/submitted_jobids_${SWEEP_ID}.txt"
echo "[sweep] $SWEEP_ID  TIME=$TIME  ids -> $JOBIDS"

# controller FIRST (1 CPU; must be RUNNING, not pending — never queued behind the fleet's caps)
if squeue -u "$USER" -h -n idx-build -o "%i" | grep -q .; then
  echo "[controller] already in squeue; not resubmitting"
else
  bash "$HERE/controller_submit.sh" "$SWEEP_ID" nolim
fi

submitted=0
sub() {  # sub <jobname> <mem-per-cpu> <extra sbatch args...>
  local name="$1" mem="$2"; shift 2
  local out id
  out=$(sbatch --job-name="$name" --chdir="$PROJ" --nodes=1 --ntasks=32 --cpus-per-task=1 \
        --ntasks-per-core=2 --mem-per-cpu="$mem" --output="$LOGDIR/${name}_%j.log" --time="$TIME" \
        --export=ALL,SWEEP_ID="$SWEEP_ID" "$@" "$HERE/worker_32x1.slurm")
  id=$(grep -oP '[0-9]+$' <<< "$out"); [[ -n "$id" ]] && echo "$id" >> "$JOBIDS"
  echo "[sub] $name -> ${id:-FAILED}  $*"
  submitted=$((submitted+1))
  if (( submitted % 10 == 0 )); then echo "[throttle] $submitted submitted; sleep 30"; sleep 30; fi
}

# ---- OPEN partitions first (§8) ----
# cpu: 12 jobs, no nodelist (only >=32-thread nodes fit the shape; the 64G ask self-excludes the
# 64G-total lynx08/09). 12 x 32 = 384 of the 400 cap; some pend until the 3.1.2 follow-up drains.
for i in $(seq 12); do sub sparse-moe 2G --partition=cpu; done
# gpu allowlist, one node per job (include-list, never --exclude): cheetah03 (72t) twice,
# affogato13-15 (32t whole-node, 128G), ai01-04 (32t whole-node, 64G total -> 1900M per cpu)
sub edge-distill 2G --partition=gpu --nodelist=cheetah03 --gpus-per-node=0
sub edge-distill 2G --partition=gpu --nodelist=cheetah03 --gpus-per-node=0
for node in affogato13 affogato14 affogato15; do
  sub edge-distill 2G --partition=gpu --nodelist=$node --gpus-per-node=0
done
for node in ai01 ai02 ai03 ai04; do
  sub edge-distill 1900M --partition=gpu --nodelist=$node --gpus-per-node=0
done
# gnolim: 80-cpu per-user cap -> at most 2 jobs (ai07-10 are 32t/128G; one may pend until the
# 3.1.2 follow-up's gnolim jobs drain)
sub quant-probe 2G --partition=gnolim --nodelist=ai07 --gpus-per-node=0
sub quant-probe 2G --partition=gnolim --nodelist=ai08 --gpus-per-node=0
# nolim: 80-cpu cap -> at most 2 jobs, and only if some nolim node fits 32 threads
if sinfo -p nolim -N -h -o "%c" | awk '$1>=32{ok=1} END{exit !ok}'; then
  for i in 1 2; do sub ctx-window 2G --partition=nolim; done
else
  echo "[skip] nolim: no node with >= 32 CPUs"
fi

# ---- Reserved nodes LAST (§8), reservation discovered dynamically (never hardcode) ----
RES=$(scontrol show reservation -o 2>/dev/null | grep -i 'Users=.*sl5nw' | grep -oP 'ReservationName=\K\S+' | head -1)
if [[ -n "${RES:-}" ]]; then
  echo "[reservation] using $RES"
  # jaguar03 (224t): §10 user headroom — leave >= 16 CPUs. Count CPUs my other jobs already hold
  # there, then jobs = floor((224 - 16 - held) / 32), capped at 6.
  held=$(squeue -h -u "$USER" -w jaguar03 -o "%C" | awk '{s+=$1} END{print s+0}')
  njag=$(( (224 - 16 - held) / 32 )); (( njag > 6 )) && njag=6; (( njag < 0 )) && njag=0
  echo "[jaguar03] held=$held -> submitting $njag jobs (headroom kept: $((224 - held - njag*32)) CPUs)"
  for i in $(seq $njag); do
    sub diff-prior 2G --partition=gpu --reservation="$RES" --nodelist=jaguar03 --qos=csresnolim --gpus-per-node=0
  done
  # puma01 (160t, ~246G allocatable): memory-capped at 3 x 64G jobs; 64 threads stay free anyway
  for i in 1 2 3; do
    sub seq2graph 2G --partition=cpu --reservation="$RES" --nodelist=puma01 --qos=csresnolim
  done
else
  echo "[reservation] none active — jaguar03 is NOT reserved; leaving it to the normal gpu"
  echo "  allowlist is NOT automatic here: per §10 keep sl5nw gpu usage <= 384 CPUs; add jobs by hand."
fi

echo "[done] submitted $submitted worker jobs; ids in $JOBIDS"
echo "[done] queue: queue/$SWEEP_ID/   data: data/$SWEEP_ID/local/   decisions: optuna/decisions.jsonl"
echo "[next] verify the NEW shape on the first job:  sacct -j \$(head -1 $JOBIDS) --format=JobID,JobName,AllocCPUS,ReqMem,NodeList"
echo "[next] start the monitor:  nohup bash $HERE/monitor.sh $SWEEP_ID >> $LOGDIR/monitor.log 2>&1 &"
