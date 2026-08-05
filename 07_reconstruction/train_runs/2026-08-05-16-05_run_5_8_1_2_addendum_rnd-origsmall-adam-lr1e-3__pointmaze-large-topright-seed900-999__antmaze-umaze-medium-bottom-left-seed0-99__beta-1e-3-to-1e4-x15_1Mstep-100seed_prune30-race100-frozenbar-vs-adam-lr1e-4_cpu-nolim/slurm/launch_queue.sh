#!/bin/bash
# Prepare the Adam learning-rate 1e-3 addendum sweep: build (or reuse) the work queue, then hand
# over to plan_jobs.py, which sizes and submits the worker jobs from the live cluster state.
#
# This run goes to the cpu and nolim partitions ONLY — no gpu, no gnolim, no reservation.
#
#   fresh :  bash launch_queue.sh [tag]                  builds a new queue, prints the plan
#   refill:  SWEEP_ID=<existing id> bash launch_queue.sh  reuses the queue, prints the plan again
#   submit:  add SUBMIT=1 to either form to actually submit the printed plan
set -u
[[ "$USER" == "sl5nw" ]] || { echo "owner-only script; refusing to run as $USER"; exit 3; }
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$(dirname "$HERE")"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"
mkdir -p "$RUN_DIR/logs"

# build (fresh) or reuse (refill) the queue; the queue is never rebuilt over an existing one
if [[ -n "${SWEEP_ID:-}" ]]; then
  [[ -d "$RUN_DIR/queue/$SWEEP_ID/pending" ]] || { echo "ERROR: no queue for SWEEP_ID=$SWEEP_ID"; exit 1; }
  echo "[refill] reusing sweep $SWEEP_ID (pending=$(ls "$RUN_DIR/queue/$SWEEP_ID/pending" | wc -l))"
else
  TAG="${1:-lr1e3}"
  SWEEP_ID="$(date +%Y-%m-%d-%H-%M)_${TAG}"
  "$PY" "$HERE/build_queue.py" --sweep_id "$SWEEP_ID" || { echo "build_queue failed"; exit 1; }
fi
# the monitoring job reads the sweep id from this file, so it needs no arguments
echo "$SWEEP_ID" > "$HERE/SWEEP_ID.txt"
JOBIDS="$HERE/submitted_jobids_${SWEEP_ID}_${USER}.txt"
echo "[sweep] $SWEEP_ID   ids -> $JOBIDS"
echo

# the frozen bars must exist and be committed before anything is raced against them
[[ -f "$HERE/FROZEN_BARS.json" ]] || {
  echo "ERROR: $HERE/FROZEN_BARS.json missing — run compute_frozen_bars.py once and commit it"; exit 1; }
echo "[bars] $(sha256sum "$HERE/FROZEN_BARS.json" | cut -c1-16)...  $(
  "$PY" -c "import json,sys; d=json.load(open('$HERE/FROZEN_BARS.json'))['bars'];
print('; '.join(f\"{k.split('_start_')[0]}: {v['mean']:.4f} (weight {v['beta']}, {v['score_rule']})\" for k,v in sorted(d.items())))")"
echo

# plan_jobs.py reads the live node table, the per-user pool caps and the partition time limits, and
# emits one sbatch line per job. SUBMIT=1 makes it submit and append every id to the id file.
if [[ "${SUBMIT:-0}" == "1" ]]; then
  "$PY" "$HERE/plan_jobs.py" --sweep_id "$SWEEP_ID" --submit
  echo
  echo "[verify] sacct -j \$(head -1 \"$JOBIDS\") --format=JobID,JobName,AllocCPUS,ReqMem,NodeList,Timelimit"
  echo "         AllocCPUS must equal the job's --ntasks; if it is double, --ntasks-per-core=2 was lost."
else
  "$PY" "$HERE/plan_jobs.py" --sweep_id "$SWEEP_ID"
  echo
  echo "[dry run] nothing was submitted. Re-run with SUBMIT=1 to submit the plan above."
fi
echo
echo "[next] arm the 20-minute monitoring loop (its own 1-cpu job; it reads the sweep id from"
echo "       $HERE/SWEEP_ID.txt and its own id goes into the same id file):"
echo "       sbatch --parsable --partition=nolim --time=<partition MaxTime> --job-name=lr1e3mon \\"
echo "              $HERE/monitor_loop.slurm | cut -d';' -f1 | tee -a $JOBIDS"
