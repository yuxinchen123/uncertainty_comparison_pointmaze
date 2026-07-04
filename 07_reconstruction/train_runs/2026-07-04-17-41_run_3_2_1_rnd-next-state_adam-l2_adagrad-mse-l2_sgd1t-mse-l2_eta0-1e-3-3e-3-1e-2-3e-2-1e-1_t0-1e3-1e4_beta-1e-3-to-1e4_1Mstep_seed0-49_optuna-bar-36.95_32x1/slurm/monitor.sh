#!/bin/bash
# 10-minute monitoring loop for run 3.2.1. Start in the background right after launching:
#   nohup bash monitor.sh <sweep_id> >> ../logs/monitor.log 2>&1 &
# Each cycle: (1) refresh submitted_jobids_<sweep_id>.txt with any new/requeued ids that carry this
# run's job names (the id file is the ONLY legal scancel source — never blanket-cancel), (2) snapshot
# squeue, (3) resubmit the controller if its job vanished (it is restart-safe), (4) write the
# progress snapshot. Cancellation safety: this script never cancels anything.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$(dirname "$HERE")"
PY="/u/sl5nw/.conda/envs/exploration/bin/python"
SWEEP_ID="${1:?usage: monitor.sh <sweep_id>}"
JOBIDS="$HERE/submitted_jobids_${SWEEP_ID}.txt"
NAMES="sparse-moe|edge-distill|quant-probe|ctx-window|diff-prior|seq2graph|idx-build"

while true; do
  echo "=== monitor $(date '+%Y-%m-%dT%H:%M:%S') sweep=$SWEEP_ID ==="
  # snapshot this run's jobs (matched by our codenames; ids are the source of truth)
  squeue -u "$USER" -h -o "%.10i %-14j %.10T %.5C %.20R" | grep -E "$NAMES" || echo "(no jobs in squeue)"
  # refresh the id file: append any id with our names that is not recorded yet (e.g. a requeue)
  squeue -u "$USER" -h -o "%i %j" | grep -E "$NAMES" | while read -r id name; do
    grep -qx "$id" "$JOBIDS" 2>/dev/null || { echo "$id" >> "$JOBIDS"; echo "[monitor] appended new id $id ($name)"; }
  done
  # keep the controller alive (restart-safe; controller_submit prints the pending-fallback hint)
  if ! squeue -u "$USER" -h -n idx-build -o "%i" | grep -q .; then
    echo "[monitor] controller missing — resubmitting"
    bash "$HERE/controller_submit.sh" "$SWEEP_ID" nolim
  fi
  # progress snapshot (also written to optuna/progress.txt by the controller each cycle)
  "$PY" "$HERE/progress.py" --sweep_id "$SWEEP_ID" || echo "[monitor] progress.py failed"
  sleep 600
done
