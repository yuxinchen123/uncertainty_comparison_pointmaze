#!/bin/bash
# Monitor the follow-up sweep every ~10 min (global rule: after submitting, keep the submitted-ids file
# authoritative and the run observable). Appends to logs/monitor_<sweep>.log:
#   - each cycle: our jobs' Slurm states (from the id file — the only ids this session may ever cancel)
#     and the progress script's coverage lines;
#   - "ANOMALY ..." lines when failed/ markers or stale running/ markers appear (runs keep going);
#   - "JOBS_GONE ..." + final coverage when every job has left squeue, then exits.
# Run detached (nohup setsid) so it survives the launching session. Usage: monitor_unit_ridge_1e8.sh <sweep_id>
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$(dirname "$HERE")"
PY="/u/sl5nw/.conda/envs/exploration/bin/python"
SWEEP="$1"
IDFILE="$HERE/submitted_jobids_${SWEEP}.txt"
LOG="$RUN_DIR/logs/monitor_${SWEEP}.log"
Q="$RUN_DIR/queue/$SWEEP"

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  ids="$(paste -sd, "$IDFILE")"
  njobs="$(squeue -h -j "$ids" -o "%i" 2>/dev/null | wc -l)"
  states="$(squeue -h -j "$ids" -o "%T" 2>/dev/null | sort | uniq -c | tr -s ' \n' ' ')"
  {
    echo "[$ts] jobs_in_squeue=$njobs/$(wc -l < "$IDFILE") states:${states:- none}"
    "$PY" "$HERE/progress_unit_ridge_1e8.py" "$SWEEP" 2>&1
    nfail="$(ls "$Q/failed" 2>/dev/null | wc -l)"
    if (( nfail > 0 )); then echo "[$ts] ANOMALY failed_markers=$nfail"; fi
    if (( njobs == 0 )); then echo "[$ts] JOBS_GONE — sweep jobs all left squeue; monitor exiting"; fi
  } >> "$LOG"
  (( njobs == 0 )) && exit 0
  sleep 600
done
