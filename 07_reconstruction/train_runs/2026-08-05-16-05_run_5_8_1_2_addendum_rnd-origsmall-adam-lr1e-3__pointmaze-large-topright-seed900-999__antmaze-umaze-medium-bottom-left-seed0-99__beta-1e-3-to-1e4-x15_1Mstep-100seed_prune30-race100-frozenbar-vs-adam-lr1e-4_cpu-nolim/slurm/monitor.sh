#!/bin/bash
# ONE monitoring tick for this run (owner only). Called every ~20 minutes by monitor_loop.slurm, or
# by hand. Does the correctness-critical bookkeeping, tops the fleet back up to the pool caps, and
# prints the report tables of the shared sweep-monitoring skill.
#
# Order of a tick:
#   1. refresh the run's OWN job-id file (dedup, then report the states of MY ids only — never
#      enumerate jobs any other way)
#   2. requeue_orphans.py --repend_failed — reclaim walltime-killed running/ markers and re-pend
#      failed/ markers
#   3. truncation_controller.py --once — one frozen-bar pass (truncate from 30 seeds, survivor at 100)
#   4. truncation_check.py — re-verify every decision (this run replaces the sweep_prune skill's
#      cell-relative bar with a per-environment frozen bar, so the controller is verified, not trusted)
#   5. disk guard — free space on the filesystem and this sweep's record bytes
#   6. top up the fleet to the live pool caps, but only while pending/ still has work
#   7. monitoring_report.py --snapshot — one combined Markdown report
#
# All output goes to stdout AND is appended to slurm/monitor_history.log.
#   bash monitor.sh <sweep_id>
set -u
[[ "$USER" == "sl5nw" ]] || { echo "owner-only monitor"; exit 3; }
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$(dirname "$HERE")"
MON="$RUN_DIR/20_mins_monitoring"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"
SWEEP_ID="${1:?usage: monitor.sh <sweep_id>}"
JOBIDS="$HERE/submitted_jobids_${SWEEP_ID}_${USER}.txt"
LOG="$HERE/monitor_history.log"
RESERVE_GB=15
stamp=$(date '+%Y-%m-%dT%H:%M:%S')

{
  echo "==================== [tick $stamp] sweep=$SWEEP_ID ===================="
  # 1. refresh own id file: dedup in place, then report the current states of MY ids only
  if [[ -f "$JOBIDS" ]]; then
    sort -u -o "$JOBIDS" "$JOBIDS"
    ids=$(paste -sd, "$JOBIDS")
    if [[ -n "$ids" ]]; then
      echo "[jobs] $(squeue -u "$USER" -j "$ids" -h -o '%T' 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
    else
      echo "[jobs] id file empty"
    fi
  else
    echo "[jobs] no id file yet ($JOBIDS)"
  fi
  # 2. reclaim orphans and re-pend failed markers (infrastructure kills retry on fresh workers)
  "$PY" "$HERE/requeue_orphans.py" --sweep_id "$SWEEP_ID" --repend_failed 2>&1 | sed 's/^/[requeue] /'
  # 3. one frozen-bar controller pass
  "$PY" "$HERE/truncation_controller.py" --sweep_id "$SWEEP_ID" --n_required 30 --n_target 100 --once 2>&1 \
    | sed 's/^/[truncation] /'
  # 4. re-verify every decision (a violation is printed loudly; it does not abort the tick)
  "$PY" "$MON/truncation_check.py" --sweep_id "$SWEEP_ID" --n_required 30 --n_target 100 2>&1 \
    | sed 's/^/[check] /'
  # 5. disk guard
  free_gb=$(df -BG --output=avail "$RUN_DIR" | tail -1 | tr -dc '0-9')
  rec_mb=$(du -sm "$RUN_DIR/data/$SWEEP_ID" 2>/dev/null | cut -f1)
  echo "[disk] free ${free_gb}G on the filesystem; sweep records ${rec_mb:-0}M"
  if (( free_gb < RESERVE_GB )); then
    echo "[disk] LOUD WARNING: free space ${free_gb}G is below the ${RESERVE_GB}G reserve"
  fi
  # 6. top up: refill the cpu and nolim pools to their caps while there is still pending work
  pending=$(ls "$RUN_DIR/queue/$SWEEP_ID/pending" 2>/dev/null | wc -l)
  if (( pending > 0 )); then
    "$PY" "$HERE/plan_jobs.py" --sweep_id "$SWEEP_ID" --submit --jobname "lr1e3t$(date +%H%M)" 2>&1 \
      | sed 's/^/[topup] /'
  else
    echo "[topup] pending/ is empty; no new worker jobs"
  fi
  # 7. one combined Markdown report -> 20_mins_monitoring/outputs/<stamp>_monitoring.md
  "$PY" "$MON/monitoring_report.py" --sweep_id "$SWEEP_ID" --snapshot 2>&1
} 2>&1 | tee -a "$LOG"
