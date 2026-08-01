#!/bin/bash
# Run 8.1.2 monitor — ONE monitoring tick (owner-only). Called every ~20 min by monitor_loop.slurm
# (or by hand). Does the correctness-critical bookkeeping and prints the report tables of the
# sweep-monitoring skill. Does NOT submit jobs (top-up is a separate, reviewed owner step).
# Order of a tick:
#   1. refresh the run's own job-id file (dedup; report current states of MY ids only — never
#      enumerate jobs any other way)
#   2. requeue_orphans.py --repend_failed — reclaim walltime-killed running/ markers AND blanket
#      re-pend failed/ markers into their ORIGIN pool (pool-aware; run-8.1's step 2b folded in)
#   3. stage1_controller.py --once — one frozen-bar pass (prune from 30 seeds, survivor at 100)
#   4. stage1_check.py — re-verify every decision (this run replaces the sweep_prune skill's
#      cell-relative bar with a frozen bar, so the controller is verified, not trusted)
#   5. disk guard — free GB on the filesystem + record bytes of this sweep (loud below reserve)
#   6. monitoring_report.py --snapshot — ONE combined Markdown report
# All output goes to stdout AND is appended to slurm/monitor_history.log.
#   bash monitor.sh <sweep_id>
set -u
[[ "$USER" == "sl5nw" ]] || { echo "owner-only monitor"; exit 3; }
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$(dirname "$HERE")"
MON="$RUN_DIR/20_mins_monitoring"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"
SWEEP_ID="${1:?usage: monitor.sh <sweep_id>}"
JOBIDS="$HERE/submitted_jobids_${SWEEP_ID}.txt"
LOG="$HERE/monitor_history.log"
RESERVE_GB=15
stamp=$(date '+%Y-%m-%dT%H:%M:%S')

{
  echo "==================== [tick $stamp] sweep=$SWEEP_ID ===================="
  # 1. refresh own id file: dedup in place, then report current states of MY ids only
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
  # 2. reclaim orphans + pool-aware blanket re-pend of failed/ (infra kills retry on fresh workers)
  "$PY" "$HERE/requeue_orphans.py" --sweep_id "$SWEEP_ID" --repend_failed 2>&1 | sed 's/^/[requeue] /'
  # 3. one frozen-bar controller pass
  "$PY" "$HERE/stage1_controller.py" --sweep_id "$SWEEP_ID" --n_required 30 --n_target 100 --once 2>&1 | sed 's/^/[stage1] /'
  # 4. re-verify every decision (does not abort the tick on a violation; it prints loudly)
  "$PY" "$MON/stage1_check.py" --sweep_id "$SWEEP_ID" --n_required 30 --n_target 100 2>&1 | sed 's/^/[stage1_check] /'
  # 5. disk guard: free space + this sweep's record bytes (metrics only in this run, no checkpoints)
  free_gb=$(df -BG --output=avail "$RUN_DIR" | tail -1 | tr -dc '0-9')
  rec_mb=$(du -sm "$RUN_DIR/data/$SWEEP_ID" 2>/dev/null | cut -f1)
  echo "[disk] free ${free_gb}G on the filesystem; sweep records ${rec_mb:-0}M"
  if (( free_gb < RESERVE_GB )); then
    echo "[disk] LOUD WARNING: free space ${free_gb}G is below the ${RESERVE_GB}G reserve"
  fi
  # 6. ONE combined Markdown report -> 20_mins_monitoring/outputs/<stamp>_monitoring.md
  "$PY" "$MON/monitoring_report.py" --sweep_id "$SWEEP_ID" --snapshot 2>&1
} 2>&1 | tee -a "$LOG"
