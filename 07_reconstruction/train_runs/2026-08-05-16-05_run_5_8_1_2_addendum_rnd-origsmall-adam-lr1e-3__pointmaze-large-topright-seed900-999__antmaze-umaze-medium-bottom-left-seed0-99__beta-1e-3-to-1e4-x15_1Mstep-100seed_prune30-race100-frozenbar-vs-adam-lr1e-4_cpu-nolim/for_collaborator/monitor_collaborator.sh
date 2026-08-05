#!/bin/bash
# PASSIVE monitor for a collaborator: every 10 minutes it refreshes your own id file, prints a
# snapshot of the queue, and lets the self-gating launcher top your workers back up. It exits when
# the owner's completion sentinel appears.
#
# Passive means: it never renames a queue marker, never truncates a configuration, never requeues
# anything, never writes into the owner's files. The owner's own loop does all of that every 20
# minutes. If something looks wrong, write a problem file and stop — do not repair.
#
#   nohup bash monitor_collaborator.sh > <your log dir>/monitor.log 2>&1 &
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/packet_env.sh"
INTERVAL="${INTERVAL:-600}"

while true; do
  stamp=$(date '+%Y-%m-%dT%H:%M:%S')
  echo "==================== [collaborator tick $stamp] ===================="
  if [[ -f "$COMPLETE_SENTINEL" ]]; then
    echo "SWEEP_COMPLETE exists — the run is finished. Monitor exiting."
    break
  fi
  bash "$HERE/refresh_my_ids.sh"
  q="$RUN_DIR/queue/$SWEEP_ID"
  echo "[queue] pending=$(ls "$q/pending" 2>/dev/null | wc -l)" \
       "running=$(ls "$q/running" 2>/dev/null | wc -l)" \
       "done=$(ls "$q/done" 2>/dev/null | wc -l)" \
       "failed=$(ls "$q/failed" 2>/dev/null | wc -l)" \
       "pruned=$(ls "$q/pruned" 2>/dev/null | wc -l)"
  open_reports=$(ls "$FC/problems/open"/*.md 2>/dev/null | wc -l)
  echo "[problems] $open_reports open report(s) — the owner's loop handles them"
  # top up: the launcher is self-gating (own caps, headroom, node availability, queue depth), so
  # calling it every tick simply keeps your share of the fleet full and does nothing when it is
  bash "$HERE/launch_workers_collaborator.sh"
  sleep "$INTERVAL"
done
