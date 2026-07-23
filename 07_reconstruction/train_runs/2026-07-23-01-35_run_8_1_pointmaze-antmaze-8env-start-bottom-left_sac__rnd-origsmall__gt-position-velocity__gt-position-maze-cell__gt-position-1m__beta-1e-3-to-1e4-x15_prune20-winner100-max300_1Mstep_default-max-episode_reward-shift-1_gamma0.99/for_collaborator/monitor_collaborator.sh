#!/bin/bash
# PASSIVE 10-minute monitoring loop for a collaborator (coordination contract).
# Start once after your first wave:  nohup bash monitor_collaborator.sh >> logs/monitor_collaborator.log 2>&1 &
# Each cycle: (1) refresh YOUR id file from squeue by job-name prefix — a Slurm requeue gives a job a
# NEW id, and the id file is your ONLY legal scancel source; (2) snapshot your jobs + queue depth;
# (3) auto-refill by re-running the launcher — it self-gates on queue depth and your caps, so calling
# it every cycle only ever ADDS jobs when useful; (4) exit when the sweep is complete.
# This loop is PASSIVE: it never renames queue markers, never prunes, never requeues, never cancels
# anything. Everything corrective is the OWNER's monitor's job — when in doubt write a problem file
# to problems/open/ and stop.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/packet_env.sh"

# the same protection as the launcher: this loop is for collaborators, not the owner
[[ "$USER" == "sl5nw" ]] && { echo "you are the sweep owner — use slurm/monitor.sh instead"; exit 3; }

while true; do
  echo "=== collab-monitor $(date '+%Y-%m-%dT%H:%M:%S') sweep=$SWEEP_ID user=$USER ==="
  # the sweep is finished: nothing left to submit or watch on this side
  if [[ -f "$COMPLETE_SENTINEL" ]]; then
    echo "SWEEP_COMPLETE exists ($COMPLETE_SENTINEL) — you are done; exiting"
    exit 0
  fi
  # refresh MY id file: append any of my ids whose job name carries one of the packet's prefixes
  # (matches requeued jobs' new ids too); one writer -> flock-guarded append
  squeue -u "$USER" -h -o "%i %j" | grep -E " $PREFIX_RE" | while read -r id name; do
    grep -qx "$id" "$IDFILE" 2>/dev/null || \
      { flock "$IDFILE.lock" bash -c "echo $id >> '$IDFILE'"; echo "[collab-monitor] appended new id $id ($name)"; }
  done
  # snapshot my jobs + the queue depth
  squeue -u "$USER" -h -o "%.10i %.9j %.9T %.5C %.16R" | grep -E "$PREFIX_RE" || echo "(none of my worker jobs in squeue)"
  echo "[queue] pending=$(ls "$RUN_DIR/queue/$SWEEP_ID/pending" 2>/dev/null | wc -l)"
  # auto-refill: the launcher exits immediately when the queue is shallow, my caps are full, or the
  # sweep completed — so calling it every cycle only ever ADDS jobs when they are useful
  bash "$HERE/launch_workers_collaborator.sh" || echo "[collab-monitor] launcher declined/failed (see above)"
  sleep 600
done
