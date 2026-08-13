#!/bin/bash
# One ext96h monitoring tick: orphan requeue -> top-up -> counts + progress -> sentinel.
# Called every 20 minutes by ext96h_monitor_loop.slurm; also runnable by hand.
set -u
RUN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SWEEP_ID="${1:?usage: ext96h_monitor.sh <sweep_id>}"
Q="$RUN_DIR/queue/$SWEEP_ID"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"
export PYTHONNOUSERSITE=1

echo "=== ext96h tick $(date '+%Y-%m-%dT%H:%M:%S') sweep=$SWEEP_ID ==="

# 1. reclaim orphans and failed markers (checkpoints make every re-pend cheap)
"$PY" "$RUN_DIR/slurm/ext96h_requeue_orphans.py" --sweep_id "$SWEEP_ID"

# 2. top up the open pools + reservation while work remains
"$PY" "$RUN_DIR/slurm/ext96h_plan_jobs.py" --sweep_id "$SWEEP_ID" --submit

# 3. queue counts, per-configuration done counts, checkpoint storage
pend=$(ls "$Q/pending" 2>/dev/null | wc -l)
runn=$(ls "$Q/running" 2>/dev/null | wc -l)
done_=$(ls "$Q/done" 2>/dev/null | wc -l)
fail=$(ls "$Q/failed" 2>/dev/null | wc -l)
echo "[counts] pending=$pend running=$runn done=$done_ failed=$fail (of 900)"
for arm in run5origrnd alg2.3 gtposvel; do
  echo "  done $arm: $(ls "$Q/done" 2>/dev/null | grep -c "_${arm}_")"
done

# 4. sentinel when every run is done
if [ "$done_" -eq 900 ]; then
  echo "SWEEP96H_COMPLETE $(date '+%Y-%m-%dT%H:%M:%S')" > "$RUN_DIR/SWEEP96H_COMPLETE"
  echo "[sentinel] SWEEP96H_COMPLETE written"
fi
