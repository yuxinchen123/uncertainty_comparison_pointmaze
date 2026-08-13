#!/bin/bash
# One ext4m monitoring tick: orphan requeue -> top-up -> counts + progress -> sentinel.
# Called every 20 minutes by ext4m_monitor_loop.slurm; also runnable by hand.
set -u
RUN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SWEEP_ID="${1:?usage: ext4m_monitor.sh <sweep_id>}"
Q="$RUN_DIR/queue/$SWEEP_ID"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"
export PYTHONNOUSERSITE=1

echo "=== ext4m tick $(date '+%Y-%m-%dT%H:%M:%S') sweep=$SWEEP_ID ==="

# 1. reclaim orphans and failed markers (checkpoints make every re-pend cheap)
"$PY" "$RUN_DIR/slurm/ext4m_requeue_orphans.py" --sweep_id "$SWEEP_ID"

# 2. top up the open pools + reservation while work remains
"$PY" "$RUN_DIR/slurm/ext4m_plan_jobs.py" --sweep_id "$SWEEP_ID" --submit

# 3. queue counts, per-configuration done counts, checkpoint storage
pend=$(ls "$Q/pending" 2>/dev/null | wc -l)
runn=$(ls "$Q/running" 2>/dev/null | wc -l)
done_=$(ls "$Q/done" 2>/dev/null | wc -l)
fail=$(ls "$Q/failed" 2>/dev/null | wc -l)
echo "[counts] pending=$pend running=$runn done=$done_ failed=$fail (of 900)"
for arm in run5origrnd alg2.3 gtposvel; do
  echo "  done $arm: $(ls "$Q/done" 2>/dev/null | grep -c "_${arm}_")"
done
echo "[storage] checkpoints: $(du -sh "$RUN_DIR/checkpoints/$SWEEP_ID" 2>/dev/null | cut -f1)"

# 4. sentinel when every run is done
if [ "$done_" -eq 900 ]; then
  echo "SWEEP4M_COMPLETE $(date '+%Y-%m-%dT%H:%M:%S')" > "$RUN_DIR/SWEEP4M_COMPLETE"
  echo "[sentinel] SWEEP4M_COMPLETE written"
fi
