#!/bin/bash
# The standing monitoring loop AS A SLURM JOB (shared sweep-monitoring skill): one 1-cpu task runs
# one monitor.sh tick every INTERVAL seconds until the run's completion sentinel appears. Every path
# is absolute — sbatch spools this script, so dirname "$0" would point at the spool directory. The
# sweep id is read from the run's slurm/SWEEP_ID.txt, so this script takes no arguments.
# Partition, --time and --job-name are supplied on the sbatch command line.
#
# COMPLETION = pending/ empty AND running/ empty AND all 90 configurations decided (truncated or
# survivor in the decision log). 90 = 2 predictor learning rates x 3 environments x 15 bonus
# weights, the sweep as extended in place on 2026-08-06 (slurm/extend_queue.py).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=0
#SBATCH --mem=16G
#SBATCH --output=/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-05-16-05_run_5_8_1_2_addendum_rnd-origsmall-adam-lr1e-3__pointmaze-large-topright-seed900-999__antmaze-umaze-medium-bottom-left-seed0-99__beta-1e-3-to-1e4-x15_1Mstep-100seed_prune30-race100-frozenbar-vs-adam-lr1e-4_cpu-nolim/logs/monitor_loop_%j.log
set -u
RUN_DIR="/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-05-16-05_run_5_8_1_2_addendum_rnd-origsmall-adam-lr1e-3__pointmaze-large-topright-seed900-999__antmaze-umaze-medium-bottom-left-seed0-99__beta-1e-3-to-1e4-x15_1Mstep-100seed_prune30-race100-frozenbar-vs-adam-lr1e-4_cpu-nolim"
MONITOR="$RUN_DIR/slurm/monitor.sh"
SENTINEL="$RUN_DIR/SWEEP_COMPLETE"
SWEEP_ID="$(cat "$RUN_DIR/slurm/SWEEP_ID.txt")"
INTERVAL="${INTERVAL:-1200}"   # 20 minutes between ticks
Q="$RUN_DIR/queue/$SWEEP_ID"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"
export PYTHONNOUSERSITE=1
N_CONFIGS=90

echo "[$(date '+%Y-%m-%dT%H:%M:%S')] monitor_loop.slurm start job=${SLURM_JOB_ID:-?} node=$(hostname) sweep=$SWEEP_ID interval=${INTERVAL}s"
while true; do
  if [[ -f "$SENTINEL" ]]; then
    echo "[$(date '+%Y-%m-%dT%H:%M:%S')] SWEEP_COMPLETE present; monitor loop exiting"
    break
  fi
  bash "$MONITOR" "$SWEEP_ID"
  pend=$(ls "$Q/pending" 2>/dev/null | wc -l)
  run=$(ls "$Q/running" 2>/dev/null | wc -l)
  dec=$("$PY" - "$RUN_DIR/slurm/truncation_decisions_${SWEEP_ID}.jsonl" <<'EOF'
import json, sys, os
# count UNIQUE decided configuration keys in the decision log (0 when the log does not exist yet)
path = sys.argv[1]
keys = set()
if os.path.exists(path):
    for line in open(path):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("verdict") in ("truncated", "survivor"):
            keys.add(d["config_key"])
print(len(keys))
EOF
)
  echo "[$(date '+%Y-%m-%dT%H:%M:%S')] state: pending=$pend running=$run decided=$dec/$N_CONFIGS"
  if (( pend == 0 && run == 0 && dec >= N_CONFIGS )); then
    echo "[$(date '+%Y-%m-%dT%H:%M:%S')] SWEEP_COMPLETE (queue empty, every configuration decided); writing sentinel and exiting"
    touch "$SENTINEL"
    break
  fi
  sleep "$INTERVAL"
done
